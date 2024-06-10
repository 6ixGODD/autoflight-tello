import signal
import sys
import threading
import time
from typing import Optional

import av
import cv2
import numpy as np
import tellopy

from controller import BaseErrorController
from models import BaseDetectorBackend, BasePoseEstimatorBackend
from utils.files import increment_path
from utils.logger import get_logger


class TelloController:
    COLOR_LINE = (255, 255, 0)

    def __init__(
            self,
            error_controller: BaseErrorController,
            pose_estimator: Optional[BasePoseEstimatorBackend] = None,
            detector: Optional[BaseDetectorBackend] = None,

            expected_height: float = 150.0,
            connection_timeout: float = 10.0,
            frame_skip: int = 300,
            record_video: bool = False,
            save_dir: str = 'output'
    ):
        assert pose_estimator or detector, 'Either pose estimator or detector must be provided.'
        assert not (pose_estimator and detector), 'Only one of pose estimator or detector must be provided.'

        # models
        self.pose_estimator = pose_estimator
        self.detector = detector
        self.error_controller = error_controller

        # tello
        self._tello = tellopy.Tello()
        self._connection_timeout = connection_timeout
        self._frame_container = None

        # common
        self._frame_skip = frame_skip
        self._save_dir = increment_path(save_dir)
        self._logger = get_logger('TelloController', str(self._save_dir / '.log'))
        # noinspection PyUnresolvedReferences
        self._video_writer = cv2.VideoWriter(
            str(self._save_dir / 'record.avi'), cv2.VideoWriter_fourcc(*'XVID'), 30, (960, 720)
        ) if record_video else None

        # thread
        self.__control_thread = threading.Thread(target=self._control_thread_func)

        # flag
        self.__shutdown = False
        self.__record_video = record_video
        self.__enable_pose_control = self.pose_estimator.pose_classifier is not None

        # control
        self._keypoints = None
        self._expected_height = expected_height
        self.cc, self.ud, self.fb = 0, 0, 0
        self.cc_, self.ud_, self.fb_ = 0, 0, 0
        self._shutdown_queue = np.zeros(10)

    def run(self):
        signal.signal(signal.SIGINT, self._shutdown_callback)
        signal.signal(signal.SIGTERM, self._shutdown_callback)
        self._logger.info('Starting Tello controller...')
        self._init()

        # Skip the first few frames
        self._logger.info(f'Skipping {self._frame_skip} frames...')
        for _ in range(self._frame_skip):
            self._frame_container.decode(video=0)
        self._logger.info('Tello controller running...')
        cv2.namedWindow('Tello', cv2.WINDOW_NORMAL)
        self._tello.takeoff()
        self._tello.up(20)
        while not self.__shutdown:
            __skip = False
            for frame in self._frame_container.decode(video=0):
                if __skip:
                    __skip = False
                    continue
                if frame is None:
                    self._logger.warning('Frame is None.')
                    self.shutdown()
                frame = frame.to_ndarray(format='bgr24').astype('uint8')
                if self.pose_estimator:  # pose estimator
                    keypoints, image = self.pose_estimator.estimate(frame)

                    if (
                            len(keypoints) > 0 and
                            keypoints[self.pose_estimator.center_index][0] and
                            keypoints[self.pose_estimator.center_index][1]
                    ):
                        self._keypoints = keypoints
                        keypoint_center = (int(self._keypoints[self.pose_estimator.center_index][0]),
                                           int(self._keypoints[self.pose_estimator.center_index][1]))
                        center = (image.shape[1] // 2, image.shape[0] // 2)
                        # noinspection PyTypeChecker
                        image = cv2.line(image, center, keypoint_center, self.COLOR_LINE, 2)
                        err_cc, err_ud = keypoint_center[0] - center[0], keypoint_center[1] - center[1]
                        index1, index2 = self.pose_estimator.key_indexes
                        err_fb = self._calculate_euclidean_distance(
                            keypoints[index1], keypoints[index2]
                        ) - self._expected_height
                        self.cc, self.ud, self.fb = self.error_controller.update(err_cc, err_ud, err_fb)
                    else:
                        self.cc = self.ud = self.fb = 0  # reset if no keypoints
                        self.error_controller.reset()

                else:  # detector
                    bbox, image = self.detector.detect(frame)
                    if len(bbox):
                        # noinspection PyTypeChecker
                        bbox_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)  # x, y
                        center = (image.shape[1] // 2, image.shape[0] // 2)
                        image = cv2.line(image, center, bbox_center, self.COLOR_LINE, 2)
                        # distance from center
                        err_cc, err_ud, err_fb = (bbox_center[0] - center[0], bbox_center[1] - center[1],
                                                  bbox[3] - bbox[1] - self._expected_height)

                        self.cc, self.ud, self.fb = self.error_controller.update(err_cc, err_ud, err_fb)
                    else:
                        self.cc = self.ud = self.fb = 0
                        self.error_controller.reset()

                cv2.imshow('Tello', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.shutdown()
                if self.__record_video:
                    self._video_writer.write(image)
                __skip = True

            # Calculate the average of the last 10 values
            # If average > 0.5, then shutdown
            # if np.mean(self._shutdown_queue) > 0.5:
            #     self._logger.info('Shutdown command received.')
            #     self.shutdown()

    def _init(self):
        self.pose_estimator.init_model() if self.pose_estimator else None
        self.detector.init_model() if self.detector else None
        self.error_controller.reset()
        self._tello.connect()
        self._tello.wait_for_connection(self._connection_timeout)
        self._tello.start_video()
        self._frame_container = av.open(self._tello.get_video_stream())
        self.__control_thread.start()
        # self.__pose_thread.start() if self.__enable_pose_control else None

    def _control_thread_func(self):
        while not self.__shutdown:
            time.sleep(0.3)
            self._tello.clockwise(self.cc.__int__()) \
                if self.cc > 0 and self.cc != self.cc_ else self._tello.counter_clockwise(-self.cc.__int__())
            self._tello.up(self.ud.__int__()) \
                if self.ud > 0 and self.ud != self.ud_ else self._tello.down(-self.ud.__int__())
            self._tello.forward(self.fb.__int__()) \
                if self.fb > 0 and self.fb != self.fb_ else self._tello.backward(-self.fb.__int__())
            self.cc_, self.ud_, self.fb_ = self.cc, self.ud, self.fb  # Save previous values

    def _pose_classification(self):
        if len(self._keypoints):
            keypoints_flatten = self._keypoints.flatten().reshape(1, -1)
            if self.pose_estimator.classify(keypoints_flatten) == self.pose_estimator.pose_classifier.POSE_LAND:
                self._logger.debug('Pose: Land')
                self._shutdown_queue = np.roll(self._shutdown_queue, 1)
            else:
                self._shutdown_queue = np.roll(self._shutdown_queue, 0)

    def shutdown(self):
        self._logger.info('Shutting down Tello controller...')
        cv2.destroyAllWindows()
        self._tello.land()
        self._tello.quit()
        self.__shutdown = True
        try:
            self.__control_thread.join()
            # self.__pose_thread.join() if self.__enable_pose_control else None
        except AttributeError:
            pass
        if self.__record_video:
            self._video_writer.release()
        self._logger.info('Tello controller shutdown.')
        sys.exit(0)

    def _shutdown_callback(self, signum, _):
        self._logger.warning(f'Received signal {signum}')
        self.shutdown()

    @staticmethod
    def _calculate_euclidean_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
