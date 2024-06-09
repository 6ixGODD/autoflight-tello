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
from models import BaseClassifierBackend, BaseDetectorBackend, BasePoseEstimatorBackend
from utils.files import increment_path
from utils.logger import get_logger


class TelloController:
    def __init__(
            self,
            error_controller: BaseErrorController,
            pose_classifier: BaseClassifierBackend,
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
        self.pose_classifier = pose_classifier
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
            str(self._save_dir / 'record.avi'),
            cv2.VideoWriter_fourcc(*'XVID'), 30, (960, 720)
        ) if record_video else None

        # thread
        self.__control_thread = threading.Thread(target=self.__control_thread_func)
        self.__pose_thread = threading.Thread(target=self.__pose_thread_func) if pose_estimator else None

        # flag
        self.__shutdown = False
        self.__record_video = record_video

        # control
        self._keypoints = None
        self._expected_height = expected_height
        self.cc, self.ud, self.fb = 0, 0, 0
        self.cc_, self.ud_, self.fb_ = 0, 0, 0

    def run(self):
        signal.signal(signal.SIGINT, self.__shutdown_callback)
        signal.signal(signal.SIGTERM, self.__shutdown_callback)
        self._logger.info('Starting Tello controller...')
        self.__init()

        # Skip the first few frames
        self._logger.info(f'Skipping {self._frame_skip} frames...')
        for _ in range(self._frame_skip):
            self._frame_container.decode(video=0)
        self._logger.info('Tello controller running...')
        cv2.namedWindow('Tello', cv2.WINDOW_NORMAL)
        # self._tello.takeoff()
        # self._tello.up(20)
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
                    keypoints, image = self.pose_estimator.predict(frame)

                    if len(keypoints):
                        self._keypoints = keypoints
                        keypoint_center = (self._keypoints[0][0], self._keypoints[0][1])
                        center = (image.shape[1] // 2, image.shape[0] // 2)
                        # noinspection PyTypeChecker
                        image = cv2.line(image, center, keypoint_center, (255, 255, 0), 2)
                        err_cc, err_ud = keypoint_center[0] - center[0], keypoint_center[1] - center[1]
                        l_sho, r_sho = self._keypoints[5], self._keypoints[2]
                        err_fb = self.__calculate_euclidean_distance(l_sho, r_sho) - self._expected_height
                        self.cc, self.ud, self.fb = self.error_controller.update((err_cc, err_ud, err_fb))
                    else:
                        self.cc, self.ud, self.fb = 0, 0, 0
                        self.error_controller.reset()

                else:  # detector
                    bbox, image = self.detector.predict(frame)
                    if len(bbox):
                        bbox = bbox[0]
                        # noinspection PyTypeChecker
                        bbox_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                        center = (image.shape[1] // 2, image.shape[0] // 2)
                        image = cv2.line(image, center, bbox_center, (0, 255, 0), 2)
                        err_cc, err_ud = bbox_center[0] - center[0], bbox_center[1] - center[1]
                        err_fb = bbox[3] - bbox[1] - self._expected_height
                        self.cc, self.ud, self.fb = self.error_controller.update((err_cc, err_ud, err_fb))
                    else:
                        self.cc, self.ud, self.fb = 0, 0, 0
                        self.error_controller.reset()

                cv2.imshow('Tello', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.shutdown()
                if self.__record_video:
                    self._video_writer.write(image)
                __skip = True

    def __init(self):
        self.pose_estimator.init_model() if self.pose_estimator else None
        self.detector.init_model() if self.detector else None
        self.pose_classifier.init_model()
        self.error_controller.reset()
        self._tello.connect()
        self._tello.wait_for_connection(self._connection_timeout)
        self._tello.start_video()
        self._frame_container = av.open(self._tello.get_video_stream())
        self.__control_thread.start()
        self.__pose_thread.start() if self.pose_estimator else None

    def __control_thread_func(self):
        while not self.__shutdown:
            time.sleep(0.3)
            self._tello.clockwise(self.cc.__int__()) if self.cc > 0 else self._tello.counter_clockwise(
                -self.cc.__int__()
            )
            self._tello.up(self.ud.__int__()) if self.ud > 0 else self._tello.down(-self.ud.__int__())
            self._tello.forward(self.fb.__int__()) if self.fb > 0 else self._tello.backward(-self.fb.__int__())
            self.cc_, self.ud_, self.fb_ = self.cc, self.ud, self.fb

    def __pose_thread_func(self):
        while not self.__shutdown:
            if not len(self._keypoints):
                continue
            keypoints_flatten = self._keypoints.flatten()
            keypoints_flatten = keypoints_flatten.reshape(1, -1)
            if self.pose_classifier.predict(keypoints_flatten) == 2:
                self._logger.info('Recognized gesture: Land')
                self.shutdown()

    def shutdown(self):
        self._logger.info('Shutting down Tello controller...')
        cv2.destroyAllWindows()
        self._tello.land()
        self._tello.quit()
        self.__shutdown = True
        try:
            self.__control_thread.join()
            self.__pose_thread.join() if self.pose_estimator else None
        except AttributeError:
            pass
        if self.__record_video:
            self._video_writer.release()
        self._logger.info('Tello controller shutdown.')
        sys.exit(0)

    def __shutdown_callback(self, signum, frame):
        self._logger.warning(f'Received signal {signum}, frame {frame}')
        self.shutdown()

    @staticmethod
    def __calculate_euclidean_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
