import sys
import time

import av
import tellopy
from PyQt5.QtWidgets import QApplication
from av.container import InputContainer
from simple_pid import PID
import cv2
from queue import Queue

from app.configs.config import Config
from models.yolo import YoloDetector
from utils.logger import get_logger
from view import MainWidget


class Controller:
    def __init__(self):
        self.frame_provider = None
        self.frame_provider: InputContainer
        self.auto_pilot_flag = False
        self.capture_flag = False
        self.shutdown_flag = False

        self.LOGGER = get_logger('main', './logs', enable_ch=True)
        self.tello = tellopy.Tello()
        self.yolo_detector = YoloDetector(
            weights_path=Config.YOLO_WEIGHT_PATH,
            device=Config.DEVICE,
            conf_thres=Config.CONFIDENT_THRESHOLD,
            iou_thres=Config.IOU_THRESHOLD,
            tracker=Config.TRACKER_CONFIG,
            # img_size=Config.IMG_SIZE
        )

        self.pid_cc = PID(
            Config.PID_CC_KP,
            Config.PID_CC_KI,
            Config.PID_CC_KD,
            setpoint=0,
            output_limits=Config.PID_CC_OUTPUT_LIMITS
        )
        self.pid_ud = PID(
            Config.PID_UD_KP,
            Config.PID_UD_KI,
            Config.PID_UD_KD,
            setpoint=0,
            output_limits=Config.PID_UD_OUTPUT_LIMITS
        )
        self.pid_fb = PID(
            Config.PID_FB_KP,
            Config.PID_FB_KI,
            Config.PID_FB_KD,
            setpoint=0,
            output_limits=Config.PID_FB_OUTPUT_LIMITS
        )

        self.view = None
        self.view: MainWidget

    # Callbacks --------------------------------------------------------------------------------
    def _initial_callback(self):
        self.LOGGER.info("Connecting to Tello")
        self.tello.connect()
        try:
            self.tello.wait_for_connection(Config.CONNECTION_TIMEOUT)
            self.LOGGER.info("Tello connected")
        except Exception as e:
            self.LOGGER.error(f"Failed to connect to Tello: {e}")
            self.shutdown_flag = True
            return
        self.LOGGER.info("Starting video stream")
        self.tello.set_video_encoder_rate(Config.VIDEO_ENCODER_RATE)
        self.tello.start_video()
        self.frame_provider = av.open(tello.get_video_stream())
        self.LOGGER.info("Video stream started")
        # self.tello.subscribe(tello.EVENT_VIDEO_FRAME, self.handler)
        self.LOGGER.info("Taking off")
        self.tello.takeoff()

    def _auto_pilot_callback(self):
        self.LOGGER.info("Starting auto pilot")
        self.auto_pilot_flag = True

    def _disable_auto_pilot_callback(self):
        self.LOGGER.info("Disabling auto pilot")
        self.auto_pilot_flag = False

    def _capture_callback(self):
        self.LOGGER.info("Starting capture")
        self.capture_flag = True

    def _shutdown_callback(self):
        self.LOGGER.info("Shutting down")
        self.auto_pilot_flag = False
        self.capture_flag = False
        self.tello.land()
        self.tello.quit()
        self.shutdown_flag = True
        self.LOGGER.info("Tello disconnected")

    def controller_thread(self):
        while not self.shutdown_flag:
            if self.auto_pilot_flag:
                pass
            if self.capture_flag:
                self.LOGGER.info("Capture")
                self.capture_flag = False
            time.sleep(0.1)

    # Main function ----------------------------------------------------------------------------
    def run(self):
        app = QApplication(sys.argv)
        self.view = MainWidget(
            initial_callback=self._initial_callback,
            auto_pilot_callback=self._auto_pilot_callback,
            disable_auto_pilot_callback=self._disable_auto_pilot_callback,
            capture_callback=self._capture_callback,
            shutdown_callback=self._shutdown_callback
        )
        self.view.show()

        # cap = cv2.VideoCapture(0)
        # while cap.isOpened():
        #     success, frame = cap.read()
        #     if success:
        #         self.view.update_video_frame(frame)
        #         if cv2.waitKey(1) & 0xFF == ord("q"):
        #             break
        #     else:
        #         break

        # while not self.shutdown_flag:
        #     if self.auto_pilot_flag:
        #         pass
        #     if self.capture_flag:
        #         self.LOGGER.info("Capture")
        #         self.capture_flag = False
        #     time.sleep(0.1)  TODO: Implement in a new thread

        self.LOGGER.info("Exiting")
        sys.exit(app.exec_())


if __name__ == "__main__":
    controller = Controller()
    controller.run()
