import sys
import time

import av
import cv2
import tellopy
import torch
import ultralytics.engine.results
from simple_pid import PID

from app.configs.config import Config
from models.yolo import YoloDetector
from utils.logger import get_logger


class Controller:
    def __init__(self):
        self.LOGGER = get_logger(__name__, save_dir="./logs")
        self.LOGGER.info("Initializing Controller")
        self.CONFIG = Config()
        self.LOGGER.info("Loading YOLO")
        self.yolo = YoloDetector(
            weights_path=self.CONFIG.YOLO_WEIGHT_PATH,
            device=self.CONFIG.DEVICE,
            tracker=self.CONFIG.TRACKER_CONFIG,
            conf_thres=self.CONFIG.CONFIDENT_THRESHOLD,
            iou_thres=self.CONFIG.IOU_THRESHOLD
        )
        self.LOGGER.info("Initializing PID Controllers")
        self.pid_cc = PID(
            self.CONFIG.PID_CC_KP,
            self.CONFIG.PID_CC_KI,
            self.CONFIG.PID_CC_KD,
            setpoint=0,
            output_limits=self.CONFIG.PID_CC_OUTPUT_LIMITS
        )
        self.pid_ud = PID(
            self.CONFIG.PID_UD_KP,
            self.CONFIG.PID_UD_KI,
            self.CONFIG.PID_UD_KD,
            setpoint=0,
            output_limits=self.CONFIG.PID_UD_OUTPUT_LIMITS
        )
        self.pid_fb = PID(
            self.CONFIG.PID_FB_KP,
            self.CONFIG.PID_FB_KI,
            self.CONFIG.PID_FB_KD,
            setpoint=0,
            output_limits=self.CONFIG.PID_FB_OUTPUT_LIMITS
        )
        self.LOGGER.info("Connecting to Tello")
        try:
            self.drone = tellopy.Tello()
            self.drone.connect()
            self.drone.wait_for_connection(self.CONFIG.CONNECTION_TIMEOUT)

            self.drone.start_video()
            self.frame_provider = av.open(self.drone.get_video_stream())
            self.LOGGER.info("Skipping frames")

            self.LOGGER.info("Connected to Tello")
            self.drone.takeoff()
        except Exception as e:
            self.LOGGER.error(f"Error creating drone object: {e}")
            sys.exit(1)

        self.shutdown = False
        self.cc = 0
        self.ud = 0
        self.fb = 0

    def control_thread(self):
        self.LOGGER.info("Starting Control Thread")
        try:
            while not self.shutdown:
                time.sleep(.05)
                pass
        except KeyboardInterrupt as e:
            self.LOGGER.error(f"Keyboard Interrupt: {e}")
        except Exception as e:
            self.LOGGER.error(f"Error: {e}")
        finally:
            self.shutdown = True
            self.drone.quit()
            self.LOGGER.info("Control Thread Stopped")

    def run(self):
        self.LOGGER.info("Starting Controller")
        import threading
        control_thread = threading.Thread(target=self.control_thread)
        control_thread.start()
        try:
            while not self.shutdown:
                frames = self.frame_provider.decode(video=0)
                for i, frame in enumerate(frames):
                    if i < self.CONFIG.FRAME_SKIP or i % self.CONFIG.VIDEO_ENCODER_RATE != 0:
                        continue
                    frame = frame.to_ndarray(format='bgr24').astype('uint8')
                    frame = cv2.resize(frame, self.CONFIG.IMG_SIZE)
                    result: ultralytics.engine.results.Results = self.yolo.inference(frame)
                    if len(result.boxes.xywh) == 0:
                        cv2.imshow("Autopilot", frame)
                        continue
                    center_x, center_y, w, h = result.boxes.xywh[0]
                    area = w * h
                    error_x = center_x - self.CONFIG.IMG_SIZE[0] / 2
                    error_y = center_y - self.CONFIG.IMG_SIZE[1] / 2
                    error_size = area - (
                            self.CONFIG.IMG_SIZE[0] * self.CONFIG.IMG_SIZE[1] * self.CONFIG.ERROR_SIZE_THRESHOLD
                    )

                    self.cc = self.pid_cc(error_x) if abs(error_x) > self.CONFIG.ERROR_X_THRESHOLD else 0
                    self.ud = self.pid_ud(error_y) if abs(error_y) > self.CONFIG.ERROR_Y_THRESHOLD else 0
                    self.fb = self.pid_fb(error_size) if abs(error_size) > self.CONFIG.ERROR_SIZE_THRESHOLD else 0

                    image = result.plot()
                    cv2.putText(
                        image, f"CC: {self.cc}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                    )
                    cv2.putText(
                        image, f"UD: {self.ud}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                    )
                    cv2.putText(
                        image,
                        f"FB: {self.fb}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                    )
                    cv2.line(
                        image,
                        (center_x, center_y),
                        (self.CONFIG.IMG_SIZE[0] // 2,
                         self.CONFIG.IMG_SIZE[1] // 2),
                        (0, 255, 0),
                        2
                    )
                    cv2.imshow("Autopilot", image)
                    cv2.waitKey(1)

        except KeyboardInterrupt as e:
            self.LOGGER.error(f"Keyboard Interrupt: {e}")
        except Exception as e:
            self.LOGGER.error(f"Error: {e}")
            self.LOGGER.error(f"Traceback: {sys.exc_info()}")
        finally:
            self.shutdown = True
            self.drone.land()
            self.drone.quit()
            # control_thread.join()
            cv2.destroyAllWindows()
            self.LOGGER.info("Controller Stopped")


if __name__ == '__main__':
    controller = Controller()
    controller.run()
    sys.exit(1)
