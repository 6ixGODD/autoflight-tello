import telloy
from simple_pid import PID

from utils.logger import get_logger
from view import MainView
from .model.face_detector import FaceDetector

LOGGER = get_logger('main', './logs', enable_ch=True)
PID_CC = PID(0.35, 0.2, 0.2, setpoint=0, output_limits=(-50, 50))
PID_UD = PID(0.3, 0.3, 0.3, setpoint=0, output_limits=(-40, 40))
PID_FB = PID(0.3, 0.1, 0.4, setpoint=0, output_limits=(-10, 10))
view = MainView()
tello = telloy.Tello()
face_detector = FaceDetector()

if __name__ == '__main__':
    pass
