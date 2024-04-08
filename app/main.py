import tellopy
from simple_pid import PID
import cv2
import av
from utils.logger import get_logger
from view import MainWidget
from config import Config
from model.yolo import Detector

LOGGER = get_logger('main', './logs', enable_ch=True)
view = MainWidget()
tello = tellopy.Tello()
detector = Detector(
    weights_path=Config.YOLO_WEIGHT_PATH,
    device=Config.DEVICE,
    conf_thres=Config.CONF_THRES,
    iou_thres=Config.YOLO_IOU_THRES,
    img_size=Config.IMG_SIZE
)

