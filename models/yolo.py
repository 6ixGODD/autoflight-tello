import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from models import BaseDetectorBackend


class YoloDetectorBackend(BaseDetectorBackend[Results]):
    def __init__(
            self,
            weights_path: str,
            device: str,
            tracker: str,
            img_size: tuple = (640, 640),
            conf_thres: float = 0.25,
            iou_thres: float = 0.7,
    ) -> None:
        self.model = YOLO(weights_path)
        self.device = device
        self._conf_thres = conf_thres
        self._iou_thres = iou_thres
        self.tracker = tracker
        self._img_size = img_size

    def predict(self, data: np.ndarray, **kwargs) -> Results:
        return self.model.track(
            source=data,
            tracker=self.tracker,
            conf=self._conf_thres,
            iou=self._iou_thres,
            device=self.device,
            imgsz=self._img_size
        )[0]

    def init_model(self, **kwargs):
        self.model.predict(
            torch.zeros(1, 3, *self._img_size, device=self.device),
            conf=self._conf_thres,
            iou=self._iou_thres,
            device=self.device,
            imgsz=self._img_size
        )
