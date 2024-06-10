from typing import Tuple, Type

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from models import BaseClassifierBackend, BaseDetectorBackend, BasePoseEstimatorBackend


class YoloDetectorBackend(BaseDetectorBackend, BasePoseEstimatorBackend):
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
        self.pose_classifier = None

    def predict(self, data: np.ndarray, **kwargs) -> Tuple[Tuple[np.ndarray, np.ndarray], Type[np.ndarray]]:
        result: Results = self.model.track(
            source=data,
            tracker=self.tracker,
            conf=self._conf_thres,
            iou=self._iou_thres,
            device=self.device,
            imgsz=self._img_size
        )[0]
        return (result.keypoints.xyn.cpu().numpy(), result.boxes.xyxy.cpu().numpy()), result.plot()

    def init_model(self, **kwargs):
        self.model.predict(
            torch.zeros(1, 3, *self._img_size, device=self.device),
            conf=self._conf_thres,
            iou=self._iou_thres,
            device=self.device,
            imgsz=self._img_size
        )

    def register_pose_classifier(self, pose_classifier: BaseClassifierBackend[int]):
        self.pose_classifier = pose_classifier

    def classify_pose(self, keypoints: np.ndarray, **kwargs) -> int:
        return self.pose_classifier.predict(keypoints)
