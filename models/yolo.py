from typing import Optional, Tuple, Type

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from models import BaseClassifierBackend, BaseDetectorBackend, BasePoseEstimatorBackend


class YoloV8Backend(BaseDetectorBackend, BasePoseEstimatorBackend):
    # noinspection PyTypeChecker
    def __init__(
            self,
            weights_path: str,
            device: str,
            tracker: Optional[str] = None,
            img_size: tuple = (640, 640),
            conf_thres: float = 0.25,
            iou_thres: float = 0.7,
    ) -> None:
        self.model: YOLO = None
        self._pose_classifier: BaseClassifierBackend = None
        self.device = device
        self.tracker = tracker
        self._conf_thres = conf_thres
        self._iou_thres = iou_thres
        self._weights_path = weights_path
        self._img_size = img_size

    def detect(self, data: np.ndarray, **kwargs) -> Tuple[np.ndarray, Type[np.ndarray]]:
        result: Results = self.model.track(
            source=data,
            tracker=self.tracker,
            conf=self._conf_thres,
            iou=self._iou_thres,
            device=self.device,
            imgsz=self._img_size
        )[0] if self.tracker else self.model.predict(
            data,
            conf=self._conf_thres,
            iou=self._iou_thres,
            imgsz=self._img_size
        )[0]

        return result.boxes.xyxy.cpu().numpy()[0], result.plot()

    def estimate(self, data: np.ndarray, **kwargs) -> Tuple[np.ndarray, Type[np.ndarray]]:
        results: Results = self.model.track(
            source=data,
            tracker=self.tracker,
            conf=self._conf_thres,
            iou=self._iou_thres,
            device=self.device,
            imgsz=self._img_size
        )[0] if self.tracker else self.model.predict(
            data,
            conf=self._conf_thres,
            iou=self._iou_thres,
            device=self.device,
            imgsz=self._img_size
        )[0]

        return results.keypoints.xy.cpu().numpy()[0], results.plot()

    def init_model(self, **kwargs):
        # lazy loading
        self.model = YOLO(self._weights_path)
        self.model.predict(
            torch.zeros(1, 3, *self._img_size, device=self.device),
            conf=self._conf_thres,
            iou=self._iou_thres,
            device=self.device,
            imgsz=self._img_size
        )

    def classify(self, keypoints: np.ndarray, **kwargs) -> Optional[int]:
        return self._pose_classifier.classify(keypoints) if self._pose_classifier else None

    @property
    def pose_classifier(self) -> Optional[BaseClassifierBackend[int]]:
        return self._pose_classifier

    @pose_classifier.setter
    def pose_classifier(self, pose_classifier: BaseClassifierBackend[int]):
        self._pose_classifier = pose_classifier

    @property
    def center_index(self) -> int:
        return self.NOSE

    @property
    def key_indexes(self) -> Tuple[int, int]:
        return self.LEFT_WRIST, self.RIGHT_WRIST


if __name__ == '__main__':
    model = YoloV8Backend(
        weights_path="weights/yolov8n-pose.pt",
        device='cuda',
        img_size=(640, 640),
        conf_thres=0.25,
        iou_thres=0.7
    )
    model.init_model()
    import cv2

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video device")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 640))
            kp, frame = model.estimate(frame)
            print("Length:", kp.shape[0])
            print("Predicted:", kp)
            print("Center:", kp[model.center_index][0], kp[model.center_index][1])
            cv2.imshow('Pose Classifier', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
