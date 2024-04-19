import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results


class YoloDetector:
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
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.tracker = tracker
        self.img_size = img_size
        # Warm up
        self.model.predict(
            torch.zeros(1, 3, *img_size, device=device),
            conf=conf_thres,
            iou=iou_thres,
            device=device,
            imgsz=img_size
        )

    def inference(self, frame: np.ndarray) -> Results:
        return self.model.track(
            source=frame,
            tracker=self.tracker,
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=self.device,
            imgsz=self.img_size
        )[0]


if __name__ == '__main__':
    import cv2
    cap = cv2.VideoCapture(0)
    yolo = YoloDetector(
        weights_path="../weights/yolov8n-pose.pt",
        device="cuda:0",
        tracker="../configs/botsort.yaml"
    )
    while True:
        ret, frame = cap.read()
        results = yolo.inference(frame)
        cv2.imshow("frame", results.plot())
        center_x, center_y, w, h = results.boxes.xywh[0]
        print(center_x, center_y, w, h)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

