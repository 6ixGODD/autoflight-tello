from ultralytics import YOLO
from ultralytics.engine.results import Results

class Detector:
    def __init__(
            self,
            weights_path: str,
            device: str,
            img_size: tuple = (640, 640),
            conf_thres: float = 0.25,
            iou_thres: float = 0.7,
            track: bool = False
    ) -> None:
        self.model = YOLO(weights_path)
        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.track = track
        self.tracker = None
        self.model.eval()
        # Warm up
        self.model(
            torch.zeros(1, 3, *img_size, device=device),
            conf=conf_thres,
            iou=iou_thres,
            device=device,
            imgsz=img_size
        )

    def enable_tracking(self, tracking_config: str) -> None:
        """ Enable tracking for the detector
        Args:
            tracking_config: yaml file path for the tracking configuration

        Returns:
            None

        """
        self.track = True
        self.tracker = tracking_config

    def disable_tracking(self) -> None:
        """ Disable tracking for the detector
        Returns:
            None

        """
        self.track = False
        self.tracker = None

    def inference(self, frame) -> Results:
        """ Perform inference on the frame
        Args:
            frame: input frame

        Returns:
            Results: detection results, ultralytics.engine.results.Results object

        """
        if self.track:
            return self.model.track(
                source=frame,
                track=self.tracker,
                conf=self.conf_thres,
                iou=self.iou_thres,
                device=self.device,
                imgsz=self.img_size
            )[0]
        else:
            return self.model.predict(
                source=frame,
                conf=self.conf_thres,
                iou=self.iou_thres,
                device=self.device,
                imgsz=self.img_size
            )[0]
