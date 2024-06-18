import math
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from models import BaseClassifierBackend, BasePoseEstimatorBackend
from models.mobilenet.modules.keypoints import extract_keypoints, group_keypoints
from models.mobilenet.modules.load_state import load_state
from models.mobilenet.modules.pose import Pose, track_poses
from models.mobilenet.with_mobilenet import PoseEstimationWithMobileNet


class MobileNetPoseEstimatorBackend(BasePoseEstimatorBackend[np.ndarray]):
    NOSE = 0
    NECK = 1
    RIGHT_SHOULDER = 2
    RIGHT_ELBOW = 3
    RIGHT_WRIST = 4
    LEFT_SHOULDER = 5
    LEFT_ELBOW = 6
    LEFT_WRIST = 7
    RIGHT_HIP = 8
    RIGHT_KNEE = 9
    RIGHT_ANKLE = 10
    LEFT_HIP = 11
    LEFT_KNEE = 12
    LEFT_ANKLE = 13
    LEFT_EYE = 14
    RIGHT_EYE = 15
    LEFT_EAR = 16
    RIGHT_EAR = 17

    NUM_KEYPOINTS = 18

    def __init__(
            self,
            device: str = 'cuda',
            weights_path: str = 'pose.pth',
            input_size: int = 256,
            stride: int = 8,
            smooth: bool = True,
            track: bool = True
    ):
        self._device = torch.device('cuda') if device == 'cuda' and torch.cuda.is_available() else torch.device('cpu')
        self.model = PoseEstimationWithMobileNet().cuda() \
            if self._device.type == 'cuda' else PoseEstimationWithMobileNet()
        self._weights_path = weights_path
        self._input_size = input_size
        self._stride = stride
        self.__smooth = smooth
        self.__track = track

        self.previous_keypoints: np.ndarray = np.array([])
        self._pose_classifier = None

    def init_model(self):
        load_state(self.model, torch.load(self._weights_path, map_location=self._device))

    def estimate(self, data: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        image = data.copy()
        heatmaps, pafs, scale, padding = self._inference(image)

        total = 0
        all_keypoints_by_type = []
        for i in range(self.NUM_KEYPOINTS):
            total += extract_keypoints(heatmaps[:, :, i], all_keypoints_by_type, total)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)

        if len(pose_entries) == 0:
            return np.array([]), image

        all_keypoints[:, 0] = (all_keypoints[:, 0] * self._stride / 4 - padding[1]) / scale
        all_keypoints[:, 1] = (all_keypoints[:, 1] * self._stride / 4 - padding[0]) / scale

        current_poses = []
        for entry in pose_entries:
            if len(entry) == 0:
                continue
            pose_keypoints = np.ones((self.NUM_KEYPOINTS, 2), dtype=np.int32) * -1
            for i in range(self.NUM_KEYPOINTS):
                if entry[i] != -1.0:
                    pose_keypoints[i] = all_keypoints[int(entry[i])][:2].astype(np.int32)
            pose = Pose(pose_keypoints, entry[18])
            if all(pose.keypoints[i][0] != -1 for i in [0, 5, 2]):
                current_poses.append(pose)

        if not len(current_poses):
            return np.array([]), image
        track_poses(self.previous_keypoints, current_poses, smooth=self.__smooth) if self.__track else None
        self.previous_keypoints = current_poses

        current_poses[0].draw(image)
        image = cv2.addWeighted(data, 0.6, image, 0.4, 0)
        return current_poses[0].keypoints, image

    def _inference(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, List[int]]:
        h, w, _ = image.shape
        scale = self._input_size / h

        # Pre-processing
        scaled_img = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        scaled_img = (scaled_img - np.array([128, 128, 128], np.float32)) / 256.  # Normalize
        min_dims = [self._input_size, max(scaled_img.shape[1], self._input_size)]
        padded_img, pad = self._pad_width(scaled_img, self._stride, (0, 0, 0), min_dims)
        img_tensor = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        img_tensor = img_tensor.cuda() if self._device.type == 'cuda' else img_tensor

        # Inference
        stages_output = self.model(img_tensor)

        # Post-processing
        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

        return heatmaps, pafs, scale, pad

    @staticmethod
    def _pad_width(
            image: np.ndarray,
            stride: int,
            pad_value: Tuple[int, int, int],
            min_dims: List[int]
    ) -> Tuple[np.ndarray, List[int]]:
        h, w, _ = image.shape
        h = min(min_dims[0], h)
        min_dims = [math.ceil(dim / float(stride)) * stride for dim in min_dims]
        min_dims[1] = max(min_dims[1], w)
        pad = [int(math.floor((min_dims[i] - dim) / 2.0)) for i, dim in enumerate([h, w])]
        pad += [min_dims[i] - dim - pad[i] for i, dim in enumerate([h, w])]
        padded_img = cv2.copyMakeBorder(image, *pad, cv2.BORDER_CONSTANT, value=pad_value)
        return padded_img, pad

    def classify(self, keypoints: np.ndarray, **kwargs) -> Optional[int]:
        return self._pose_classifier.classify(keypoints) if self._pose_classifier else None

    @property
    def pose_classifier(self) -> 'BaseClassifierBackend[int]':
        return self._pose_classifier

    @pose_classifier.setter
    def pose_classifier(self, pose_classifier: 'BaseClassifierBackend[int]'):
        self._pose_classifier = pose_classifier

    @property
    def center_index(self) -> int:
        return self.NECK

    @property
    def key_indexes(self) -> Tuple[int, int]:
        return self.LEFT_SHOULDER, self.RIGHT_SHOULDER
