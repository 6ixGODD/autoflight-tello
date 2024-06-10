import math
from typing import List, Tuple

import cv2
import numpy as np
import torch

from models import BasePoseEstimatorBackend, BaseClassifierBackend
from models.mobilenet.modules.keypoints import extract_keypoints, group_keypoints
from models.mobilenet.modules.load_state import load_state
from models.mobilenet.modules.pose import Pose, track_poses
from models.mobilenet.with_mobilenet import PoseEstimationWithMobileNet


class MobileNetPoseEstimatorBackend(BasePoseEstimatorBackend[np.ndarray]):
    def __init__(
            self,
            device: str = 'cuda',
            weights_path: str = 'pose.pth',
            input_size: int = 256,
            stride: int = 8,
            nums_keypoints: int = 18,
            smooth: bool = True,
            track: bool = True
    ):
        self._device = torch.device('cuda') if device == 'cuda' and torch.cuda.is_available() else torch.device('cpu')
        self.model = PoseEstimationWithMobileNet().cuda() if self._device.type == 'cuda' else PoseEstimationWithMobileNet()
        self._weights_path = weights_path
        self._input_size = input_size
        self._stride = stride
        self._nums_keypoints = nums_keypoints
        self.__smooth = smooth
        self.__track = track

        self.previous_keypoints: np.ndarray = np.array([])
        self.pose_classifier = None

    def init_model(self):
        load_state(self.model, torch.load(self._weights_path, map_location=self._device))

    def predict(self, data: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        image = data.copy()
        heatmaps, pafs, scale, padding = self.__inference(image)

        total = 0
        all_keypoints_by_type = []
        for i in range(self._nums_keypoints):
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
            pose_keypoints = np.ones((self._nums_keypoints, 2), dtype=np.int32) * -1
            for i in range(self._nums_keypoints):
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

    def __inference(self, images: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, List[int]]:
        h, w, _ = images.shape
        scale = self._input_size / h

        scaled_img = cv2.resize(images, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        # Normalize
        scaled_img = (scaled_img - 128) / 256.
        min_dims = [self._input_size, max(scaled_img.shape[1], self._input_size)]
        padded_img, pad = self.__pad_width(scaled_img, self._stride, (0, 0, 0), min_dims)
        img_tensor = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        img_tensor = img_tensor.cuda() if self._device.type == 'cuda' else img_tensor
        stages_output = self.model(img_tensor)

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

        return heatmaps, pafs, scale, pad

    @staticmethod
    def __pad_width(
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

    def register_pose_classifier(self, pose_classifier: 'BaseClassifierBackend[int]'):
        self.pose_classifier = pose_classifier

    def classify_pose(self, keypoints: np.ndarray, **kwargs) -> int:
        return self.pose_classifier.predict(keypoints)

    @property
    def is_pose_classifier_registered(self) -> bool:
        return self.pose_classifier is not None
