from abc import ABC, abstractmethod
from typing import Generic, Tuple, TypeVar

import numpy as np

T_co = TypeVar('T_co', covariant=True)


class BaseModelBackend(ABC):
    @abstractmethod
    def init_model(self, **kwargs):
        pass


class BasePoseEstimatorBackend(BaseModelBackend, Generic[T_co]):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    @abstractmethod
    def estimate(self, data: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def init_model(self, **kwargs):
        pass

    @abstractmethod
    def classify(self, keypoints: np.ndarray, **kwargs) -> T_co:
        pass

    @property
    @abstractmethod
    def pose_classifier(self) -> 'BaseClassifierBackend[T_co]':
        pass

    @pose_classifier.setter
    @abstractmethod
    def pose_classifier(self, pose_classifier: 'BaseClassifierBackend[T_co]'):
        pass

    @property
    @abstractmethod
    def center_index(self) -> int:
        pass

    @property
    @abstractmethod
    def key_indexes(self) -> Tuple[int, int]:
        pass


class BaseDetectorBackend(BaseModelBackend):
    @abstractmethod
    def detect(self, data: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def init_model(self, **kwargs):
        pass


class BaseClassifierBackend(BaseModelBackend, Generic[T_co]):
    NEGATIVE = 0
    POSE_CAPTURE = 1
    POSE_LAND = 2

    @abstractmethod
    def classify(self, data: np.ndarray, **kwargs) -> T_co:
        pass

    @abstractmethod
    def init_model(self, **kwargs):
        pass
