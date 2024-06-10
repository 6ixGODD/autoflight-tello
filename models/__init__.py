from abc import ABC, abstractmethod
from typing import Generic, Tuple, TypeVar

import numpy as np

T_co = TypeVar('T_co', covariant=True)


class BaseModelBackend(ABC, Generic[T_co]):
    @abstractmethod
    def predict(self, data, **kwargs) -> T_co:
        pass

    @abstractmethod
    def init_model(self, **kwargs):
        pass


POSE_CAPTURE = 1
POSE_LAND = 2


class BasePoseEstimatorBackend(BaseModelBackend[Tuple[T_co, np.ndarray]]):
    @abstractmethod
    def predict(self, data, **kwargs) -> Tuple[T_co, np.ndarray]:
        pass

    @abstractmethod
    def init_model(self, **kwargs):
        pass

    @abstractmethod
    def register_pose_classifier(self, pose_classifier: 'BaseClassifierBackend[int]'):
        pass

    @abstractmethod
    def classify_pose(self, keypoints: np.ndarray, **kwargs) -> int:
        pass

    @property
    @abstractmethod
    def is_pose_classifier_registered(self) -> bool:
        pass


class BaseDetectorBackend(BaseModelBackend[Tuple[T_co, np.ndarray]]):
    @abstractmethod
    def predict(self, data, **kwargs) -> Tuple[T_co, np.ndarray]:
        pass

    @abstractmethod
    def init_model(self, **kwargs):
        pass


class BaseClassifierBackend(BaseModelBackend[T_co]):
    @abstractmethod
    def predict(self, data, **kwargs) -> T_co:
        pass

    @abstractmethod
    def init_model(self, **kwargs):
        pass
