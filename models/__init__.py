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


class BasePoseEstimatorBackend(BaseModelBackend[Tuple[np.ndarray, np.ndarray]]):
    @abstractmethod
    def predict(self, data, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def init_model(self, **kwargs):
        pass


class BaseDetectorBackend(BaseModelBackend[Tuple[np.ndarray, np.ndarray]]):
    @abstractmethod
    def predict(self, data, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
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
