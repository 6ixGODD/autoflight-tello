from abc import ABC, abstractmethod
from typing import Generic, List, Tuple, TypeVar

import numpy as np

T = TypeVar('T')

T_co = TypeVar('T_co', covariant=True)


class BaseModelBackend(ABC, Generic[T]):
    @abstractmethod
    def predict(self, data, **kwargs) -> T:
        pass

    @abstractmethod
    def init_model(self, **kwargs):
        pass


class BasePoseEstimatorBackend(BaseModelBackend[Tuple[List[T_co], np.ndarray]]):
    @abstractmethod
    def predict(self, data, **kwargs) -> Tuple[List[T_co], np.ndarray]:
        pass

    @abstractmethod
    def init_model(self, **kwargs):
        pass


class BaseDetectorBackend(BaseModelBackend[T_co]):
    @abstractmethod
    def predict(self, data, **kwargs) -> T_co:
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
