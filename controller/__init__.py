from abc import ABC, abstractmethod
from typing import Tuple


class BaseErrorController(ABC):
    @abstractmethod
    def update(self, errors: Tuple[float, float, float]) -> Tuple[float, float, float]:
        pass

    @abstractmethod
    def reset(self):
        pass
