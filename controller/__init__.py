from abc import ABC, abstractmethod
from typing import Tuple


class BaseErrorController(ABC):
    @abstractmethod
    def update(self, err_cc: float, err_ud: float, err_fb: float) -> Tuple[float, float, float]:
        """Update the error controller with the current error values.

        Args:
            err_cc (float): Clockwise/Counter-clockwise error.
            err_ud (float): Up/Down error.
            err_fb (float): Forward/Backward error.

        Returns:
            Tuple[float, float, float]: The updated values for clockwise/counter-clockwise, up/down, and forward/backward.
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset the error controller to its initial state.
        """
        pass
