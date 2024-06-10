from typing import Tuple

from simple_pid import PID

from controller import BaseErrorController


class PIDController(BaseErrorController):
    def __init__(
            self,
            cc_params: Tuple[float, float, float],
            cc_limits: Tuple[float, float],
            ud_params: Tuple[float, float, float],
            ud_limits: Tuple[float, float],
            fb_params: Tuple[float, float, float],
            fb_limits: Tuple[float, float]
    ):
        self.cc_pid, self.ud_pid, self.fb_pid = (
            PID(*cc_params, output_limits=cc_limits), PID(*ud_params, output_limits=ud_limits),
            PID(*fb_params, output_limits=fb_limits)
        )

    def update(self, err_cc: float, err_ud: float, err_fb: float) -> Tuple[float, float, float]:
        return self.cc_pid(err_cc), self.ud_pid(err_ud), self.fb_pid(err_fb)

    def reset(self):
        self.cc_pid.reset()
        self.ud_pid.reset()
        self.fb_pid.reset()
