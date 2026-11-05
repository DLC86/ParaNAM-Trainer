"""Discrete learning-rate reductions with fractional optimizer-update cadence."""

import math

from torch.optim.lr_scheduler import LRScheduler


class UpdateCadenceLR(LRScheduler):
    """Apply gamma after each real-valued interval, retaining fractional progress.

    Reductions occur at ceil(k * step_size), rather than rounding the interval
    independently. Both optimizer groups follow the same counter, which is
    serialized with the interval and base rates in the scheduler state dict.
    """

    def __init__(self, optimizer, step_size, gamma=0.994, last_epoch=-1):
        if not math.isfinite(step_size) or step_size <= 0:
            raise ValueError("step_size must be positive and finite")
        if not math.isfinite(gamma) or not 0 < gamma <= 1:
            raise ValueError("gamma must be finite and in (0, 1]")
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch=last_epoch)

    def _get_closed_form_lr(self):
        reductions = math.floor(max(0, self.last_epoch) / self.step_size)
        return [rate * self.gamma**reductions for rate in self.base_lrs]

    def get_lr(self):
        return self._get_closed_form_lr()
