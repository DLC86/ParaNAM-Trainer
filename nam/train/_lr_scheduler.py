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
        if not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("gamma must be finite and in [0, 1]")
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch=last_epoch)

    def _get_closed_form_lr(self):
        reductions = math.floor(max(0, self.last_epoch) / self.step_size)
        return [rate * self.gamma**reductions for rate in self.base_lrs]

    def get_lr(self):
        return self._get_closed_form_lr()


class GroupUpdateCadenceLR(UpdateCadenceLR):
    """Advance one group's clock without changing another scheduler's rates."""

    def __init__(self, optimizer, group_index, step_size, gamma=0.994, last_epoch=-1):
        if not 0 <= group_index < len(optimizer.param_groups):
            raise ValueError("Invalid optimizer group index")
        self.group_index = group_index
        super().__init__(optimizer, step_size, gamma, last_epoch)

    def _get_closed_form_lr(self):
        selected = super()._get_closed_form_lr()[self.group_index]
        return [
            selected if index == self.group_index else group["lr"]
            for index, group in enumerate(self.optimizer.param_groups)
        ]


class GroupBudgetCosineLR(LRScheduler):
    """Closed-form warmup/cosine clock for one group and a frozen step budget.

    At counter zero the LR is zero. It rises linearly to the configured base LR
    over 2% of the budget, then reaches 1% at the final counter. Fractional warmup
    boundaries avoid rounding the requested fraction for short diagnostic jobs.
    Every rate is derived from base_lrs and last_epoch, never multiplied into the
    previous rate. State dictionaries retain the budget and counter on resume.
    """

    def __init__(self, optimizer, group_index, total_steps, last_epoch=-1):
        if type(total_steps) is not int or total_steps < 1:
            raise ValueError("total_steps must be a positive integer")
        if not 0 <= group_index < len(optimizer.param_groups):
            raise ValueError("Invalid optimizer group index")
        self.group_index = group_index
        self.total_steps = total_steps
        self.warmup_steps = 0.02 * total_steps
        self.minimum_factor = 0.01
        super().__init__(optimizer, last_epoch=last_epoch)

    def _get_closed_form_lr(self):
        step = min(self.total_steps, max(0, self.last_epoch))
        if step < self.warmup_steps:
            factor = step / self.warmup_steps
        else:
            phase = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            factor = (
                self.minimum_factor
                + (1 - self.minimum_factor) * (1 + math.cos(math.pi * phase)) / 2
            )
        return [
            self.base_lrs[index] * factor if index == self.group_index else group["lr"]
            for index, group in enumerate(self.optimizer.param_groups)
        ]

    def get_lr(self):
        return self._get_closed_form_lr()
