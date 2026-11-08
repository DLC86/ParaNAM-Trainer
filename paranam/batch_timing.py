"""Wall-clock batch throughput and remaining work, excluding pause downtime."""

from collections import deque
import math


class BatchTiming:
    def __init__(self):
        self.durations = deque()
        self.validation_times = deque()

    def record(self, seconds):
        if math.isfinite(seconds) and seconds > 0:
            self.durations.append(seconds)

    @property
    def seconds_per_batch(self):
        return sum(self.durations) / len(self.durations) if self.durations else None

    @property
    def iterations_per_second(self):
        return 1 / self.seconds_per_batch if self.seconds_per_batch else None

    def remaining(self, batches, validation_passes=0):
        if self.seconds_per_batch is None:
            return None
        validation = (
            sum(self.validation_times) / len(self.validation_times)
            if self.validation_times
            else 0
        )
        return (
            max(0, batches) * self.seconds_per_batch
            + max(0, validation_passes) * validation
        )
