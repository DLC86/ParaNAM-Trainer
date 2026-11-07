"""A monotonic display clock, independent of worker-report delivery intervals."""

import time


class ElapsedClock:
    def __init__(self):
        self.key = None
        self.running = False
        self.base = self.started = 0.0

    def update(self, key, elapsed, running, now=None):
        now = time.monotonic() if now is None else now
        if key != self.key or running != self.running:
            self.key, self.running = key, running
            self.base, self.started = float(elapsed), now
        elif not running:
            self.base, self.started = float(elapsed), now
        return self.value(now)

    def value(self, now=None):
        now = time.monotonic() if now is None else now
        return max(0, self.base + (now - self.started if self.running else 0))
