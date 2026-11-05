"""Compact terminal-style progress updates without discarding log history."""

import codecs
import re
from pathlib import Path


def epoch_line(epoch, total, label, fraction, phase, iterations_per_second=None):
    fraction = max(0.0, min(float(fraction), 1.0))
    filled = int(24 * fraction)
    bar = "#" * filled + "." * (24 - filled)
    speed = (
        f" | {iterations_per_second:.2f} it/s"
        if iterations_per_second is not None
        else ""
    )
    return f"Epoch {epoch}/{total} | {label} | [{bar}] {fraction:6.1%}{speed} | {phase}"


class TrainingLog:
    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.lines = []
        self.current = ""
        self.carriage_return = False
        self.escape = ""

    def feed(self, text):
        for char in text:
            if self.escape:
                self.escape += char
                if re.fullmatch(r"\x1b\[[0-?]*[ -/]*[@-~]", self.escape):
                    self.escape = ""
                elif len(self.escape) > 64 or (len(self.escape) == 2 and char != "["):
                    self.escape = ""
                continue
            if char == "\x1b":
                self.escape = char
                continue
            if char == "\n":
                self.lines.append(self.current)
                self.current = ""
                self.carriage_return = False
            elif char == "\r":
                self.carriage_return = True
            else:
                if self.carriage_return:
                    self.current = ""
                    self.carriage_return = False
                self.current += char

    def read(self):
        if not self.path.exists():
            return ""
        if self.path.stat().st_size < self.offset:
            self.__init__(self.path)
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            data = stream.read()
            self.offset = stream.tell()
        self.feed(self.decoder.decode(data))
        return "\n".join([*self.lines, self.current])
