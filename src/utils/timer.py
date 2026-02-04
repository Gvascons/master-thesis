"""Wall-clock timing context manager."""

import time
from dataclasses import dataclass, field


@dataclass
class TimerResult:
    """Stores elapsed time from a Timer context manager."""

    elapsed: float = 0.0
    _start: float = field(default=0.0, repr=False)


class Timer:
    """Context manager for wall-clock timing.

    Usage:
        timer = Timer()
        with timer:
            do_something()
        print(f"Took {timer.result.elapsed:.2f}s")
    """

    def __init__(self):
        self.result = TimerResult()

    def __enter__(self):
        self.result._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.result.elapsed = time.perf_counter() - self.result._start
