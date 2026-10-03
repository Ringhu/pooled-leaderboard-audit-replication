# Stub: timeit is a no-op decorator factory; logger is stdlib logging.
import logging

logger = logging.getLogger("rcabench_platform_stub")


def timeit(*_a, **_k):
    def deco(f):
        return f
    return deco
