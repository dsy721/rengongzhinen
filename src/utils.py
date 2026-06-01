# TODO: 通用工具模块，提供随机种子、日志、计时与显存信息等基础能力。

from __future__ import annotations

import logging
import random
import time
from contextlib import AbstractContextManager

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


class Timer(AbstractContextManager):
    def __enter__(self):
        self._start = time.time()
        self.elapsed = 0.0
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.time() - self._start
        return False


def gpu_mem_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0
