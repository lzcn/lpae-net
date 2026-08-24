"""Runtime utilities: logging, random seeding, device selection and YAML I/O."""

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def get_logger(name: str = "main") -> logging.Logger:
    return logging.getLogger(name)


def setup_logging(log_dir, name: str, level=logging.INFO) -> Path:
    """Log to both stdout and ``<log_dir>/<name>.log``.

    Returns:
        The resolved log directory.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        force=True,
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_dir / f"{name}.log", mode="w")],
    )
    return log_dir


def set_seed(seed):
    """Seed python / numpy / torch for reproducibility; no-op when None."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device(gpus=None) -> torch.device:
    """Pick CUDA when available; ``gpus`` restricts visible CUDA devices."""
    if gpus:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", ",".join(str(g) for g in gpus))
    if torch.cuda.is_available():
        return torch.device("cuda")
    logging.getLogger("main").warning("CUDA is not available, running on CPU.")
    return torch.device("cpu")


def load_yaml(path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(obj, path):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)
