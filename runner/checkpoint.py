"""Training bookkeeping: scalar meters and best-checkpoint management."""

from collections import defaultdict, deque
from pathlib import Path

import torch


class AverageMeters:
    """Windowed moving average over named scalars."""

    def __init__(self, win_size: int = 50):
        self._values = defaultdict(lambda: deque(maxlen=win_size))

    def update(self, scalars: dict):
        for key, value in scalars.items():
            self._values[key].append(float(value.item() if torch.is_tensor(value) else value))

    def avg(self) -> dict:
        return {k: sum(v) / len(v) for k, v in self._values.items()}

    def format(self) -> str:
        return ", ".join(f"{k}: {v:.4f}" for k, v in sorted(self.avg().items()))


class CheckpointSaver:
    """Keep the ``n_saved`` checkpoints with the highest validation score.

    Files are named ``best_model_{epoch:03d}_val_auc={score:.4f}.pt`` and store
    ``{"model": state_dict, "epoch": int, "val_auc": float}``.
    """

    def __init__(self, dirpath, n_saved: int = 5):
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        self.n_saved = n_saved
        self.history = []  # list of (score, epoch, path)

    def save(self, net, epoch: int, score: float) -> Path:
        path = self.dirpath / f"best_model_{epoch:03d}_val_auc={score:.4f}.pt"
        torch.save({"model": net.state_dict(), "epoch": epoch, "val_auc": score}, path)
        self.history.append((score, epoch, path))
        self.history.sort(key=lambda x: x[0], reverse=True)
        while len(self.history) > self.n_saved:
            _, _, worst = self.history.pop()
            worst.unlink(missing_ok=True)
        return path

    @property
    def best(self):
        """Path to the checkpoint with the highest validation score."""
        return self.history[0][2] if self.history else None
