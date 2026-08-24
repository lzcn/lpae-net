"""Readers that map an item key to its data (e.g. pre-extracted features)."""

import io
import logging
import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import lmdb
import numpy as np
import PIL.Image
import torch

logger = logging.getLogger(__name__)


class DataReader(ABC):
    """Callable reader: ``reader(key) -> data``.

    Args:
        path: Data path whose format depends on the reader type.
        transform: Optional callable applied to the loaded data.
        default: Fallback value when the key does not exist.
    """

    def __init__(self, path: str, transform: Callable = None, default: Any = None):
        self.path = os.path.expanduser(path)
        self.transform = transform
        self.default = default

    @abstractmethod
    def load(self, key) -> Any:
        raise NotImplementedError

    def __call__(self, key):
        data = self.load(key)
        if data is None:
            if self.default is None:
                raise KeyError(f"Key {key!r} not found in {self.path}")
            return self.default
        if self.transform is not None:
            return self.transform(data)
        return data


def _open_lmdb(path: str):
    return lmdb.open(os.path.expanduser(path), readonly=True, lock=False, readahead=False, meminit=False)


class TensorLMDBReader(DataReader):
    """Float32 tensor stored as raw bytes in an LMDB environment."""

    def __init__(self, path: str, transform: Callable = None, default: Any = None):
        super().__init__(path, transform=transform, default=default)
        self._env = None

    def load(self, key: str) -> Optional[torch.Tensor]:
        if self._env is None:
            self._env = _open_lmdb(self.path)
        with self._env.begin(write=False) as txn:
            buf = txn.get(key.encode())
        if buf is None:
            return None
        feature = np.frombuffer(buf, dtype=np.float32)
        return torch.from_numpy(feature.copy())

    def __getstate__(self):
        state = self.__dict__.copy()
        # LMDB environments cannot be pickled (e.g. with DataLoader workers)
        state["_env"] = None
        return state


class ImageLMDBReader(DataReader):
    """PIL image stored (as encoded bytes) in an LMDB environment."""

    def __init__(self, path: str, transform: Callable = None, default: Any = None):
        super().__init__(path, transform=transform, default=default)
        self._env = None

    def load(self, key: str) -> Optional[PIL.Image.Image]:
        if self._env is None:
            self._env = _open_lmdb(self.path)
        with self._env.begin(write=False) as txn:
            buf = txn.get(key.encode())
        if buf is None:
            return None
        return PIL.Image.open(io.BytesIO(buf)).convert("RGB")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        return state


class TensorPKLReader(DataReader):
    """Dict-like pickle file mapping keys to numpy arrays."""

    def __init__(self, path: str, transform: Callable = None, default: Any = None):
        super().__init__(path, transform=transform, default=default)
        path = os.path.expanduser(path)
        with open(path, "rb") as f:
            self._data = pickle.load(f)

    def load(self, key: str) -> Optional[torch.Tensor]:
        if key not in self._data:
            return None
        feature = np.asarray(self._data[key], dtype=np.float32)
        return torch.from_numpy(feature.copy())


_READERS = {
    "TensorLMDB": TensorLMDBReader,
    "ImageLMDB": ImageLMDBReader,
    "TensorPKL": TensorPKLReader,
}


def get_reader(reader: str = "TensorLMDB", path: str = None, **kwargs) -> DataReader:
    """Create a reader by name.

    Args:
        reader: One of ``TensorLMDB | ImageLMDB | TensorPKL``.
        path: Path to the data (LMDB folder or pickle file).
    """
    if reader not in _READERS:
        raise ValueError(f"Unknown reader {reader!r}, must be one of {sorted(_READERS)}")
    return _READERS[reader](path, **kwargs)
