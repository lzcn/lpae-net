"""Datum: binds an item-id -> item-key mapping to a data reader."""

import json
import os
from typing import List, Optional

import torch

from datasets.reader import DataReader, get_reader


def load_json(fn: str) -> List[List[str]]:
    """Load ``items.json``: a list of lists of item keys, one list per category."""
    path = os.path.expanduser(fn)
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(c, list) for c in data):
        raise ValueError(f"Expected a list of lists in {path}, got {type(data).__name__}")
    return data


class Datum:
    """Resolves (item_id, item_type) pairs into reader keys and loads the data.

    Args:
        item_list: Item keys indexed by ``[category][item_id]``. When None the
            key is simply ``str(item_id)``.
        reader: A callable reader returning data given a key.
    """

    def __init__(self, item_list: Optional[List[List[str]]] = None, reader: DataReader = None):
        self.item_list = item_list
        self.reader = reader

    def get_key(self, item_ids, item_types, max_size: int = 0) -> List[str]:
        """Keys of valid items, repeated to length ``max(max_size, len(item_ids))``."""
        keys: List[str] = []
        target_size = max(max_size, len(list(item_ids)))
        for item, cate in zip(item_ids, item_types):
            if int(cate) == -1:
                continue
            keys.append(self.item_list[int(cate)][int(item)] if self.item_list is not None else str(item))
        while len(keys) < target_size:
            keys.append(keys[-1])
        return keys

    def get_item(self, item_id: int, item_type: int) -> torch.Tensor:
        key = self.item_list[item_type][item_id] if self.item_list is not None else str(item_id)
        return self.reader(key)

    def get_data(self, item_ids, item_types, max_size: int = 1) -> torch.Tensor:
        """Data of one outfit with shape ``(max_size, *data_shape)``."""
        keys = self.get_key(item_ids, item_types, max_size)
        return torch.stack([self.reader(key) for key in keys], dim=0)


def build_datums(item_list_fn: Optional[str], readers: List[dict]) -> List[Datum]:
    """Create one :class:`Datum` per reader config.

    Args:
        item_list_fn: Path to ``items.json``; used when the file exists.
        readers: Reader configs, each with keys ``reader`` and ``path``
            (plus optional ``transform`` / ``default``).
    """
    item_list = load_json(item_list_fn) if item_list_fn and os.path.exists(item_list_fn) else None
    datums = []
    for cfg in readers:
        cfg = dict(cfg)
        reader = get_reader(**cfg)
        datums.append(Datum(item_list=item_list, reader=reader))
    return datums
