"""Outfit tuple utilities.

Tuple format (headerless CSV):
    [user_id, size, item_1, ..., item_K, type_1, ..., type_K]

    - user_id : int, user index
    - size    : int, number of valid items (<= K)
    - item    : int, index into ``items.json`` (padded with -1)
    - type    : int, category index (padded with -1)
    - K       : max outfit length, inferred as (cols - 2) // 2

The real image / feature key of an item is stored in ``items.json`` so that
tuples stay compact and negative outfits can be sampled quickly.
"""

import logging
import os
from typing import List, Optional, Tuple

import numpy as np

NONE_TYPE = -1
logger = logging.getLogger(__name__)


def _upgrade_legacy(arr: np.ndarray) -> np.ndarray:
    """Insert the ``size`` column if the legacy format (without size) is used."""
    if arr.ndim != 2:
        raise ValueError(f"tuples must be 2D, got {arr.shape}")
    n_cols = arr.shape[1]
    # new: cols = 2 + 2 * K (even). legacy: cols = 1 + 2 * K (odd)
    if n_cols % 2 == 0:
        return arr
    users = arr[:, :1]
    items, types = np.split(arr[:, 1:], 2, axis=1)
    sizes = (items != NONE_TYPE).sum(axis=1, keepdims=True)
    out = np.hstack([users, sizes, items, types])
    logger.info("Upgraded legacy tuples to [user,size,items...,types...] shape=%s", out.shape)
    return out


def load_outfit_tuples(path: str, required: bool = True) -> Optional[np.ndarray]:
    """Load outfit tuples from a CSV file into a 2D int array.

    Args:
        path: CSV file with one outfit tuple per line.
        required: When False, return None instead of raising if the file
            does not exist (some splits ship without fixed negatives).
    """
    path = os.path.expanduser(str(path))
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(path)
        logger.debug("Optional tuple file %s not found.", path)
        return None
    arr = np.atleast_2d(np.loadtxt(path, delimiter=",", dtype=int))
    arr = _upgrade_legacy(arr)
    return arr


def tuple_exists(path: str) -> bool:
    return os.path.exists(os.path.expanduser(path))


def split_tuple(tuples: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split tuples into (users, sizes, items, types)."""
    k = (tuples.shape[1] - 2) // 2
    users = tuples[:, 0]
    sizes = tuples[:, 1]
    items = tuples[:, 2 : 2 + k]
    types = tuples[:, 2 + k : 2 + 2 * k]
    return users, sizes, items, types


def infer_max_size(tuples: np.ndarray) -> int:
    """Maximum number of valid items in an outfit."""
    return int(split_tuple(tuples)[1].max())


def infer_max_shape(tuples: np.ndarray) -> int:
    """Number of item slots stored in the tuples (may be larger than max_size)."""
    return (tuples.shape[1] - 2) // 2


def infer_num_type(tuples: np.ndarray) -> int:
    """Number of distinct item categories present in the tuples."""
    types = set(split_tuple(tuples)[-1].flatten())
    if NONE_TYPE in types:
        return len(types) - 1
    return len(types)


def get_item_list(data: np.ndarray) -> List[np.ndarray]:
    """Item ids available in each category (index-aligned with type id).

    The padding category (-1), if present, is placed last.
    """
    _, _, item_ids, item_types = split_tuple(data)
    num_types = infer_num_type(data) + (1 if (item_types == NONE_TYPE).any() else 0)
    item_set = [set() for _ in range(num_types)]
    for ids, types in zip(item_ids, item_types):
        for idx, cate in zip(ids, types):
            item_set[int(cate)].add(int(idx))
    return [np.array(sorted(s)) for s in item_set]


def rearrange(items: list, types: list) -> Tuple[list, list]:
    """Move valid items to the front and pad the rest with -1."""
    new_items, new_types = [], []
    for item_id, item_type in zip(items, types):
        if item_type == NONE_TYPE:
            continue
        new_items.append(item_id)
        new_types.append(item_type)
    while len(new_items) < len(items):
        new_items.append(NONE_TYPE)
        new_types.append(NONE_TYPE)
    return new_items, new_types
