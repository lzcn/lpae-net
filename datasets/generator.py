"""Generators that produce positive / negative outfit tuples for each epoch."""

import logging
from typing import Dict, Optional, Type

import numpy as np

from datasets import utils

_generator_registry: Dict[str, Type["Generator"]] = {}


class Generator:
    """Base class of tuple generators.

    Subclasses are registered automatically by class name and must implement
    :meth:`run`. A generator is callable: ``tuples = generator(data)``.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _generator_registry[cls.__name__] = cls

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def __call__(self, data: Optional[np.ndarray] = None) -> np.ndarray:
        return self.run(data)

    def run(self, data: Optional[np.ndarray] = None) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Fix(Generator):
    """Always return the tuples provided at initialization."""

    def __init__(self, data: np.ndarray, **kwargs):
        super().__init__()
        assert data is not None, "data must be provided."
        self.data = data

    def run(self, *args) -> np.ndarray:
        return self.data


class RandomMix(Generator):
    """Sample negative outfits by randomly mixing items of the same category.

    For an outfit :math:`\\{x_1, \\ldots, x_n\\}`, each item :math:`x_i` is
    replaced by a random item :math:`x_i^-` (of the same type when
    ``type_aware=True``), rejecting samples that duplicate a positive outfit.

    Args:
        ratio: Number of negative outfits per positive outfit.
        type_aware: Sample replacement items from the same category.
    """

    def __init__(self, ratio: int = 1, type_aware: bool = False, **kwargs):
        super().__init__()
        self.type_aware = type_aware
        self.ratio = int(ratio)

    def run(self, data: np.ndarray) -> np.ndarray:
        item_list = utils.get_item_list(data)
        num_types = utils.infer_num_type(data)
        max_items = utils.infer_max_shape(data)
        pos_uids, pos_sizes, pos_items, pos_types = utils.split_tuple(data)
        if self.type_aware:
            self.logger.info(
                "Sampling %dx outfits from %d sets: %s", self.ratio, len(item_list), list(map(len, item_list))
            )
        else:
            self.logger.info("Sampling %dx outfits from %d items", self.ratio, sum(map(len, item_list)))
        neg_uids = pos_uids.repeat(self.ratio, axis=0).reshape((-1, 1))
        neg_sizes = pos_sizes.repeat(self.ratio, axis=0).reshape((-1, 1))
        neg_types: list = []
        neg_items: list = []
        pos_set = set(map(tuple, pos_items))
        for size, item_types in zip(pos_sizes, pos_types):
            n_sampled = 0
            while n_sampled < self.ratio:
                if self.type_aware:
                    sampled_types = item_types[:size]
                else:
                    sampled_types = np.random.randint(num_types, size=size)
                sampled_items = [int(np.random.choice(item_list[int(i)])) for i in sampled_types]
                sampled_items = sampled_items[:size] + [utils.NONE_TYPE] * (max_items - size)
                if tuple(sampled_items) in pos_set:
                    continue
                if not self.type_aware:
                    # keep the category layout of a regular outfit
                    sampled_types = list(map(int, sampled_types)) + [utils.NONE_TYPE] * (max_items - size)
                    sampled_types = np.array(sampled_types, dtype=int)
                n_sampled += 1
                neg_items.append(sampled_items)
                neg_types.append(list(map(int, sampled_types)))
        neg_items = np.array(neg_items, dtype=int)
        neg_types = np.array(neg_types, dtype=int)
        return np.hstack([neg_uids, neg_sizes, neg_items, neg_types])


def get_generator(mode: Optional[str] = None, data=None, **kwargs) -> Optional[Generator]:
    """Create a generator by name; return None when ``mode`` is None."""
    if mode is None:
        return None
    if mode not in _generator_registry:
        raise ValueError(f"Generator mode {mode!r} is not supported. Supported: {sorted(_generator_registry)}")
    return _generator_registry[mode](data=data, **kwargs)
