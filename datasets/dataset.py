"""Torch datasets over outfit tuples.

Datasets hold positive and negative tuples; *generators* control how the
tuples are (re)created at the beginning of every epoch via :meth:`next`.

- :class:`PairwiseOutfit`: one positive + one negative outfit per sample
  (used for training with the pairwise ranking loss).
- :class:`PointwiseOutfit`: positives and negatives merged into a labeled list
  (used for AUC / NDCG evaluation).
- :class:`FITB`: fill-in-the-blank questions, ``num_answers`` outfits per
  question where the first one is the correct answer.
"""

import logging
from typing import List, Optional

import numpy as np
import torch

from datasets import utils
from datasets.datum import Datum
from datasets.generator import Generator, get_generator

_DATASETS = {}


class BaseOutfitData(torch.utils.data.Dataset):
    """Base dataset holding positive/negative tuples and their generators.

    Args:
        datum: Data readers used to load item features.
        pos_data: Positive outfit tuples.
        neg_data: Optional fixed negative outfit tuples.
        pos_mode: Name of the generator for positive tuples ("Fix" or None).
        neg_mode: Name of the generator for negative tuples (e.g. "RandomMix").
        pos_param: Keyword arguments for the positive generator.
        neg_param: Keyword arguments for the negative generator.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _DATASETS[cls.__name__] = cls

    def __init__(
        self,
        datum: List[Datum],
        pos_data: np.ndarray,
        neg_data: np.ndarray = None,
        phase: str = "train",
        pos_mode: Optional[str] = "Fix",
        neg_mode: Optional[str] = None,
        pos_param: dict = None,
        neg_param: dict = None,
    ):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.datum = datum
        self.phase = phase
        self.ini_pos = pos_data
        self.ini_neg = neg_data
        self.pos_data = pos_data
        self.neg_data = neg_data
        self.max_size = utils.infer_max_size(pos_data)
        self.pos_generator: Optional[Generator] = get_generator(pos_mode, data=pos_data, **(pos_param or {}))
        self.neg_generator: Optional[Generator] = get_generator(neg_mode, data=neg_data, **(neg_param or {}))
        self.next(log=False)

    def next(self, log: bool = True):
        """Regenerate tuples for the next epoch."""
        if self.pos_generator is not None:
            self.pos_data = self.pos_generator(self.ini_pos)
        if self.neg_generator is not None:
            self.neg_data = self.neg_generator(self.pos_data)
        if log:
            if self.pos_data is not None:
                self.logger.info("[%s] positive tuples ready: shape=%s", self.phase, self.pos_data.shape)
            if self.neg_data is not None:
                self.logger.info("[%s] negative tuples ready: shape=%s", self.phase, self.neg_data.shape)
        self.process()

    def process(self):
        raise NotImplementedError

    def __getitem__(self, n):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class PairwiseOutfit(BaseOutfitData):
    """Yields ``(pos_outfit, neg_outfit)`` pairs of the same user."""

    def process(self):
        pos_uidx, pos_sizes, pos_items, pos_types = utils.split_tuple(self.pos_data)
        neg_uidx, neg_sizes, neg_items, neg_types = utils.split_tuple(self.neg_data)
        ratio = len(self.neg_data) // len(pos_uidx)
        assert ratio * len(pos_uidx) == len(self.neg_data), "negatives must be a multiple of positives"
        assert (pos_uidx.repeat(ratio, axis=0) == neg_uidx).all(), "positive/negative user mismatch"
        self.uidxs = neg_uidx
        # positive data repeated to align with negatives
        self.pos_sizes = pos_sizes.repeat(ratio, axis=0)
        self.pos_items = pos_items.repeat(ratio, axis=0)
        self.pos_types = pos_types.repeat(ratio, axis=0)
        self.neg_sizes = neg_sizes
        self.neg_items = neg_items
        self.neg_types = neg_types
        self.max_size = int(max(pos_sizes.max(), neg_sizes.max()))

    def __getitem__(self, n):
        pos_items, pos_types = self.pos_items[n], self.pos_types[n]
        neg_items, neg_types = self.neg_items[n], self.neg_types[n]
        data = torch.stack(
            [
                self.datum[0].get_data(pos_items, pos_types, self.max_size),
                self.datum[0].get_data(neg_items, neg_types, self.max_size),
            ],
            dim=0,
        )
        cate = torch.stack([torch.tensor(list(map(int, pos_types))), torch.tensor(list(map(int, neg_types)))], dim=0)
        size = torch.tensor([int(self.pos_sizes[n]), int(self.neg_sizes[n])])
        return dict(data=data, uidx=int(self.uidxs[n]), cate=cate, size=size)

    def __len__(self):
        return len(self.uidxs)


class PointwiseOutfit(BaseOutfitData):
    """Merges positives and negatives into labeled outfits (1 / 0)."""

    def process(self):
        tuples = np.vstack((self.pos_data, self.neg_data))
        self.uidxs, self.sizes, self.items, self.types = utils.split_tuple(tuples)
        self.labels = np.array([1] * len(self.pos_data) + [0] * len(self.neg_data), dtype=np.int64)
        self.max_size = int(self.sizes.max())

    def __getitem__(self, n):
        items, types = self.items[n], self.types[n]
        data = self.datum[0].get_data(items, types, self.max_size)
        cate = torch.tensor(list(map(int, types)))
        return dict(
            data=data,
            uidx=int(self.uidxs[n]),
            cate=cate,
            size=int(self.sizes[n]),
            label=int(self.labels[n]),
        )

    def __len__(self):
        return len(self.uidxs)


class FITB(BaseOutfitData):
    """Fill-in-the-blank evaluation.

    ``pos_data`` holds one correct outfit per question and ``neg_data`` holds
    ``ratio`` candidate replacements per question. Outfits are ordered so that
    for question ``i`` its answers occupy rows ``i * num_answers`` to
    ``(i + 1) * num_answers - 1``, with the correct answer first.
    """

    def process(self):
        num_questions = len(self.pos_data)
        num_negatives = len(self.neg_data) // num_questions
        num_answers = num_negatives + 1
        assert num_negatives * num_questions == len(self.neg_data)
        self.num_questions = num_questions
        self.num_answers = num_answers
        self.logger.info("[%s] %d FITB questions with %d answers each", self.phase, num_questions, num_answers)
        pos_data = self.pos_data.reshape((num_questions, 1, -1))
        neg_data = self.neg_data.reshape((num_questions, num_negatives, -1))
        outfits = np.concatenate((pos_data, neg_data), axis=1).reshape((num_questions * num_answers, -1))
        pos_label = np.ones((num_questions, 1), dtype=np.int64)
        neg_label = np.zeros((num_questions, num_negatives), dtype=np.int64)
        self.labels = np.hstack((pos_label, neg_label)).reshape(-1)
        self.uidxs, self.sizes, self.items, self.types = utils.split_tuple(outfits)
        self.max_size = int(self.sizes.max())

    def __getitem__(self, n):
        items, types = self.items[n], self.types[n]
        data = self.datum[0].get_data(items, types, self.max_size)
        cate = torch.tensor(list(map(int, types)))
        return dict(
            data=data,
            uidx=int(self.uidxs[n]),
            cate=cate,
            size=int(self.sizes[n]),
            label=int(self.labels[n]),
        )

    def __len__(self):
        return len(self.uidxs)


def get_dataset(name: str, **kwargs) -> BaseOutfitData:
    """Create a dataset by class name."""
    if name not in _DATASETS:
        raise ValueError(f"Unknown dataset {name!r}, must be one of {sorted(_DATASETS)}")
    return _DATASETS[name](**kwargs)
