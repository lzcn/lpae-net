"""Ranking metrics computed over per-user score/label collections."""

from collections import defaultdict
from typing import Callable, Dict, List

import numpy as np
import sklearn.metrics


def to_canonical(posi, nega):
    """Convert (positive scores, negative scores) to (y_true, y_score)."""
    posi, nega = np.asarray(posi), np.asarray(nega)
    y_true = np.array([1] * len(posi) + [0] * len(nega))
    y_score = np.hstack((posi.flatten(), nega.flatten()))
    return y_true, y_score


def pair_accuracy(posi, nega) -> float:
    """Fraction of (positive, negative) pairs ranked in the correct order."""
    posi, nega = np.asarray(posi), np.asarray(nega)
    diff = posi.reshape(-1, 1) - nega.reshape(1, -1)
    return float((diff > 0).sum() / diff.size)


def pair_rank_loss(posi, nega) -> float:
    """Pairwise soft-margin (logistic) ranking loss."""
    posi, nega = np.asarray(posi), np.asarray(nega)
    diff = posi.reshape(-1, 1) - nega.reshape(1, -1)
    return float(np.log(1.0 + np.exp(-diff)).sum() / diff.size)


def auc_score(posi, nega) -> float:
    """Area under the ROC curve over positive/negative outfits."""
    y_true, y_score = to_canonical(posi, nega)
    return float(sklearn.metrics.roc_auc_score(y_true, y_score))


def ndcg_score(posi, nega) -> float:
    """Mean normalized discounted cumulative gain over all ranks."""
    y_label, y_score = to_canonical(posi, nega)
    return float(_ndcg_score(y_score, y_label).mean())


def _ndcg_score(y_score, y_label, wtype: str = "max") -> np.ndarray:
    """NDCG at every rank position.

    References:
        - Hu Y, Yi X, Davis L S. Collaborative fashion recommendation: A
          functional tensor factorization approach. ACM MM 2015.
        - Lee C P, Lin C J. Large-scale Linear RankSVM. Neural computation 2014.
    """
    y_score = np.asarray(y_score).reshape(-1)
    y_label = np.asarray(y_label).reshape(-1)
    order = np.argsort(-y_score)
    p_label = np.take(y_label, order)
    i_label = np.sort(y_label)[::-1]
    p_gain = 2**p_label - 1
    i_gain = 2**i_label - 1
    if wtype.lower() == "max":
        discounts = np.log2(np.maximum(np.arange(len(y_label)) + 1, 2.0))
    else:
        discounts = np.log2(np.arange(len(y_label)) + 2)
    dcg = (p_gain / discounts).cumsum()
    idcg = (i_gain / discounts).cumsum()
    return dcg / idcg


class UserBundleMetric:
    """Accumulates scores/labels per user and averages metrics across users.

    Args:
        bundles: Mapping of metric name to callable ``(posi, nega) -> float``.

    Example::

        metric = UserBundleMetric(dict(auc=auc_score, ndcg=ndcg_score))
        for batch in loader:
            metric.update(scores.tolist(), labels.tolist(), uidxs.tolist())
        results = metric.compute()  # {"auc": ..., "ndcg": ...}
    """

    def __init__(self, bundles: Dict[str, Callable[[List[float], List[float]], float]]):
        self.bundles = bundles
        self._scores = defaultdict(list)
        self._labels = defaultdict(list)

    def reset(self):
        self._scores = defaultdict(list)
        self._labels = defaultdict(list)

    def update(self, scores, labels, uidxs):
        for x, y, u in zip(scores, labels, uidxs):
            self._scores[u].append(x)
            self._labels[u].append(y)

    def compute(self) -> Dict[str, float]:
        assert self._scores, "UserBundleMetric must receive at least one batch."
        results = defaultdict(list)
        for u in self._scores:
            labels = np.array(self._labels[u]).flatten()
            scores = np.array(self._scores[u]).flatten()
            pos = np.where(labels == 1)[0]
            neg = np.where(labels == 0)[0]
            for key, func in self.bundles.items():
                results[key].append(func(scores[pos], scores[neg]))
        return {k: float(np.mean(v)) for k, v in results.items()}
