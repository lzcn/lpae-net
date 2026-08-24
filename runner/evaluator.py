"""Evaluation loops: AUC / NDCG ranking metrics and the FITB task."""

import numpy as np
import torch
from tqdm import tqdm

from datasets import metrics as M
from datasets.metrics import UserBundleMetric
from runner.data import build_datum, build_loader
from runner.utils import get_logger

LOGGER = get_logger("main")

_METRIC_FUNCS = {"loss": "pair_rank_loss", "ndcg": "ndcg_score", "auc": "auc_score"}


@torch.no_grad()
def compute_metrics(net, loader, device, bundles=("loss", "ndcg", "auc")) -> dict:
    """Score every outfit in ``loader`` and average per-user metrics.

    Args:
        net: Model in eval mode (set here).
        loader: DataLoader over labeled outfits (``PointwiseOutfit``).
        bundles: Metric names, a subset of ``loss | ndcg | auc``.

    Returns:
        Mapping of metric name to float.
    """
    was_training = net.training
    net.eval()
    metric = UserBundleMetric({key: getattr(M, _METRIC_FUNCS[key]) for key in bundles})
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        data = batch["data"].to(device, non_blocking=True)
        uidx = batch["uidx"].to(device, non_blocking=True)
        cate = batch["cate"].to(device, non_blocking=True)
        scores = net(data, uidx, cate).flatten().tolist()
        metric.update(scores, batch["label"].tolist(), batch["uidx"].tolist())
    if was_training:
        net.train()
    return metric.compute()


def aggregate(runs: list) -> dict:
    """Mean/std of each metric over multiple runs."""
    keys = runs[0].keys()
    return {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))) for k in keys}


def log_results(tag: str, results: dict):
    for key, (mean, std) in results.items():
        LOGGER.info("%s %s: %.4f +- %.4f", tag, key, mean, std)


class Evaluator:
    """Run the test-split evaluations.

    Args:
        cfg: Run configuration.
        device: Torch device.
    """

    def __init__(self, cfg: dict, device: torch.device):
        self.cfg = cfg
        self.device = device
        self._datum = None

    @property
    def datum(self):
        # built lazily so fitb-only runs skip unused readers
        if self._datum is None:
            self._datum = build_datum(self.cfg)
        return self._datum

    def evaluate(self, net, num_runs: int = 1) -> dict:
        """AUC / NDCG on the test split.

        Random negatives are re-drawn before every run when
        ``test.neg_ratio > 0``, and results are reported as mean +- std.
        """
        from datasets.dataset import BaseOutfitData

        loader = build_loader(self.cfg, self.datum, "test", task="rank")
        num_runs = max(1, int(num_runs))
        assert isinstance(loader.dataset, BaseOutfitData)
        results = []
        for run in range(num_runs):
            loader.dataset.next(log=(run == 0))
            metrics = compute_metrics(net, loader, self.device)
            LOGGER.info("[Test run %d/%d] %s", run + 1, num_runs, metrics)
            results.append(metrics)
        summary = aggregate(results)
        log_results("Test", summary)
        return summary

    @torch.no_grad()
    def fitb(self, net, num_runs: int = 1) -> float:
        """Fill-in-the-blank accuracy over the pre-computed questions."""
        loader = build_loader(self.cfg, self.datum, "test", task="fitb")
        dataset = loader.dataset
        num_runs = max(1, int(num_runs))
        net.eval()
        accs = []
        for run in range(num_runs):
            scores = []
            for batch in tqdm(loader, desc=f"FITB [{run + 1}/{num_runs}]", leave=False):
                data = batch["data"].to(self.device, non_blocking=True)
                uidx = batch["uidx"].to(self.device, non_blocking=True)
                cate = batch["cate"].to(self.device, non_blocking=True)
                scores += net(data, uidx, cate).flatten().tolist()
            scores = np.asarray(scores).reshape((dataset.num_questions, dataset.num_answers))
            acc = float((scores.argmax(axis=1) == 0).mean())
            LOGGER.info("FITB Accuracy [%d]/[%d]: %.4f", run + 1, num_runs, acc)
            accs.append(acc)
        mean, std = float(np.mean(accs)), float(np.std(accs))
        LOGGER.info("FITB Accuracy: %.4f +- %.4f", mean, std)
        return mean
