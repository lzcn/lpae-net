"""Builders that turn the ``data`` section of a config into readers and loaders.

Split layout expected under ``data.root``::

    {phase}_pos        positive tuples (required)
    {phase}_neg        fixed negative tuples (used when neg_ratio <= 0)
    {phase}_pos_fitb   FITB questions, one correct outfit per line
    {phase}_neg_fitb   FITB candidate negatives, ratio lines per question
    items.json         item id -> item key mapping per category

Negative sampling: ``neg_ratio > 0`` draws fresh RandomMix negatives every
epoch; ``neg_ratio <= 0`` uses the fixed ``{phase}_neg`` file instead.
"""

from pathlib import Path

import torch

from datasets.dataset import FITB, PairwiseOutfit, PointwiseOutfit
from datasets.datum import Datum, build_datums
from datasets.utils import load_outfit_tuples
from runner.utils import get_logger

LOGGER = get_logger("main")


def build_datum(cfg) -> Datum:
    """Build the single feature reader shared by all splits."""
    data_cfg = cfg["data"]
    item_list_fn = Path(data_cfg["root"]) / "items.json"
    feat = data_cfg["features"]
    if isinstance(feat, str):
        feat = {"reader": "TensorLMDB", "path": feat}
    (datum,) = build_datums(item_list_fn=str(item_list_fn), readers=[feat])
    return datum


def build_loader(cfg, datum: Datum, phase: str, task: str = "rank") -> torch.utils.data.DataLoader:
    """Build a DataLoader for one phase.

    Args:
        cfg: Run configuration.
        datum: Feature reader created by :func:`build_datum`.
        phase: One of ``train | valid | test``.
        task: ``rank`` (AUC / NDCG evaluation with labeled outfits) or
            ``fitb`` (fill-in-the-blank questions).
    """
    data_cfg = cfg["data"]
    root = Path(data_cfg["root"])
    split_cfg = dict(cfg.get(phase) or {})
    common = dict(datum=[datum], phase=phase)

    if task == "fitb":
        pos_data = load_outfit_tuples(root / f"{phase}_pos_fitb")
        neg_data = load_outfit_tuples(root / f"{phase}_neg_fitb")
        dataset = FITB(pos_data=pos_data, neg_data=neg_data, **common)
        choices = data_cfg.get("num_fitb_choices")
        if choices:
            assert dataset.num_answers == int(choices), (
                f"data provides {dataset.num_answers} FITB answers but num_fitb_choices={choices}"
            )
        shuffle = False
    else:
        pos_data = load_outfit_tuples(root / f"{phase}_pos")
        ratio = int(split_cfg.get("neg_ratio", 10))
        type_aware = bool(split_cfg.get("type_aware", True))
        if ratio > 0:
            neg_data, neg_mode, neg_param = None, "RandomMix", dict(ratio=ratio, type_aware=type_aware)
        else:
            neg_mode, neg_param = "Fix", {}
            neg_data = load_outfit_tuples(root / f"{phase}_neg")
        cls = PairwiseOutfit if phase == "train" else PointwiseOutfit
        dataset = cls(
            pos_data=pos_data,
            neg_data=neg_data,
            pos_mode="Fix",
            neg_mode=neg_mode,
            neg_param=neg_param,
            **common,
        )
        shuffle = phase == "train"

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 256)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )
    LOGGER.info("[%s/%s] %d samples -> %d batches", phase, type(dataset).__name__, len(dataset), len(loader))
    return loader
