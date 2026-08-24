"""Compatibility training loop (pairwise ranking objective)."""

import time
from pathlib import Path

import torch

from runner.checkpoint import AverageMeters, CheckpointSaver
from runner.data import build_datum, build_loader
from runner.model import build_net, build_optimizer
from runner.utils import get_logger

LOGGER = get_logger("main")


def gather_losses(losses: dict, loss_weight: dict):
    """Weighted sum of per-term batch-mean losses plus their plain means.

    Returns:
        ``(total_loss, {name: float_mean})`` for logging.
    """
    stats, total = {}, 0.0
    for name, value in losses.items():
        value = value.mean()
        weight = float(loss_weight.get(name, 0.0))
        if weight:
            total = total + value * weight
        stats[name] = value.item()
    return total, stats


class Trainer:
    """Train a compatibility model and keep the best checkpoints.

    Args:
        cfg: Run configuration.
        device: Torch device.
        log_dir: Output folder for checkpoints and TensorBoard events.
    """

    def __init__(self, cfg: dict, device: torch.device, log_dir):
        self.cfg = cfg
        self.device = device
        self.log_dir = log_dir
        self.epochs = int(cfg.get("epochs", 200))
        self.grad_clip = float(cfg.get("grad_clip", 0.5))
        self.display_interval = int(cfg.get("display_interval", 50))
        self.summary_interval = int(cfg.get("summary_interval", 200))
        self.loss_weight = dict(cfg["net"].get("loss_weight") or {})

        self.datum = build_datum(cfg)
        self.net = build_net(cfg, device)
        self.optimizer, self.scheduler = build_optimizer(cfg, self.net)
        self.train_loader = build_loader(cfg, self.datum, "train")
        self.valid_loader = build_loader(cfg, self.datum, "valid", task="rank")

        from runner.evaluator import compute_metrics

        self._compute_metrics = compute_metrics
        self.saver = CheckpointSaver(Path(log_dir) / "checkpoints")
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=str(log_dir), flush_secs=10)
        except ImportError:
            self.writer = None
            LOGGER.warning("tensorboard is not installed, tensorboard summaries are skipped.")

    def fit(self) -> str:
        """Run the full training loop.

        Returns:
            Path to the checkpoint with the best validation AUC.
        """
        net, optimizer = self.net, self.optimizer
        iteration, best_auc = 0, float("-inf")
        for epoch in range(1, self.epochs + 1):
            # fresh negative tuples for every epoch
            net.train()
            self.train_loader.dataset.next(log=True)
            iteration = self._train_epoch(iteration, epoch)

            val_metrics = self._compute_metrics(net, self.valid_loader, self.device)
            auc = val_metrics["auc"]
            self.saver.save(net, epoch, auc)
            mark = ""
            if auc > best_auc:
                best_auc, mark = auc, " (*)"
            LOGGER.info(
                "Epoch %03d/%03d - Val[%s]%s Best AUC %.4f LR %g",
                epoch,
                self.epochs,
                ", ".join(f"{k}: {v:.4f}" for k, v in sorted(val_metrics.items())),
                mark,
                best_auc,
                optimizer.param_groups[0]["lr"],
            )
            if self.writer is not None:
                for key, value in val_metrics.items():
                    self.writer.add_scalar(f"Valid/{key}", value, epoch)
                self.writer.add_scalar("Train/lr", optimizer.param_groups[0]["lr"], epoch)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(auc)
                else:
                    self.scheduler.step()

        checkpoint = self.saver.best
        LOGGER.info("Training finished. Best validation AUC %.4f (%s)", best_auc, checkpoint.name)
        return str(checkpoint)

    def _train_epoch(self, iteration: int, epoch: int) -> int:
        net, optimizer, meters = self.net, self.optimizer, AverageMeters(win_size=50)
        t0 = time.time()
        num_batches = len(self.train_loader)
        for local_it, batch in enumerate(self.train_loader, start=1):
            iteration += 1
            data = batch["data"].to(self.device, non_blocking=True)
            uidx = batch["uidx"].to(self.device, non_blocking=True)
            cate = batch["cate"].to(self.device, non_blocking=True)

            losses, accuracies = net(data, uidx, cate)
            total, stats = gather_losses(losses, self.loss_weight)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), self.grad_clip)
            optimizer.step()

            scalars = {**stats, **{k: v.float().mean().item() for k, v in accuracies.items()}}
            meters.update(scalars)
            if local_it % self.display_interval == 0:
                LOGGER.info("Epoch %03d - Iteration %04d/%04d - %s", epoch, local_it, num_batches, meters.format())
            if self.writer is not None and iteration % self.summary_interval == 0:
                for key, value in scalars.items():
                    self.writer.add_scalar(f"Train/{key}", value, iteration)
        LOGGER.info("Epoch %03d finished in %.1fs - Train[%s]", epoch, time.time() - t0, meters.format())
        return iteration
