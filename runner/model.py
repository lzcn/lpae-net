"""Model construction, checkpoint loading and optimizer setup."""

import torch

from models import LPAENet, LPAENetParam, LatentFactorNet, LatentFactorNetParam
from runner.utils import get_logger

LOGGER = get_logger("main")


def build_net(cfg, device) -> torch.nn.Module:
    """Instantiate the network from the ``net`` section of a config.

    ``net.name`` selects the variant:

    - ``lpae_u``: :class:`~models.LPAENet` with user-specific anchors
    - ``lpae_g``: :class:`~models.LPAENet` with shared + user anchors
    - ``LatentFactorNet``: :class:`~models.LatentFactorNet`

    The number of users defaults to ``data.num_users``.
    """
    net_cfg = dict(cfg["net"])
    name = net_cfg.pop("name", "lpae_u")
    net_cfg.setdefault("num_users", int(cfg["data"]["num_users"]))
    if name in ("lpae_u", "lpae_g"):
        param = LPAENetParam.from_dict({**net_cfg, "com_memory": name == "lpae_g"})
        net = LPAENet(param)
    elif name == "LatentFactorNet":
        param = LatentFactorNetParam.from_dict(net_cfg)
        net = LatentFactorNet(param)
    else:
        raise ValueError(f"Unknown net name {name!r}, use lpae_u | lpae_g | LatentFactorNet")
    num_params = sum(p.numel() for p in net.parameters())
    LOGGER.info("Built %s with %.2fM parameters on %s", type(net).__name__, num_params / 1e6, device)
    return net.to(device)


def load_trained(net, checkpoint) -> torch.nn.Module:
    """Load matching weights from a checkpoint (logs skipped keys)."""
    LOGGER.info("Loading pre-trained model from %s", checkpoint)
    state = torch.load(checkpoint, map_location="cpu")
    state_dict = state.get("model", state)
    net_state = net.state_dict()
    matched = {k: v for k, v in state_dict.items() if k in net_state and v.shape == net_state[k].shape}
    skipped = sorted(set(state_dict) - set(matched))
    if skipped:
        LOGGER.warning("Skipped mismatched keys: %s", ", ".join(skipped))
    net.load_state_dict(matched, strict=False)
    return net


def build_optimizer(cfg, net):
    """Create ``(optimizer, scheduler)`` from the ``optim`` section.

    The scheduler is optional; when it is a
    :class:`~torch.optim.lr_scheduler.ReduceLROnPlateau` the caller must step
    it with the monitored metric.
    """
    optim_cfg = dict(cfg.get("optim") or {})
    opt_name = optim_cfg.get("name", "SGD")
    kwargs = dict(optim_cfg.get("param") or {})
    if opt_name == "SGD":
        kwargs.setdefault("momentum", 0.9)
    kwargs["lr"] = float(optim_cfg.get("lr", 0.1))
    kwargs["weight_decay"] = float(optim_cfg.get("weight_decay", 0.0))
    optimizer = getattr(torch.optim, opt_name)(net.parameters(), **kwargs)

    sched_cfg = dict(optim_cfg.get("scheduler") or {})
    scheduler = None
    if sched_cfg:
        sched_name = sched_cfg.pop("name", "ReduceLROnPlateau")
        scheduler = getattr(torch.optim.lr_scheduler, sched_name)(optimizer, **sched_cfg)
    return optimizer, scheduler
