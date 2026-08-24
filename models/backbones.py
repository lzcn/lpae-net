"""Torchvision CNN backbones with the classification head removed."""

from typing import Tuple

import torch.nn as nn


def build_backbone(name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """Return ``(backbone, num_features)`` for a torchvision model.

    The final classification layer is replaced by :class:`nn.Identity`, so the
    backbone outputs raw features.
    """
    import torchvision.models as tvm

    def _weights(cls, enabled):
        return cls.IMAGENET1K_V1 if enabled else None

    if name == "resnet18":
        net = tvm.resnet18(weights=_weights(tvm.ResNet18_Weights, pretrained))
        num_features = net.fc.in_features
        net.fc = nn.Identity()
    elif name == "resnet34":
        net = tvm.resnet34(weights=_weights(tvm.ResNet34_Weights, pretrained))
        num_features = net.fc.in_features
        net.fc = nn.Identity()
    elif name == "alexnet":
        net = tvm.alexnet(weights=_weights(tvm.AlexNet_Weights, pretrained))
        num_features = 4096
        net.classifier[-1] = nn.Identity()
    else:
        raise ValueError(f"Unknown backbone {name!r}, supported: resnet18 | resnet34 | alexnet")
    return net, num_features
