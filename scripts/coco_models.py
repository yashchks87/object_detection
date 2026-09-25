"""Faster R-CNN model registry, optimizer parameter groups, and weight EMA.

All detectors are built from torchvision with an ImageNet-pretrained backbone
(never COCO-pretrained detection weights -- we are training on COCO). The
model owns resizing (multi-scale min_size during training, min_size[-1] at
eval), normalisation, and the mapping of predictions back to original-image
pixels.

Models:
  fasterrcnn_resnet50_fpn_v2        default; improved recipe (BN in FPN/heads,
                                    2-conv RPN head, 4conv1fc box head)
  fasterrcnn_resnet50_fpn           classic Faster R-CNN R50-FPN baseline
  fasterrcnn_mobilenet_v3_large_fpn fast / low-memory, for debugging
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, ResNet50_Weights
from torchvision.models import detection
from torchvision.ops.misc import FrozenBatchNorm2d

BACKBONE_WEIGHTS = {
    'fasterrcnn_resnet50_fpn_v2': ResNet50_Weights.IMAGENET1K_V1,
    'fasterrcnn_resnet50_fpn': ResNet50_Weights.IMAGENET1K_V1,
    'fasterrcnn_mobilenet_v3_large_fpn': MobileNet_V3_Large_Weights.IMAGENET1K_V1,
}
MODELS = tuple(BACKBONE_WEIGHTS)
NORM_TYPES = (nn.modules.batchnorm._NormBase, nn.GroupNorm, nn.LayerNorm, FrozenBatchNorm2d)


def prefetch_backbone_weights(name: str) -> None:
    """Download the backbone checkpoint into the torch hub cache (call on one rank first)."""
    BACKBONE_WEIGHTS[name].get_state_dict(progress=False, check_hash=True)


def freeze_untrainable_batchnorm(module: nn.Module) -> int:
    """Replace BatchNorm layers whose affine params are frozen by FrozenBatchNorm2d.

    torchvision's v2 backbone keeps plain BatchNorm2d even in the stages that
    `trainable_backbone_layers` freezes, so their running statistics would
    still drift with tiny per-GPU batches. Returns the number replaced.
    """
    replaced = 0
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d) and not any(p.requires_grad for p in child.parameters()):
            frozen = FrozenBatchNorm2d(child.num_features, eps=child.eps)
            frozen.weight.copy_(child.weight.detach())
            frozen.bias.copy_(child.bias.detach())
            frozen.running_mean.copy_(child.running_mean)
            frozen.running_var.copy_(child.running_var)
            setattr(module, name, frozen)
            replaced += 1
        else:
            replaced += freeze_untrainable_batchnorm(child)
    return replaced


def build_model(name: str, *, num_classes: int, pretrained_backbone: bool = True,
                trainable_backbone_layers: int | None = None,
                min_size: tuple[int, ...] = (800,), max_size: int = 1333,
                detections_per_image: int = 100, score_threshold: float = 0.05,
                nms_threshold: float = 0.5) -> nn.Module:
    """num_classes EXCLUDES background; the head is built with num_classes + 1."""
    if name not in BACKBONE_WEIGHTS:
        raise ValueError(f'Unknown model {name!r}; choose from {MODELS}.')
    model = getattr(detection, name)(
        weights=None,
        weights_backbone=BACKBONE_WEIGHTS[name] if pretrained_backbone else None,
        num_classes=num_classes + 1,
        trainable_backbone_layers=trainable_backbone_layers,
        min_size=tuple(min_size), max_size=max_size,
        box_detections_per_img=detections_per_image,
        box_score_thresh=score_threshold, box_nms_thresh=nms_threshold)
    freeze_untrainable_batchnorm(model)
    return model


def parameter_groups(model: nn.Module, weight_decay: float, norm_weight_decay: float) -> list[dict]:
    """Normalisation-layer parameters get their own (usually zero) weight decay."""
    norm, other = [], []
    for module in model.modules():
        for parameter in module.parameters(recurse=False):
            if parameter.requires_grad:
                (norm if isinstance(module, NORM_TYPES) else other).append(parameter)
    groups = [{'params': other, 'weight_decay': weight_decay}]
    if norm:
        groups.append({'params': norm, 'weight_decay': norm_weight_decay})
    return groups


class ModelEMA:
    """Exponential moving average of weights AND buffers (BN statistics).

    The effective decay ramps up as decay * (1 - exp(-updates / warmup)) so
    the average is not dominated by the random-init weights early on.
    Operates on the unwrapped model; every DDP rank keeps an identical copy.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998, warmup: int = 2000):
        if not 0 < decay < 1 or warmup < 1:
            raise ValueError('EMA decay must be in (0, 1) and warmup positive.')
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.decay, self.warmup, self.updates = decay, warmup, 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = self.decay * (1 - math.exp(-self.updates / self.warmup))
        source = model.state_dict()
        for key, value in self.module.state_dict().items():
            if value.dtype.is_floating_point:
                value.lerp_(source[key].detach(), 1 - decay)
            else:
                value.copy_(source[key])

    def state_dict(self) -> dict:
        return {'module': self.module.state_dict(), 'updates': self.updates,
                'decay': self.decay, 'warmup': self.warmup}

    def load_state_dict(self, state: dict) -> None:
        self.module.load_state_dict(state['module'])
        self.updates = state['updates']
