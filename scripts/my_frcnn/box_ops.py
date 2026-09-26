"""Step 1 of the hand-written Faster R-CNN: box utilities.

Conventions used everywhere in my_frcnn:
    - Boxes are float tensors of shape [N, 4] in (x1, y1, x2, y2) absolute pixels,
      with x2 >= x1 and y2 >= y1 (continuous coordinates: width = x2 - x1, no +1).
    - Image sizes are (height, width).
    - Functions must work for N == 0, keep the input dtype/device, and be
      differentiable where it makes sense (IoU, encode, decode).
    - No Python loops over boxes: use broadcasting / vectorised tensor ops.
    - Do not call torchvision.ops or torchvision's BoxCoder here; the tests use
      them as the reference implementation.

Run the tests: python -m pytest scripts/my_frcnn -q -p no:warnings
"""

import math

import torch

DEFAULT_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
BBOX_XFORM_CLIP = math.log(1000.0 / 16)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """Area of each box.

    Args:
        boxes: [N, 4] xyxy.
    Returns:
        [N] areas.
    """
    raise NotImplementedError('TODO(you): box_area')


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise intersection-over-union.

    Args:
        boxes1: [N, 4] xyxy.
        boxes2: [M, 4] xyxy.
    Returns:
        [N, M] tensor, entry (i, j) = IoU(boxes1[i], boxes2[j]) in [0, 1].
        Non-overlapping pairs give exactly 0. Shapes [0, M] / [N, 0] for empty inputs.
    """
    raise NotImplementedError('TODO(you): box_iou')


def clip_boxes_to_image(boxes: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Clamp box coordinates so they lie inside the image.

    Args:
        boxes: [N, 4] xyxy.
        size: (height, width). x is clamped to [0, width], y to [0, height].
    Returns:
        [N, 4] clipped boxes (new tensor; do not modify the input in place).
    """
    raise NotImplementedError('TODO(you): clip_boxes_to_image')


def remove_small_boxes(boxes: torch.Tensor, min_size: float) -> torch.Tensor:
    """Indices of boxes whose width AND height are both >= min_size.

    Args:
        boxes: [N, 4] xyxy.
        min_size: minimum side length in pixels.
    Returns:
        [K] int64 indices into boxes, in increasing order.
    """
    raise NotImplementedError('TODO(you): remove_small_boxes')


def encode_boxes(gt_boxes: torch.Tensor, anchors: torch.Tensor,
                 weights: tuple[float, float, float, float] = DEFAULT_WEIGHTS) -> torch.Tensor:
    """Regression targets (dx, dy, dw, dh) that turn each anchor into its matched GT box.

    This is the box parametrisation from the Faster R-CNN paper (Ren et al. 2015,
    Sec. 3.1.2, Eq. 2): offsets of the box centre relative to the anchor centre,
    normalised by the anchor width/height, and log-space ratios of the sizes.
    Each of the four outputs is then multiplied by the corresponding entry of
    `weights` (wx, wy, ww, wh), which rescales targets to a similar magnitude.

    Args:
        gt_boxes: [N, 4] xyxy, the target box for each anchor (row i matches row i).
        anchors: [N, 4] xyxy reference boxes (anchors or proposals); positive sizes.
        weights: (wx, wy, ww, wh).
    Returns:
        [N, 4] deltas. encode_boxes(a, a) must be all zeros.
    """
    raise NotImplementedError('TODO(you): encode_boxes')


def decode_boxes(deltas: torch.Tensor, anchors: torch.Tensor,
                 weights: tuple[float, float, float, float] = DEFAULT_WEIGHTS,
                 clip: float = BBOX_XFORM_CLIP) -> torch.Tensor:
    """Inverse of encode_boxes: apply predicted deltas to anchors.

    Divide the deltas by `weights` first. Before exponentiating, clamp dw and dh
    to at most `clip` so a bad prediction cannot overflow to inf (this is why
    log(1000/16) exists: no box grows more than 1000/16 times its anchor).

    Args:
        deltas: [N, 4] (dx, dy, dw, dh), row i applies to anchors[i].
        anchors: [N, 4] xyxy.
        weights: (wx, wy, ww, wh), same as used for encoding.
        clip: upper bound applied to dw and dh after dividing by the weights.
    Returns:
        [N, 4] xyxy boxes. decode_boxes(encode_boxes(g, a), a) must recover g.
    """
    raise NotImplementedError('TODO(you): decode_boxes')
