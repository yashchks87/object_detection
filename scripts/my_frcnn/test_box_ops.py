"""Tests for step 1 (box ops). torchvision is the reference implementation.

Run with: python -m pytest scripts/my_frcnn -q -p no:warnings
"""

import math
import sys
from pathlib import Path

import pytest
import torch
import torchvision.ops as tv_ops
from torchvision.models.detection._utils import BoxCoder

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.my_frcnn.box_ops import (BBOX_XFORM_CLIP, DEFAULT_WEIGHTS, box_area, box_iou,
                                      clip_boxes_to_image, decode_boxes, encode_boxes,
                                      remove_small_boxes)


def random_boxes(n: int, *, size: float = 500.0, min_side: float = 1.0, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    xy = torch.rand(n, 2, generator=generator) * size
    wh = torch.rand(n, 2, generator=generator) * size / 2 + min_side
    return torch.cat([xy, xy + wh], dim=1)


class TestBoxArea:
    def test_known_values(self):
        boxes = torch.tensor([[0., 0., 10., 10.], [5., 5., 7., 10.], [3., 3., 3., 8.]])
        assert torch.allclose(box_area(boxes), torch.tensor([100., 10., 0.]))

    def test_matches_torchvision(self):
        boxes = random_boxes(64)
        assert torch.allclose(box_area(boxes), tv_ops.box_area(boxes))

    def test_empty(self):
        assert box_area(torch.zeros(0, 4)).shape == (0,)


class TestBoxIou:
    def test_known_values(self):
        a = torch.tensor([[0., 0., 10., 10.]])
        b = torch.tensor([[0., 0., 10., 10.],    # identical -> 1
                          [5., 0., 15., 10.],    # half overlap: 50 / 150
                          [20., 20., 30., 30.],  # disjoint -> 0
                          [10., 0., 20., 10.],   # touching edge -> 0
                          [2., 2., 4., 4.]])     # contained: 4 / 100
        expected = torch.tensor([[1., 1 / 3, 0., 0., 0.04]])
        assert torch.allclose(box_iou(a, b), expected)

    def test_shape_and_matches_torchvision(self):
        a, b = random_boxes(37, seed=1), random_boxes(53, seed=2)
        iou = box_iou(a, b)
        assert iou.shape == (37, 53)
        assert torch.allclose(iou, tv_ops.box_iou(a, b), atol=1e-6)

    def test_range_and_symmetry(self):
        a, b = random_boxes(20, seed=3), random_boxes(30, seed=4)
        iou = box_iou(a, b)
        assert iou.min() >= 0 and iou.max() <= 1
        assert torch.allclose(iou, box_iou(b, a).T)

    def test_empty(self):
        boxes = random_boxes(5)
        assert box_iou(torch.zeros(0, 4), boxes).shape == (0, 5)
        assert box_iou(boxes, torch.zeros(0, 4)).shape == (5, 0)

    def test_differentiable(self):
        a = random_boxes(4, seed=5).requires_grad_()
        box_iou(a, random_boxes(6, seed=6)).sum().backward()
        assert a.grad is not None and torch.isfinite(a.grad).all()


class TestClipAndFilter:
    def test_clip_known_values(self):
        boxes = torch.tensor([[-5., -5., 50., 50.], [10., 10., 700., 500.], [100., 20., 200., 30.]])
        clipped = clip_boxes_to_image(boxes, (480, 640))  # (height, width)
        expected = torch.tensor([[0., 0., 50., 50.], [10., 10., 640., 480.], [100., 20., 200., 30.]])
        assert torch.equal(clipped, expected)

    def test_clip_does_not_modify_input_and_matches_torchvision(self):
        boxes = random_boxes(50, size=900) - 100
        original = boxes.clone()
        clipped = clip_boxes_to_image(boxes, (600, 800))
        assert torch.equal(boxes, original)
        assert torch.equal(clipped, tv_ops.clip_boxes_to_image(boxes, (600, 800)))

    def test_remove_small_boxes(self):
        boxes = torch.tensor([[0., 0., 10., 10.], [0., 0., 0.5, 10.], [0., 0., 10., 0.5],
                              [5., 5., 6., 6.]])
        keep = remove_small_boxes(boxes, 1.0)
        assert keep.dtype == torch.int64
        assert keep.tolist() == [0, 3]

    def test_remove_small_boxes_matches_torchvision(self):
        boxes = random_boxes(100, min_side=0.0, seed=7)
        assert torch.equal(remove_small_boxes(boxes, 30.0), tv_ops.remove_small_boxes(boxes, 30.0))


class TestBoxCoding:
    def test_identity_encodes_to_zero(self):
        anchors = random_boxes(10)
        assert torch.allclose(encode_boxes(anchors, anchors), torch.zeros(10, 4), atol=1e-6)

    def test_known_values(self):
        anchor = torch.tensor([[0., 0., 10., 10.]])  # centre (5, 5), size 10x10
        gt = torch.tensor([[2., 4., 22., 14.]])      # centre (12, 9), size 20x10
        deltas = encode_boxes(gt, anchor, weights=(1., 1., 1., 1.))
        assert torch.allclose(deltas, torch.tensor([[0.7, 0.4, math.log(2.0), 0.0]]))
        weighted = encode_boxes(gt, anchor)  # default (10, 10, 5, 5)
        assert torch.allclose(weighted, torch.tensor([[7.0, 4.0, 5 * math.log(2.0), 0.0]]))

    @pytest.mark.parametrize('weights', [DEFAULT_WEIGHTS, (1., 1., 1., 1.)])
    def test_encode_matches_torchvision(self, weights):
        gt, anchors = random_boxes(40, seed=8), random_boxes(40, seed=9)
        expected = BoxCoder(weights).encode_single(gt, anchors)
        assert torch.allclose(encode_boxes(gt, anchors, weights), expected, atol=1e-5)

    @pytest.mark.parametrize('weights', [DEFAULT_WEIGHTS, (1., 1., 1., 1.)])
    def test_decode_matches_torchvision(self, weights):
        anchors = random_boxes(40, seed=10)
        deltas = torch.randn(40, 4, generator=torch.Generator().manual_seed(11))
        expected = BoxCoder(weights).decode_single(deltas, anchors)
        assert torch.allclose(decode_boxes(deltas, anchors, weights), expected, atol=1e-4)

    def test_round_trip(self):
        gt, anchors = random_boxes(40, seed=12), random_boxes(40, seed=13)
        assert torch.allclose(decode_boxes(encode_boxes(gt, anchors), anchors), gt, atol=1e-3)

    def test_decode_clamps_huge_size_deltas(self):
        anchors = torch.tensor([[0., 0., 16., 16.]])
        deltas = torch.tensor([[0., 0., 1e4, 1e4]])
        boxes = decode_boxes(deltas, anchors, weights=(1., 1., 1., 1.))
        assert torch.isfinite(boxes).all()
        side = math.exp(BBOX_XFORM_CLIP) * 16  # = 1000
        assert torch.allclose(boxes, torch.tensor([[8 - side / 2, 8 - side / 2, 8 + side / 2, 8 + side / 2]]))

    def test_empty_and_dtype(self):
        empty = torch.zeros(0, 4)
        assert encode_boxes(empty, empty).shape == (0, 4)
        assert decode_boxes(empty, empty).shape == (0, 4)
        anchors = random_boxes(3).double()
        assert decode_boxes(torch.zeros(3, 4, dtype=torch.float64), anchors).dtype == torch.float64
