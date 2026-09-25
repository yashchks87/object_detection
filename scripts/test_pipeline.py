"""Unit tests for the COCO Faster R-CNN pipeline: data, metric, schedule, model.

Run with: python -m pytest scripts/test_pipeline.py -q
"""

import io
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pycocotools.coco import COCO

from scripts.coco_data import clean_targets, collate_detection, decode_rgb, hflip, validate_paths
from scripts.coco_metrics import detections_to_array, evaluate_coco
from scripts.coco_models import ModelEMA, build_model, freeze_untrainable_batchnorm, parameter_groups
from scripts.train_frcnn import DEFAULT_MDS, build_parser, lr_factor


def encode(image: Image.Image, fmt: str) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


class TestData:
    def test_clean_targets_drops_crowd_and_degenerate_and_clips(self):
        boxes = np.array([[10, 10, 50, 50], [0, 0, 20, 20], [5, 5, 5, 30], [-10, -5, 700, 30]],
                         dtype=np.float32)
        classes = np.array([1, 2, 3, 4])
        iscrowd = np.array([0, 1, 0, 0], dtype=np.uint8)
        kept, kept_classes = clean_targets(boxes, classes, iscrowd, width=640, height=480)
        assert kept_classes.tolist() == [1, 4]
        assert kept[1].tolist() == [0, 0, 640, 30]

    def test_clean_targets_empty(self):
        kept, classes = clean_targets(np.zeros((0, 4), np.float32), np.zeros(0, np.int64),
                                      np.zeros(0, np.uint8), 10, 10)
        assert kept.shape == (0, 4) and classes.shape == (0,)

    def test_hflip_boxes_and_involution(self):
        image = torch.arange(2 * 3 * 4, dtype=torch.uint8).reshape(1, 2, 12).expand(3, 2, 12)
        boxes = np.array([[1, 0, 4, 2]], dtype=np.float32)
        flipped_image, flipped = hflip(image, boxes)
        assert flipped.tolist() == [[8, 0, 11, 2]]
        assert torch.equal(flipped_image[..., 0], image[..., -1])
        back_image, back = hflip(flipped_image, flipped)
        assert np.allclose(back, boxes) and torch.equal(back_image, image)

    @pytest.mark.parametrize('mode,fmt', [('RGB', 'JPEG'), ('L', 'JPEG'), ('CMYK', 'JPEG'),
                                          ('RGBA', 'PNG'), ('P', 'PNG'), ('L', 'PNG')])
    def test_decode_rgb_all_modes(self, mode, fmt):
        image = decode_rgb(encode(Image.new(mode, (37, 21)), fmt))
        assert image.shape == (3, 21, 37) and image.dtype == torch.uint8

    def test_collate_keeps_lists(self):
        sample = {'image': torch.zeros(3, 4, 5, dtype=torch.uint8),
                  'target': {'boxes': torch.zeros(0, 4), 'labels': torch.zeros(0, dtype=torch.int64)},
                  'image_id': 7}
        batch = collate_detection([sample, sample])
        assert len(batch['images']) == 2 and batch['image_ids'] == [7, 7]

    def test_validate_paths_rejects_volume_cache(self):
        with pytest.raises(ValueError):
            validate_paths('/tmp/remote', '/Volumes/x/cache')
        with pytest.raises(ValueError):
            validate_paths('/tmp/remote', '/tmp/remote/cache')


def make_coco(tmp: Path) -> COCO:
    import json
    gt = {
        'images': [{'id': 1, 'width': 100, 'height': 100}, {'id': 2, 'width': 100, 'height': 100}],
        'categories': [{'id': 1, 'name': 'a'}, {'id': 3, 'name': 'b'}],
        'annotations': [
            {'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [10, 10, 30, 30], 'area': 900, 'iscrowd': 0},
            {'id': 2, 'image_id': 1, 'category_id': 3, 'bbox': [50, 50, 40, 20], 'area': 800, 'iscrowd': 0},
            {'id': 3, 'image_id': 2, 'category_id': 3, 'bbox': [0, 0, 60, 60], 'area': 3600, 'iscrowd': 0},
        ],
    }
    path = tmp / 'gt.json'
    path.write_text(json.dumps(gt))
    from scripts.coco_metrics import load_coco_gt
    return load_coco_gt(path)


class TestMetrics:
    def test_detections_to_array_converts_xyxy_and_category(self):
        rows = detections_to_array([{'image_id': 5, 'boxes': np.array([[1, 2, 4, 8]]),
                                     'scores': np.array([0.9]), 'classes': np.array([1])}], [1, 3])
        assert rows.tolist() == [[5, 1, 2, 3, 6, 0.9, 3]]

    def test_perfect_predictions_score_one(self, tmp_path):
        coco = make_coco(tmp_path)
        detections = [
            {'image_id': 1, 'boxes': np.array([[10, 10, 40, 40], [50, 50, 90, 70]], np.float32),
             'scores': np.array([0.9, 0.8]), 'classes': np.array([0, 1])},
            {'image_id': 2, 'boxes': np.array([[0, 0, 60, 60]], np.float32),
             'scores': np.array([0.7]), 'classes': np.array([1])},
        ]
        metrics = evaluate_coco(coco, detections, [1, 2], [1, 3], ['a', 'b'])
        assert math.isclose(metrics['AP'], 1.0) and math.isclose(metrics['AP50'], 1.0)
        assert metrics['per_class_ap'] == pytest.approx({'a': 1.0, 'b': 1.0})

    def test_wrong_class_scores_zero_for_that_class(self, tmp_path):
        coco = make_coco(tmp_path)
        detections = [{'image_id': 2, 'boxes': np.array([[0, 0, 60, 60]], np.float32),
                       'scores': np.array([0.7]), 'classes': np.array([0])}]
        metrics = evaluate_coco(coco, detections, [2], [1, 3], ['a', 'b'])
        assert metrics['per_class_ap']['b'] == 0.0 and math.isnan(metrics['per_class_ap']['a'])

    def test_no_detections(self, tmp_path):
        metrics = evaluate_coco(make_coco(tmp_path), [], [1, 2], [1, 3], ['a', 'b'])
        assert metrics['AP'] == 0.0 and metrics['num_detections'] == 0


class TestSchedule:
    common = dict(total_steps=1000, steps_per_epoch=100, warmup_iters=50, warmup_factor=0.001,
                  lr_steps=[6, 8], gamma=0.1, min_lr_ratio=0.0)

    def test_multistep(self):
        f = lambda step: lr_factor(step, schedule='multistep', **self.common)  # noqa: E731
        assert math.isclose(f(0), 0.001)
        assert math.isclose(f(50), 1.0) and math.isclose(f(599), 1.0)
        assert math.isclose(f(600), 0.1) and math.isclose(f(800), 0.01)

    def test_cosine(self):
        f = lambda step: lr_factor(step, schedule='cosine', **self.common)  # noqa: E731
        assert math.isclose(f(500), 0.5, rel_tol=1e-6) and f(1000) < 1e-12

    def test_parser_defaults_validate(self):
        args = build_parser().parse_args(['--cache', '/local_disk0/x', '--out', '/tmp/y'])
        assert args.model == 'fasterrcnn_resnet50_fpn_v2' and args.min_size[-1] == 800


class TestModel:
    def test_forward_train_and_eval(self):
        torch.manual_seed(0)
        model = build_model('fasterrcnn_mobilenet_v3_large_fpn', num_classes=3,
                            pretrained_backbone=False, min_size=(128,), max_size=160)
        images = [torch.rand(3, 120, 150), torch.rand(3, 100, 90)]
        targets = [{'boxes': torch.tensor([[10., 10., 60., 70.]]), 'labels': torch.tensor([2])},
                   {'boxes': torch.zeros(0, 4), 'labels': torch.zeros(0, dtype=torch.int64)}]
        model.train()
        losses = model(images, targets)
        assert set(losses) == {'loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg'}
        sum(losses.values()).backward()
        model.eval()
        with torch.no_grad():
            outputs = model(images)
        assert len(outputs) == 2 and outputs[0]['boxes'].shape[1] == 4
        if len(outputs[0]['boxes']):  # predictions are in original-image pixels
            assert outputs[0]['boxes'][:, 2].max() <= 150 and outputs[0]['boxes'][:, 3].max() <= 120

    def test_freeze_untrainable_batchnorm(self):
        net = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.BatchNorm2d(4))
        net[1].running_mean.fill_(2.0)
        for parameter in net[1].parameters():
            parameter.requires_grad_(False)
        assert freeze_untrainable_batchnorm(net) == 1
        assert type(net[1]).__name__ == 'FrozenBatchNorm2d' and isinstance(net[2], nn.BatchNorm2d)
        assert torch.all(net[1].running_mean == 2.0)

    def test_parameter_groups_split_norm(self):
        net = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
        groups = parameter_groups(net, 1e-4, 0.0)
        assert [len(g['params']) for g in groups] == [2, 2]
        assert groups[1]['weight_decay'] == 0.0

    def test_ema_tracks_weights_and_buffers(self):
        net = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        ema = ModelEMA(net, decay=0.5, warmup=1)
        with torch.no_grad():
            net[0].weight.add_(1.0)
        net[1].running_mean.fill_(4.0)
        before = ema.module[0].weight.clone()
        ema.update(net)
        assert torch.all(ema.module[0].weight > before)
        assert torch.all(ema.module[1].running_mean > 0)
        assert ema.module[1].num_batches_tracked == net[1].num_batches_tracked


@pytest.mark.skipif(not (DEFAULT_MDS / 'val' / 'index.json').is_file(), reason='COCO MDS shards not mounted')
def test_real_val_records():
    from scripts.coco_data import COCODetectionDataset

    cache_root = Path('/local_disk0/tmp') if Path('/local_disk0').is_dir() else Path(tempfile.gettempdir())
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_root) as cache:
        dataset = COCODetectionDataset(remote=str(DEFAULT_MDS / 'val'), local=cache, training=False)
        for index in range(8):
            sample = dataset[index]
            image, target = sample['image'], sample['target']
            assert image.dtype == torch.uint8 and image.shape[0] == 3
            boxes = target['boxes']
            assert boxes.dtype == torch.float32 and boxes.shape[1] == 4
            assert torch.all(boxes[:, 2] > boxes[:, 0]) and torch.all(boxes[:, 3] > boxes[:, 1])
            assert torch.all(boxes[:, 2] <= image.shape[2]) and torch.all(boxes[:, 3] <= image.shape[1])
            assert all(1 <= label <= 80 for label in target['labels'].tolist())
