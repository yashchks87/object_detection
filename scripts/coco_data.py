"""Streaming data pipeline for COCO 2017 detection training.

Reads the MDS shards produced by scripts/create_shards.py (see its module
docstring for the record schema) and yields torchvision-detection-ready
samples:

    image     uint8 [3, H, W] RGB at the ORIGINAL resolution. Resizing and
              normalisation happen on the GPU inside the model's
              GeneralizedRCNNTransform, which also maps predictions back to
              original-image pixels, so evaluation needs no coordinate fixes.
    target    {'boxes': float32 [M, 4] xyxy pixels, 'labels': int64 [M]}
              labels are 1-based (0 is the Faster R-CNN background class):
              label = contiguous class index + 1.
    image_id  COCO image id (for COCOeval and cross-rank de-duplication).

Target cleaning (matches the torchvision/detectron2 references): crowd
regions are dropped (Faster R-CNN has no ignore-region support), boxes are
clipped to the image and anything thinner than MIN_BOX_SIZE pixels is
removed, since torchvision raises on degenerate boxes -- and in DDP an
exception on one rank would deadlock the others. Images left without boxes
stay in training as pure negatives.
"""

from __future__ import annotations

import io
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from streaming import StreamingDataset
from torchvision.io import ImageReadMode, decode_image

MIN_BOX_SIZE = 1e-2  # pixels


def validate_paths(remote: str, local: str) -> tuple[Path, Path]:
    remote_path, local_path = Path(remote), Path(local)
    if not remote_path.is_absolute() or not local_path.is_absolute():
        raise ValueError('Remote dataset and local cache must be absolute paths.')
    remote_path, local_path = remote_path.resolve(), local_path.resolve()
    if remote_path == local_path or remote_path in local_path.parents or local_path in remote_path.parents:
        raise ValueError('Remote dataset and local cache must be separate, non-nested directories.')
    if any(root == local_path or root in local_path.parents for root in (Path('/Volumes'), Path('/dbfs'))):
        raise ValueError('Cache must be on node-local disk, not /Volumes or /dbfs.')
    return remote_path, local_path


def load_dataset_meta(remote_root: Path) -> dict:
    """dataset_meta.json with class names and the contiguous-index -> COCO id map."""
    meta_path = remote_root / 'dataset_meta.json'
    if not meta_path.is_file():
        raise ValueError(f'Missing dataset_meta.json under {remote_root}.')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    names, categories = meta.get('class_names'), meta.get('categories')
    if not names or not categories or [c['name'] for c in categories] != names:
        raise ValueError('dataset_meta.json class_names and categories are missing or inconsistent.')
    return meta


def decode_rgb(data: bytes) -> torch.Tensor:
    """Decode JPEG/PNG bytes (RGB, grayscale, CMYK, palette) to uint8 [3, H, W]."""
    try:
        return decode_image(torch.frombuffer(bytearray(data), dtype=torch.uint8), mode=ImageReadMode.RGB)
    except RuntimeError:  # exotic encodings libjpeg-turbo/libpng reject
        with Image.open(io.BytesIO(data)) as image:
            array = np.asarray(image.convert('RGB'))
        return torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous()


def clean_targets(boxes: np.ndarray, classes: np.ndarray, iscrowd: np.ndarray,
                  width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Drop crowd regions, clip to the image, remove degenerate boxes."""
    boxes = boxes.copy()
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, width)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, height)
    keep = ((iscrowd == 0)
            & (boxes[:, 2] - boxes[:, 0] > MIN_BOX_SIZE)
            & (boxes[:, 3] - boxes[:, 1] > MIN_BOX_SIZE))
    return boxes[keep], classes[keep]


def hflip(image: torch.Tensor, boxes: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    width = image.shape[-1]
    flipped = boxes.copy()
    flipped[:, 0] = width - boxes[:, 2]
    flipped[:, 2] = width - boxes[:, 0]
    return image.flip(-1), flipped


class COCODetectionDataset(StreamingDataset):
    """Streams COCO MDS records as (uint8 image, detection target, image_id).

    With torchrun, StreamingDataset partitions samples across ranks and loader
    workers by itself -- do NOT wrap it in a DistributedSampler. All ranks on a
    node must share the same `local` cache directory.
    """

    def __init__(self, remote: str, local: str, *, training: bool, hflip_prob: float = 0.5,
                 batch_size: int = 1, **kwargs):
        remote_path, local_path = validate_paths(remote, local)
        if not (remote_path / 'index.json').is_file():
            raise ValueError(f'No index.json under {remote_path}; not an MDS split directory.')
        if not 0 <= hflip_prob <= 1:
            raise ValueError('hflip_prob must be in [0, 1].')
        self.training = training
        self.hflip_prob = hflip_prob if training else 0.0
        kwargs.setdefault('shuffle', training)
        kwargs.setdefault('validate_hash', None)
        kwargs.setdefault('download_timeout', 300)
        super().__init__(remote=str(remote_path), local=str(local_path),
                         batch_size=batch_size, **kwargs)

    def __getitem__(self, index: int) -> dict:
        record = super().__getitem__(index)
        image = decode_rgb(record['image'])
        height, width = image.shape[-2:]
        boxes, classes = clean_targets(
            np.frombuffer(record['gt_boxes'], dtype=np.float32).reshape(-1, 4),
            np.frombuffer(record['gt_classes'], dtype=np.int64),
            np.frombuffer(record['gt_iscrowd'], dtype=np.uint8), width, height)
        if self.hflip_prob and random.random() < self.hflip_prob:
            image, boxes = hflip(image, boxes)
        return {
            'image': image,
            'target': {'boxes': torch.from_numpy(np.ascontiguousarray(boxes, dtype=np.float32)).reshape(-1, 4),
                       'labels': torch.from_numpy(classes.astype(np.int64) + 1)},
            'image_id': int(record['image_id']),
        }


def collate_detection(samples: list[dict]) -> dict:
    """Variable-size images stay a list; the model pads/batches them on the GPU."""
    return {
        'images': [sample['image'] for sample in samples],
        'targets': [sample['target'] for sample in samples],
        'image_ids': [sample['image_id'] for sample in samples],
    }
