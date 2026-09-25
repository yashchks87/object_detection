"""Official COCO box evaluation (pycocotools COCOeval) for detector outputs.

Ground truth comes from the original instances_val2017.json, so crowd
regions, area ranges and the 101-point interpolated AP are exactly the
leaderboard definition. Predictions use contiguous class indices (0..79) and
are mapped back to COCO category ids here.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

STAT_NAMES = ('AP', 'AP50', 'AP75', 'APs', 'APm', 'APl',
              'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl')


def load_coco_gt(annotation_file: Path) -> COCO:
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(annotation_file))


def detections_to_array(detections: list[dict], category_ids: list[int]) -> np.ndarray:
    """Rows of [image_id, x, y, w, h, score, category_id] for COCO.loadRes.

    Each detection dict holds 'image_id', 'boxes' [N, 4] xyxy pixels,
    'scores' [N] and 'classes' [N] contiguous class indices.
    """
    lookup = np.asarray(category_ids, dtype=np.float64)
    rows = []
    for detection in detections:
        boxes = np.asarray(detection['boxes'], dtype=np.float64).reshape(-1, 4)
        if not len(boxes):
            continue
        xywh = boxes.copy()
        xywh[:, 2:] -= xywh[:, :2]
        rows.append(np.column_stack([
            np.full(len(boxes), detection['image_id'], dtype=np.float64), xywh,
            np.asarray(detection['scores'], dtype=np.float64),
            lookup[np.asarray(detection['classes'], dtype=np.int64)]]))
    return np.concatenate(rows) if rows else np.zeros((0, 7))


def evaluate_coco(coco_gt: COCO, detections: list[dict], image_ids: list[int],
                  category_ids: list[int], class_names: list[str]) -> dict:
    """COCO bbox metrics restricted to `image_ids` (all evaluated images).

    Returns the 12 standard stats (STAT_NAMES), per-class AP@[.5:.95] and
    AP50 (NaN for classes absent from the evaluated GT), and the pycocotools
    summary text.
    """
    results = detections_to_array(detections, category_ids)
    metrics = {name: 0.0 for name in STAT_NAMES}
    metrics.update(per_class_ap={name: float('nan') for name in class_names},
                   per_class_ap50={name: float('nan') for name in class_names},
                   num_detections=len(results), num_images=len(image_ids), summary='no detections')
    if not len(results):
        return metrics
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        coco_dt = coco_gt.loadRes(results)
        evaluator = COCOeval(coco_gt, coco_dt, iouType='bbox')
        evaluator.params.imgIds = sorted(set(image_ids))
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    metrics.update({name: float(value) for name, value in zip(STAT_NAMES, evaluator.stats)})
    metrics['summary'] = buffer.getvalue()
    # precision: [iou_thresholds(10), recall(101), classes, area(4), max_dets(3)]
    precision = evaluator.eval['precision']
    order = {category: k for k, category in enumerate(evaluator.params.catIds)}
    for name, category in zip(class_names, category_ids):
        k = order[category]
        for key, values in (('per_class_ap', precision[:, :, k, 0, -1]),
                            ('per_class_ap50', precision[0, :, k, 0, -1])):
            valid = values[values > -1]
            metrics[key][name] = float(valid.mean()) if valid.size else float('nan')
    return metrics
