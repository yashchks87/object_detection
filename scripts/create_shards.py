#!/usr/bin/env python
"""Generate MosaicML Streaming (MDS) shards for COCO 2017 object detection.

Reads the raw COCO images + instance annotations from a UC Volume and writes
one self-contained MDS record per image, suitable for `streaming.StreamingDataset`.

Output layout (under --out-root):
    train/  index.json + group_*/shard.*.mds     (COCO train2017)
    val/    index.json + group_*/shard.*.mds     (COCO val2017)
    test/   index.json + group_*/shard.*.mds     (COCO test2017, no GT)
    dataset_meta.json

Record schema (MDS columns):
    image_id     int64    COCO image id
    file_name    str      e.g. '000000000009.jpg'
    image        bytes    raw image file bytes (not re-encoded; JPEG or PNG)
    width        int32
    height       int32
    gt_boxes     bytes    [M, 4] float32 (x1, y1, x2, y2) absolute pixels;
                          decode: np.frombuffer(b, np.float32).reshape(-1, 4)
    gt_classes   bytes    [M] int64 contiguous index into class_names;
                          decode: np.frombuffer(b, np.int64)
    gt_iscrowd   bytes    [M] uint8; decode: np.frombuffer(b, np.uint8)
    annotations  json     original COCO annotation dicts (incl. segmentation)
    meta         json     image mode/format + COCO image info (license, urls, ...)

Design notes:
    - Work is split into fixed-size image groups; a process pool converts groups
      in parallel (I/O bound: reads images from the Volume FUSE mount).
    - Each worker writes its group to fast local disk, then copies it to the
      Volume with size verification (index.json copied last as a completion
      marker), then deletes the local staging copy.
    - Restart-safe: groups already complete on the Volume (index.json present and
      every shard matching its recorded size + xxh64 hash) are skipped.
    - Only one instance may run at a time (exclusive flock next to --staging-root);
      concurrent runs would clobber each other's staging and Volume files.
    - Group indexes are merged into one index.json per split via
      `streaming.base.util.merge_index`.
    - With --terminate-cluster, the Databricks cluster is terminated (NOT
      deleted) after everything succeeds.

Usage (smoke test; always use a separate --out-root with --limit):
    python create_shards.py --limit 20 --samples-per-group 8 --out-root /local_disk0/tmp/coco_mds_smoke \
        --staging-root /local_disk0/tmp/coco_mds_smoke_staging

Full run with auto-terminate (see run_sharding.sh):
    nohup python -u create_shards.py --terminate-cluster > shard.log 2>&1 &
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import io
import json
import logging
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path

import numpy as np

LOG = logging.getLogger("create_shards")

MDS_COLUMNS = {
    "image_id": "int64",
    "file_name": "str",
    "image": "bytes",
    "width": "int32",
    "height": "int32",
    # bytes (not ndarray) because streaming's ndarray codec rejects empty
    # arrays, and the test split / images without objects have no boxes.
    "gt_boxes": "bytes",
    "gt_classes": "bytes",
    "gt_iscrowd": "bytes",
    "annotations": "json",
    "meta": "json",
}

ALLOWED_FORMATS = {"JPEG", "PNG"}
IMAGE_INFO_META_KEYS = ("license", "coco_url", "flickr_url", "date_captured")

# Set once in the parent process before the worker pool is forked; workers
# access it read-only through copy-on-write memory (no pickling of task data).
_WORKER_CTX: dict = {}


# --------------------------------------------------------------------------- #
# Raw dataset loading / task building
# --------------------------------------------------------------------------- #
@dataclass
class SampleTask:
    """Everything a worker needs to emit one MDS record."""

    image_id: int
    file_name: str
    path: str
    width: int | None = None                 # None -> read from the image header
    height: int | None = None
    annotations: list[dict] = field(default_factory=list)
    image_info: dict = field(default_factory=dict)


def load_json(path: Path):
    LOG.info("Loading %s (%.1f MB)", path.name, path.stat().st_size / 1e6)
    with open(path) as f:
        return json.load(f)


def build_tasks(images_dir: Path, instances: dict) -> list[SampleTask]:
    """One task per annotated image (including images without objects)."""
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in instances["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)
    tasks = [
        SampleTask(
            image_id=img["id"],
            file_name=img["file_name"],
            path=str(images_dir / img["file_name"]),
            width=img["width"],
            height=img["height"],
            annotations=anns_by_image.get(img["id"], []),
            image_info={k: img[k] for k in IMAGE_INFO_META_KEYS if k in img},
        )
        for img in instances["images"]
    ]
    tasks.sort(key=lambda t: t.image_id)
    n_empty = sum(1 for t in tasks if not t.annotations)
    LOG.info("Built %d tasks from %s (%d annotations, %d images without objects)",
             len(tasks), images_dir, len(instances["annotations"]), n_empty)
    return tasks


def build_test_tasks(images_dir: Path) -> list[SampleTask]:
    """test2017 has no annotation file: list the directory (image_id = file stem)."""
    LOG.info("Listing %s ...", images_dir)
    names = sorted(n for n in os.listdir(images_dir) if n.lower().endswith((".jpg", ".jpeg", ".png")))
    tasks = [SampleTask(image_id=int(Path(n).stem), file_name=n, path=str(images_dir / n)) for n in names]
    LOG.info("Built %d test tasks from %s", len(tasks), images_dir)
    return tasks


def build_categories(instances: dict) -> tuple[list[dict], dict[int, int]]:
    categories = sorted(instances["categories"], key=lambda c: c["id"])
    return categories, {c["id"]: i for i, c in enumerate(categories)}


# --------------------------------------------------------------------------- #
# Record conversion
# --------------------------------------------------------------------------- #
def task_to_record(task: SampleTask, cat_to_idx: dict[int, int]) -> dict:
    """Read one image and build its MDS record. Raises OSError/RuntimeError on bad data."""
    from PIL import Image

    with open(task.path, "rb") as f:
        image_bytes = f.read()
    with Image.open(io.BytesIO(image_bytes)) as img:
        fmt, mode, (w, h) = img.format, img.mode, img.size
    if fmt not in ALLOWED_FORMATS:
        raise RuntimeError(f"{task.path}: unsupported image format {fmt!r}")

    width = task.width if task.width is not None else w
    height = task.height if task.height is not None else h

    anns = task.annotations
    boxes = np.zeros((len(anns), 4), dtype=np.float32)
    for i, ann in enumerate(anns):
        x, y, bw, bh = ann["bbox"]
        boxes[i] = (x, y, x + bw, y + bh)
    classes = np.array([cat_to_idx[a["category_id"]] for a in anns], dtype=np.int64)
    iscrowd = np.array([a.get("iscrowd", 0) for a in anns], dtype=np.uint8)

    return {
        "image_id": task.image_id,
        "file_name": task.file_name,
        "image": image_bytes,
        "width": width,
        "height": height,
        "gt_boxes": boxes.tobytes(),
        "gt_classes": classes.tobytes(),
        "gt_iscrowd": iscrowd.tobytes(),
        "annotations": anns,
        "meta": {"mode": mode, "format": fmt, **task.image_info},
    }


# --------------------------------------------------------------------------- #
# Group processing
# --------------------------------------------------------------------------- #
def file_xxh64(path: Path, chunk: int = 16 << 20) -> str:
    import xxhash

    h = xxhash.xxh64()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def group_is_complete(volume_group_dir: Path) -> bool:
    """A group is done iff its index.json exists and lists files of the right size and xxh64 hash."""
    index_path = volume_group_dir / "index.json"
    if not index_path.exists():
        return False
    try:
        index = json.loads(index_path.read_text())
        for shard in index["shards"]:
            raw = shard["raw_data"]
            f = volume_group_dir / raw["basename"]
            if not f.exists() or f.stat().st_size != raw["bytes"]:
                return False
            expected = raw.get("hashes", {}).get("xxh64")
            if expected and file_xxh64(f) != expected:
                return False
        return True
    except (json.JSONDecodeError, KeyError, OSError):
        return False


def copy_group_to_volume(local_dir: Path, volume_dir: Path, retries: int = 3) -> None:
    """Copy shard files then index.json last (completion marker), verifying sizes."""
    files = sorted(p for p in local_dir.iterdir() if p.name != "index.json")
    files.append(local_dir / "index.json")
    for attempt in range(1, retries + 1):
        try:
            volume_dir.mkdir(parents=True, exist_ok=True)
            for src in files:
                dst = volume_dir / src.name
                shutil.copyfile(src, dst)
                if dst.stat().st_size != src.stat().st_size:
                    raise IOError(f"Size mismatch after copy: {dst}")
            return
        except OSError as exc:
            LOG.warning("Copy attempt %d/%d for %s failed: %s", attempt, retries, volume_dir, exc)
            if attempt == retries:
                raise
            time.sleep(5 * attempt)


def process_group(group_key: tuple[str, int]) -> dict:
    """Worker: convert one group of images into MDS shards on the Volume."""
    from streaming import MDSWriter

    split, group_idx = group_key
    ctx = _WORKER_CTX
    tasks: list[SampleTask] = ctx["groups"][group_key]
    group_name = f"group_{group_idx:05d}"
    local_dir = Path(ctx["staging_root"]) / split / group_name
    volume_dir = Path(ctx["out_root"]) / split / group_name

    if group_is_complete(volume_dir):
        LOG.info("[%s/%s] already complete on volume; skipping", split, group_name)
        return {"split": split, "group": group_name, "samples": len(tasks), "skipped_existing": True}

    shutil.rmtree(local_dir, ignore_errors=True)
    local_dir.mkdir(parents=True)
    t0 = time.time()
    n_bytes = n_objects = 0
    failed_samples: list[str] = []
    with MDSWriter(
        out=str(local_dir),
        columns=MDS_COLUMNS,
        compression=ctx["compression"],
        hashes=["xxh64"],
        size_limit=ctx["size_limit"],
        progress_bar=False,
    ) as writer:
        for task in tasks:
            try:
                record = task_to_record(task, ctx["cat_to_idx"])
            except (RuntimeError, OSError) as exc:
                LOG.error("[%s/%s] image %s failed: %s", split, group_name, task.file_name, exc)
                failed_samples.append(f"{split}/{task.file_name}")
                continue
            n_bytes += len(record["image"])
            n_objects += len(record["annotations"])
            writer.write(record)

    copy_group_to_volume(local_dir, volume_dir)
    shutil.rmtree(local_dir, ignore_errors=True)
    LOG.info(
        "[%s/%s] wrote %d images (%.1f MB, %d objects) in %.1fs",
        split, group_name, len(tasks) - len(failed_samples), n_bytes / 1e6, n_objects, time.time() - t0,
    )
    return {
        "split": split,
        "group": group_name,
        "samples": len(tasks) - len(failed_samples),
        "failed_samples": failed_samples,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_split(split: str, tasks: list[SampleTask], args, executor_groups: dict) -> None:
    n_groups = max(1, math.ceil(len(tasks) / args.samples_per_group))
    for i in range(n_groups):
        executor_groups[(split, i)] = tasks[i * args.samples_per_group : (i + 1) * args.samples_per_group]


def merge_split_index(out_root: Path, split: str) -> int:
    """Merge per-group index.json files into a single split-level index.json."""
    from streaming.base.util import merge_index

    split_dir = out_root / split
    index_paths = sorted(str(p) for p in split_dir.glob("group_*/index.json"))
    if not index_paths:
        raise RuntimeError(f"No group indexes found under {split_dir}")
    # merge_index shutil.move()s into split_dir and fails if a (derived) index already exists.
    (split_dir / "index.json").unlink(missing_ok=True)
    merge_index(index_paths, out=str(split_dir), keep_local=True)
    merged = json.loads((split_dir / "index.json").read_text())
    n = sum(s["samples"] for s in merged["shards"])
    LOG.info("[%s] merged %d group indexes -> %d samples total", split, len(index_paths), n)
    return n


def verify_split(out_root: Path, split: str, n_checks: int = 5) -> None:
    """Prove the split is trainable: read + decode random samples via StreamingDataset."""
    from PIL import Image
    from streaming import StreamingDataset

    ds = StreamingDataset(local=str(out_root / split), batch_size=1, shuffle=False, predownload=1)
    assert len(ds) > 0, f"{split}: empty dataset"
    rng = np.random.default_rng(0)
    for idx in rng.choice(len(ds), size=min(n_checks, len(ds)), replace=False):
        rec = ds[int(idx)]
        with Image.open(io.BytesIO(rec["image"])) as img:
            img.load()
            assert img.size == (rec["width"], rec["height"]), f"{split}[{idx}]: size mismatch"
        boxes = np.frombuffer(rec["gt_boxes"], dtype=np.float32).reshape(-1, 4)
        classes = np.frombuffer(rec["gt_classes"], dtype=np.int64)
        iscrowd = np.frombuffer(rec["gt_iscrowd"], dtype=np.uint8)
        assert len(boxes) == len(classes) == len(iscrowd) == len(rec["annotations"])
        assert np.isfinite(boxes).all(), f"{split}[{idx}]: non-finite boxes"
    LOG.info("[%s] verification OK: %d samples, spot-checked %d records", split, len(ds), n_checks)
    del ds


def acquire_single_instance_lock(staging_root: Path):
    """Hold an exclusive flock for the process lifetime; exit if another run holds it."""
    lock_path = staging_root.parent / f"{staging_root.name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        LOG.error("Another create_shards.py run holds %s; refusing to start "
                  "(check: pgrep -af create_shards.py).", lock_path)
        sys.exit(3)
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def terminate_cluster() -> None:
    """Terminate (stop, NOT delete) the current Databricks cluster."""
    import requests

    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    token = os.environ.get("DATABRICKS_TOKEN", "")
    cluster_id = os.environ.get("DATABRICKS_CLUSTER_ID", "")
    if not (host and token and cluster_id):
        LOG.error("Missing DATABRICKS_HOST/TOKEN/CLUSTER_ID env vars; cannot terminate cluster")
        return
    LOG.info("Terminating cluster %s ...", cluster_id)
    for attempt in range(1, 6):
        try:
            resp = requests.post(
                f"{host}/api/2.1/clusters/delete",
                headers={"Authorization": f"Bearer {token}"},
                json={"cluster_id": cluster_id},
                timeout=30,
            )
            if resp.ok:
                LOG.info("Cluster termination requested successfully.")
                return
            LOG.warning("Terminate attempt %d failed: HTTP %d %s", attempt, resp.status_code, resp.text[:500])
        except requests.RequestException as exc:
            LOG.warning("Terminate attempt %d failed: %s", attempt, exc)
        time.sleep(10 * attempt)
    LOG.error("Failed to terminate cluster after 5 attempts; please terminate it manually.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path,
                   default=Path("/Volumes/daai_ke_team/default/images/object_detection_datasets/coco"))
    p.add_argument("--out-root", type=Path, default=None, help="Default: <data-root>/mds_shards")
    p.add_argument("--staging-root", type=Path, default=Path("/local_disk0/coco_mds_staging"),
                   help="Fast local disk used to stage shards before copying to the Volume.")
    p.add_argument("--train-images", default="train", help="Train image dir under --data-root.")
    p.add_argument("--val-images", default="val2017", help="Val image dir under --data-root.")
    p.add_argument("--test-images", default="test2017", help="Test image dir under --data-root.")
    p.add_argument("--annotations-dir", default="annotations",
                   help="Dir under --data-root with instances_{train,val}2017.json.")
    p.add_argument("--splits", default="train,val,test", help="Comma-separated subset of {train,val,test}.")
    p.add_argument("--num-workers", type=int, default=min(32, os.cpu_count() or 8))
    p.add_argument("--samples-per-group", type=int, default=1024)
    p.add_argument("--size-limit", default="128mb", help="MDS shard size limit.")
    p.add_argument("--compression", default=None,
                   help="MDS compression (e.g. 'zstd:6'). Default: none (images are already compressed).")
    p.add_argument("--limit", type=int, default=None, help="Cap images per split (smoke testing).")
    p.add_argument("--terminate-cluster", action="store_true",
                   help="Terminate (stop, not delete) this Databricks cluster after success.")
    args = p.parse_args()
    if args.out_root is None:
        args.out_root = args.data_root / "mds_shards"
    return args


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(processName)s %(message)s",
        stream=sys.stderr,
        force=True,
    )
    args = parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    unknown = set(splits) - {"train", "val", "test"}
    if unknown:
        LOG.error("Unknown splits: %s", unknown)
        return 2
    LOG.info("Args: %s", vars(args))
    _lock = acquire_single_instance_lock(args.staging_root)  # noqa: F841 (held until exit)
    t_start = time.time()

    # ---- Build per-image tasks --------------------------------------------
    ann_dir = args.data_root / args.annotations_dir
    val_instances = load_json(ann_dir / "instances_val2017.json")  # also the category source of truth
    categories, cat_to_idx = build_categories(val_instances)
    split_tasks: dict[str, list[SampleTask]] = {}
    if "train" in splits:
        train_instances = load_json(ann_dir / "instances_train2017.json")
        split_tasks["train"] = build_tasks(args.data_root / args.train_images, train_instances)
        del train_instances
    if "val" in splits:
        split_tasks["val"] = build_tasks(args.data_root / args.val_images, val_instances)
    del val_instances
    if "test" in splits:
        split_tasks["test"] = build_test_tasks(args.data_root / args.test_images)
    if args.limit:
        split_tasks = {k: v[: args.limit] for k, v in split_tasks.items()}
    split_tasks = {k: split_tasks[k] for k in splits}
    gc.collect()

    # ---- Fan out groups to a process pool ---------------------------------
    groups: dict[tuple[str, int], list[SampleTask]] = {}
    for split, tasks in split_tasks.items():
        run_split(split, tasks, args, groups)
    total_samples = sum(len(v) for v in groups.values())
    LOG.info("Processing %d images in %d groups with %d workers", total_samples, len(groups), args.num_workers)

    args.staging_root.mkdir(parents=True, exist_ok=True)
    _WORKER_CTX.update(
        groups=groups,
        staging_root=str(args.staging_root),
        out_root=str(args.out_root),
        compression=args.compression,
        size_limit=args.size_limit,
        cat_to_idx=cat_to_idx,
    )

    failed_groups: list[tuple[str, int]] = []
    failed_samples: list[str] = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=get_context("fork")) as pool:
        futures = {pool.submit(process_group, key): key for key in sorted(groups)}
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                result = fut.result()
                failed_samples.extend(result.get("failed_samples", []))
            except Exception:
                LOG.exception("Group %s failed", key)
                failed_groups.append(key)
            done += 1
            elapsed = time.time() - t_start
            LOG.info("Progress: %d/%d groups (%.1f min elapsed, ETA %.1f min)",
                     done, len(groups), elapsed / 60, elapsed / done * (len(groups) - done) / 60)

    shutil.rmtree(args.staging_root, ignore_errors=True)
    if failed_groups:
        LOG.error("%d group(s) FAILED: %s -- fix and re-run (completed groups are skipped automatically). "
                  "Cluster left running.", len(failed_groups), failed_groups)
        return 1

    # ---- Merge per-group indexes, verify, write dataset metadata ----------
    split_counts = {split: merge_split_index(args.out_root, split) for split in split_tasks}
    for split in split_tasks:
        verify_split(args.out_root, split)

    dataset_meta = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_data_root": str(args.data_root),
        "source_image_dirs": {"train": args.train_images, "val": args.val_images, "test": args.test_images},
        "class_names": [c["name"] for c in categories],
        "categories": categories,
        "category_id_to_class_idx": cat_to_idx,
        "splits": split_counts,
        "failed_samples": failed_samples,
        "columns": MDS_COLUMNS,
        "conventions": {
            "image": "raw image file bytes (not re-encoded); JPEG except a few PNGs (meta['format']); "
                     "some are grayscale ('L') or CMYK (meta['mode']) -- convert to RGB when decoding",
            "gt_boxes": "bytes -> np.frombuffer(b, np.float32).reshape(-1, 4); (x1,y1,x2,y2) absolute pixels; "
                        "degenerate boxes are kept as-is (filter w/h < 1 at train time if needed)",
            "gt_classes": "bytes -> np.frombuffer(b, np.int64); contiguous index into class_names; "
                          "map back to COCO category_id via categories[idx]['id'] for COCO eval",
            "gt_iscrowd": "bytes -> np.frombuffer(b, np.uint8); 1 = crowd region (usually ignored in loss)",
            "annotations": "original COCO annotation dicts (bbox is [x,y,w,h]); "
                           "segmentation is polygon list or RLE dict (iscrowd=1)",
            "test": "no ground truth; gt_* are empty, width/height read from the JPEG header",
        },
    }
    meta_path = args.out_root / "dataset_meta.json"
    meta_path.write_text(json.dumps(dataset_meta, indent=2))
    LOG.info("Wrote %s", meta_path)
    if failed_samples:
        LOG.warning("%d image(s) were skipped due to unreadable data: %s", len(failed_samples), failed_samples)

    LOG.info("ALL DONE in %.1f min. Split sizes: %s", (time.time() - t_start) / 60, split_counts)
    if args.terminate_cluster:
        terminate_cluster()
    return 0


if __name__ == "__main__":
    sys.exit(main())
