# Session log: COCO 2017 -> MosaicML Streaming (MDS) shards

Date: 2026-09-25
Chat transcript (condensed but complete) between Yash and Devin covering the
design and build of the COCO MDS sharding scripts, the full sharding run and
its overlapping-runs incident, verification of the final shards, and the
reconstruction of `create_shards.py` after the repo was re-cloned.

---

## 1. Build MDS shard-generation scripts for COCO (modelled on the Lyft repo)

**Request:** create sharding scripts in `/root/object_detection` following the
pattern of `/root/lyft_3d_object_detection/scripts/create_shards.py` and
`run_sharding.sh`.

Environment discovered:
- Databricks runtime 17.3 cluster. The Python with `mosaicml-streaming`
  installed is `$PYSPARK_PYTHON`
  (`/local_disk0/.ephemeral_nfs/cluster_libraries/python/bin/python`); the repo
  `venv/` does NOT have `streaming`.
- Target repo at the time only had `scripts/copy_to_volume.py` (+ `copy.log`),
  which copied `/local_disk0/train2017` (118,287 files, 19.31 GB) to the Volume
  `coco/train/`.

Data layout on the Volume
(`/Volumes/daai_ke_team/default/images/object_detection_datasets/coco`):

| Path | Contents |
|---|---|
| `train/` | 118,287 images - complete train2017 copy (used) |
| `train2017/` | only 3,755 files - an unfinished extraction (NOT used) |
| `val2017/` | 5,000 images |
| `test2017/` | 40,670 images (no annotation file on the Volume) |
| `annotations/` | `instances_{train,val}2017.json`, captions, keypoints |
| `*.zip` | original archives |

Annotation probe (val2017): 36,781 annotations, 446 `iscrowd` (RLE
segmentation), 36,335 polygon segmentations, 1 degenerate box (w or h < 1),
80 categories with non-contiguous COCO ids.

### Design decisions

- **train** = `coco/train/` (118,287), **val** = COCO's own `val2017` (5,000,
  annotated; not a holdout carved from train as in Lyft), **test** =
  `test2017` (40,670, no GT; width/height read from the image header).
- Images stored as the **original file bytes, not re-encoded** (no quality
  loss, no CPU cost at shard time).
- Boxes pre-converted to `(x1,y1,x2,y2)` float32 absolute pixels; classes
  mapped to a **contiguous 0..79 index** (`categories[idx]['id']` maps back to
  the COCO id for COCO eval); iscrowd kept as a separate array; full original
  annotation dicts (incl. segmentation) kept for later mask work.
- `bytes` (not `ndarray`) for GT arrays because streaming's ndarray codec
  rejects empty arrays (test split, images without objects).
- No compression by default (JPEGs are already compressed).
- Same machinery as Lyft: fixed-size groups converted in a fork-based process
  pool, staged on `/local_disk0`, copied to the Volume with `index.json` last
  as a completion marker, completed groups skipped on re-run, per-group
  indexes merged with `streaming.base.util.merge_index`, spot-check via
  `StreamingDataset`, `dataset_meta.json`, optional `--terminate-cluster`
  (Databricks `clusters/delete` = stop, not delete).

### Record schema (MDS columns)

| Column | Encoding | Notes |
|---|---|---|
| `image_id` | int64 | COCO image id |
| `file_name` | str | e.g. `000000000009.jpg` |
| `image` | bytes | raw file bytes; JPEG except a few PNGs; some grayscale (`L`) / CMYK - convert to RGB when decoding |
| `width`, `height` | int32 | |
| `gt_boxes` | bytes | `np.frombuffer(b, np.float32).reshape(-1, 4)`; xyxy pixels; degenerate boxes kept |
| `gt_classes` | bytes | `np.frombuffer(b, np.int64)`; contiguous index into `class_names` |
| `gt_iscrowd` | bytes | `np.frombuffer(b, np.uint8)`; 1 = crowd (usually ignored in loss) |
| `annotations` | json | original COCO dicts (bbox is xywh), sorted by annotation `id`, `image_id` key dropped |
| `meta` | json | `mode`, (`format`), `license`, `coco_url`, `flickr_url`, `date_captured` |

Output layout: `<out-root>/{train,val,test}/index.json + group_NNNNN/shard.*.mds`
and `<out-root>/dataset_meta.json`. Default out-root: `coco/mds_shards`.
Default 1,024 images per group -> 116 train + 5 val + 40 test = **161 groups**.

### Files

| File | Purpose |
|---|---|
| `scripts/create_shards.py` | the sharding job (args: `--data-root`, `--out-root`, `--staging-root`, `--train-images`, `--val-images`, `--test-images`, `--annotations-dir`, `--splits`, `--num-workers`, `--samples-per-group`, `--size-limit`, `--compression`, `--limit`, `--terminate-cluster`) |
| `scripts/run_sharding.sh` | nohup launcher copied from Lyft: picks a Python with `streaming`, logs to `scripts/logs/shard_<ts>.log`, terminates the cluster by default (`--no-terminate` to opt out) |

## 2. Smoke test

```bash
$PYSPARK_PYTHON create_shards.py --limit 20 --samples-per-group 8 --num-workers 8 \
    --out-root /local_disk0/tmp/coco_mds_smoke --staging-root /local_disk0/tmp/coco_mds_smoke_staging
```

- Tasks built: train 118,287 (860,001 annotations, 1,021 images without
  objects), val 5,000 (36,781 annotations, 48 without objects), test 40,670.
- `ALL DONE in 0.7 min. Split sizes: {'train': 20, 'val': 20, 'test': 20}`.
- Round-trip check via `StreamingDataset`: image bytes **byte-identical** to
  the source files; every box equals `[x, y, x+w, y+h]`; every class maps back
  to the right `category_id`.
- **Bug found on re-run:** `merge_index` `shutil.move()`s the merged index into
  the split dir and fails with `Destination path .../index.json already
  exists`. Fix: delete the derived split-level `index.json` before merging.
  Re-run then skipped all completed groups and exited 0.
- Added a note to `run_sharding.sh`: always use a separate `--out-root` with
  `--limit`, otherwise the limited groups are "complete" and poison the real
  out-root.

## 3. Full run and the overlapping-runs incident

The full run produced `No such file or directory` and `Size mismatch` errors.
Diagnosis (no bug in the script itself):

- **Two runs overlapped.** `pkill 36240` did not stop the first run because
  `pkill` matches process *names*, not PIDs. Run 1 kept going alongside run 2.
- **They clashed on shared paths.** Both used the same staging and output
  folders: one run deleted a group's staging files while the other was
  copying them (`No such file or directory`); both wrote the same Volume files
  at once (`Size mismatch`).
- Run 1 lost 39 groups, run 2 lost 42 (mostly groups the other run handled);
  together ~120 of 161 groups were written.
- **One real data issue:** `train/000000320612.jpg` is actually a **PNG** and
  the script rejected it. It sits in `train/group_00063`.

Fixes made:
- PNG images accepted; bytes stored unchanged; `meta["format"]` records `JPEG`
  or `PNG` (Pillow and `torchvision.io.decode_image` read both).
- Groups are **xxh64 hash-checked** (not just size-checked) before being
  skipped, because some groups were written by both runs (~3 s for one group).
- An exclusive `flock` stops a second `create_shards.py` from running.
- `run_sharding.sh` refuses to start if a run is already going and prints the
  PID(s).

Verified without running sharding: PNG record converts (500x375,
`{'mode': 'RGB', 'format': 'PNG'}`), hash check passes on
`train/group_00003`, lock blocks a second instance, shell syntax valid.

Resume commands given:

```bash
cd /root/object_detection/scripts
pgrep -af create_shards.py                     # must print nothing
bash run_sharding.sh --num-workers 94          # good groups skipped, failed ones regenerated
tail -f $(ls -t logs/shard_*.log | head -1)
```

To stop a run in future: `pkill -f create_shards.py` (or `kill <PID>`).

## 4. "Is it done? Confirm the shards."

Checks run (no sharding process running):

| Split | Expected | In shards | Groups |
|---|---|---|---|
| train | 118,287 | **118,286** | 116 |
| val | 5,000 | 5,000 | 5 |
| test | 40,670 | 40,670 | 40 |

- All 161 groups have their `index.json`; each split has a top-level
  `index.json`; every shard listed exists at its recorded size (0 missing /
  wrong size); index sample counts add up.
- `dataset_meta.json` written 2026-09-25 18:29 UTC, records `train: 118286`,
  `failed_samples: []`.
- Compared every train `file_name` in the shards (read straight from the MDS
  headers) against `train/` (which matches the COCO annotations exactly): the
  **only missing image is `000000320612.jpg`, the PNG**. So the run that wrote
  the shards used the pre-PNG-fix script, and the skip wasn't logged.
- The fixed scripts were gone: the repo had been re-cloned (18:35), leaving
  only `copy_to_volume.py`, and no `logs/` folder.

## 5. Recreate `create_shards.py` from this chat

The full earlier file wasn't available in the chat, so it was rebuilt from the
Lyft reference script plus the exact format of the shards on the Volume, with
all fixes from sections 2-3 included (PNG, hash-checked skip, flock,
`merge_index` re-run fix). `run_sharding.sh` recreated from Lyft + smoke-test
note + already-running guard.

Matching the Volume output exactly required three fixes found by record-level
comparison against `val/group_00000`:
1. Drop the `image_id` key from each annotation dict.
2. Sort each image's annotations by annotation `id`.
3. Convert boxes in float32 (`boxes[:, 2:] += boxes[:, :2]`), matching the
   original rounding (float64 addition differed in the last bit).

Verification:
- Regenerated val `group_00000` (1,024 images): shard xxh64 hashes
  `60c87a4131f5d722` / `fded22cd3ac922a7` - **byte-identical** to the Volume
  (with `meta["format"]` suppressed, since the Volume shards predate it).
- Test: 200 records match field by field. `categories` and
  `category_id_to_class_idx` match `dataset_meta.json`.
- PNG record OK; launcher guard refused to start with a (fake) running
  `create_shards.py`.
- Only intentional difference from the existing shards: new records carry
  `meta["format"]`; readers should use `meta.get("format")`.

Git: Yash committed and pushed the first draft as `1f08e65 Shard creator file
is added.`; the three matching fixes above were left as uncommitted changes to
`scripts/create_shards.py`.

## 6. "I don't need to re-run, right?"

**No re-run needed.** val and test are complete; train is missing one image
out of 118,287, which has no practical effect on training. Note
`dataset_meta.json` says `train: 118286`.

Before any future run, commit the fixes (the pushed version would write
records that don't match the existing shards):

```bash
cd /root/object_detection
git add scripts/create_shards.py
git commit -m "Match create_shards.py output to existing Volume shards"
git push
```

Optional, to add the missing PNG: delete `mds_shards/train/group_00063/index.json`
so the group isn't skipped, then re-run train only
(`bash run_sharding.sh --splits train --no-terminate`); all other groups are
hash-checked and skipped, and the train index + `dataset_meta.json` are
rebuilt. Caveat: a `--splits train` run rewrites `dataset_meta.json` with only
the train split count.

---

## State at end of session

- Shards: `/Volumes/daai_ke_team/default/images/object_detection_datasets/coco/mds_shards`
  (train 118,286 / val 5,000 / test 40,670; 161 groups).
- Scripts: `scripts/create_shards.py`, `scripts/run_sharding.sh`
  (run with `$PYSPARK_PYTHON`, not the repo venv).

Reading the shards:

```python
import io, json, numpy as np
from PIL import Image
from streaming import StreamingDataset

root = "/Volumes/daai_ke_team/default/images/object_detection_datasets/coco/mds_shards"
meta = json.load(open(f"{root}/dataset_meta.json"))
ds = StreamingDataset(remote=f"{root}/train", local="/local_disk0/mds_cache/coco_train",
                      batch_size=16, shuffle=True)
r = ds[0]
img = Image.open(io.BytesIO(r["image"])).convert("RGB")
boxes = np.frombuffer(r["gt_boxes"], np.float32).reshape(-1, 4)     # xyxy pixels
classes = np.frombuffer(r["gt_classes"], np.int64)                  # index into meta["class_names"]
iscrowd = np.frombuffer(r["gt_iscrowd"], np.uint8)
```
