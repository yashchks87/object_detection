# object_detection (COCO 2017, Faster R-CNN)

Environment: Databricks cluster, 4x A10G (23GB), 48 vCPU. Use the repo venv:
`source venv/bin/activate` (`pip install -r requirements.txt`). W&B login lives in `~/.netrc`.

Data: MDS shards at `/Volumes/daai_ke_team/default/images/object_detection_datasets/coco/mds_shards`
(built by `scripts/create_shards.py`); COCOeval GT = `coco/annotations/instances_val2017.json`.

## Commands
- Unit tests: `python -m pytest scripts/test_pipeline.py -q -p no:warnings`
- Train (detached by default; survives ssh close): `./scripts/run_training.sh <run_name> [train_frcnn.py args]`
- Resume: `./scripts/run_training.sh <run_name> --resume auto`
- Auto-stop the cluster after a successful run: pass `--terminate-cluster` at launch, or for a run
  already started without it: `./scripts/terminate_when_done.sh <run_name>` (detached watcher)
- Foreground smoke test:
  `DETACH=0 RUNS_DIR=/local_disk0/tmp/smoke_runs ./scripts/run_training.sh smoke_x --epochs 2 --max-train-batches 60 --val-max-batches 10 --lr-steps 1 --warmup-iters 20 --wandb-mode offline --no-progress`
- Logs: `/local_disk0/run_logs/<run>.log`; outputs: `$RUNS_DIR/<run>/` (default on the UC Volume `coco/runs`).

## Learning track: hand-written Faster R-CNN (`scripts/my_frcnn/`)
Yash is writing this model himself to learn (and for interview prep). Production code elsewhere in
the repo is unaffected by these rules.
- NEVER write or edit implementation code in `scripts/my_frcnn/` (anything that is not a test), and
  never paste working solutions into chat, even if asked casually. If Yash explicitly says "write it
  for me", confirm once before doing so.
- Allowed: specs (signatures + docstrings with `raise NotImplementedError('TODO(you): ...')`),
  failing tests (`test_*.py`, torchvision as the reference), reviewing his diffs (point to the bug,
  explain why, do not fix it), and explaining torchvision's version only AFTER his tests pass.
- Hints only on request, escalating: (1) conceptual nudge, (2) paper/section pointer, (3) pseudocode.
- Tests: `python -m pytest scripts/my_frcnn -q -p no:warnings` (CPU-only; safe while training runs).
- Build order (one step at a time; next step's stubs + tests only once the current one passes):
  1. box ops (area, IoU, clip, remove small, encode/decode) - `box_ops.py` 2. NMS (plain PyTorch)
  3. FPN 4. anchor generator 5. matcher + balanced sampler 6. RPN (head, loss, proposals)
  7. RoIAlign (torchvision op first, own bilinear version later) 8. box head (losses, postprocess)
  9. wire in as `--model my_frcnn` in `train_frcnn.py`.
- Validation ladder: unit tests -> overfit ~10 images to ~0 loss -> smoke run -> full 2 epochs vs.
  the torchvision baseline `frcnn_v2_001` (epoch 1: loss 0.794 / 19.7 AP, epoch 2: 0.650 / 25.5 AP).

## Notes
- Never stream small writes (logs) to /Volumes (FUSE wedges processes); checkpoints are staged locally and published by a background thread.
- Streaming cache must be on /local_disk0. Measured: v2 model bs=4/GPU ~7GB peak, ~36 img/s on 4 GPUs (~55 min/epoch).
