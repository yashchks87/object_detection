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

## Notes
- Never stream small writes (logs) to /Volumes (FUSE wedges processes); checkpoints are staged locally and published by a background thread.
- Streaming cache must be on /local_disk0. Measured: v2 model bs=4/GPU ~7GB peak, ~36 img/s on 4 GPUs (~55 min/epoch).
