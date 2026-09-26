# Session log: Faster R-CNN training pipeline for COCO 2017 (MDS shards)

Date: 2026-09-25
Chat transcript (condensed but complete) between Yash and Devin covering the
design and build of the production Faster R-CNN training stack on the COCO
MDS shards, modelled on `/root/lyft_3d_object_detection`, plus the benchmark
and smoke test that followed.

---

## 1. Request

**Request:** create a production-level (not rookie-level) Faster R-CNN
training codebase in `/root/object_detection`, at the level of the Lyft repo
(`/root/lyft_3d_object_detection`), with effective W&B logging, and a
training starter with nohup capability so training keeps running after the
ssh session is closed. Data:
`/Volumes/daai_ke_team/default/images/object_detection_datasets/coco/mds_shards`.

## 2. Exploration

Reference repo read in full: `scripts/train_lidar.py`, `lidar_data.py`,
`lidar_models.py`, `run_training.sh`, `run_chain.sh`,
`plans/2026-09-21-lidar-training-session.md`. Patterns carried over:
torchrun DDP with a `Cluster` helper, StreamingDataset self-partitioning (no
DistributedSampler), node-local shard cache (never on /Volumes),
`CheckpointPublisher` (stage locally, publish to the Volume from a background
thread; FUSE-safe rewrite-instead-of-append for metrics.jsonl), SIGTERM ->
KeyboardInterrupt so `finally` blocks run, W&B finished with exit_code=1 on
crash, `_TRAINING_SUCCESS` marker, `--terminate-cluster` handled by the
launcher after the log copy, live log only on `/local_disk0`, detached
launch via `setsid --fork`.

Environment:
- Databricks cluster, **4x NVIDIA A10G (23GB)**, 48 vCPU, 181GB RAM,
  `/local_disk0` 3.5TB free.
- Repo `venv/` (Python 3.12) was empty at first; `$PYSPARK_PYTHON` has no
  wandb/pycocotools. Note: `which torchrun` points at
  `/databricks/python3/bin/torchrun`, so the launcher uses
  `venv/bin/python -m torch.distributed.run` instead.
- Internet OK (PyPI, download.pytorch.org). `DATABRICKS_HOST/TOKEN/CLUSTER_ID`
  set. No W&B login at first.

Shards (`mds_shards/`): `dataset_meta.json` (80 `class_names`, `categories`
with non-contiguous COCO ids, `category_id_to_class_idx`, splits train
118,286 / val 5,000 / test 40,670), train 19GB, val 799MB. Probe of
`val/group_00000` (803 images): 800 RGB + 3 grayscale (`L`), 71 crowd boxes,
0 degenerate, 6 images without objects. COCOeval GT:
`coco/annotations/instances_val2017.json` (20MB).

## 3. Dependencies

`requirements.txt` created (pinned, same torch/streaming/wandb as Lyft; all
versions > 7 days old): torch 2.14.0, torchvision 0.29.0,
mosaicml-streaming 0.13.0, wandb 0.30.0, numpy 2.1.3, pillow 12.3.0,
pycocotools 2.0.11, tqdm 4.70.1, requests 2.34.2, pytest 9.1.1.

Yash installed the dependencies into the object_detection venv and ran
`wandb login` (logged in as `yashchks87`, key in `/root/.netrc`). Verified:
torch 2.14.0+cu130, torchvision 0.29.0+cu130, streaming 0.13.0, wandb 0.30.0,
pycocotools and pytest importable, 4 GPUs visible.

## 4. Design decisions

- **Model:** `fasterrcnn_resnet50_fpn_v2` by default (also
  `fasterrcnn_resnet50_fpn`, `fasterrcnn_mobilenet_v3_large_fpn` for
  debugging). **ImageNet-pretrained backbone only** (`ResNet50_Weights.IMAGENET1K_V1`),
  never COCO detection weights, since we train on COCO.
- **v2 BatchNorm:** torchvision's v2 backbone keeps plain BatchNorm2d even in
  the frozen stages (`trainable_backbone_layers=3`), so their running stats
  would drift -> BN layers with frozen params are replaced by
  `FrozenBatchNorm2d`; remaining BN converted to **SyncBatchNorm** under DDP
  (auto-on for v2).
- **Resize inside the model:** loader yields uint8 images at original size;
  GeneralizedRCNNTransform resizes/normalises on the GPU and maps predictions
  back to original pixels (no coordinate fixes for COCOeval). **Multi-scale
  training** via `min_size=(640..800 step 32)`, eval at 800, max 1333.
- **Targets:** crowd boxes dropped (Faster R-CNN has no ignore regions),
  boxes clipped to the image, boxes thinner than 0.01px removed (torchvision
  raises on degenerate boxes, which would deadlock DDP); labels = class idx + 1
  (0 = background). Images without boxes kept as negatives. Horizontal flip 0.5.
- **Decoding:** `torchvision.io.decode_image(mode=RGB)` with PIL fallback;
  handles RGB/L/CMYK JPEG and RGBA/P/L PNG.
- **Aspect-ratio grouping rejected:** with streaming, leftover partial
  buckets would give ranks different batch counts -> DDP hang.
- **Optimisation:** SGD momentum 0.9, lr 0.02 at global batch 16 (linearly
  scaled), wd 1e-4 (0 on norm params), 1000-iter linear warmup (factor 1e-3),
  multistep decay x0.1 after epochs 16 and 22 of 26 (torchvision reference
  recipe); cosine and AdamW available. bf16 autocast (fp16 + GradScaler
  fallback). Optional grad clipping (norm always logged). Optional weight EMA
  (`--ema`, decay 0.9998 with warmup ramp) used for validation and best.pt.
- **Nonfinite handling:** always backpropagate so every rank enters the
  all-reduce; the NaN shows in the identical all-reduced grad norm, so all
  ranks skip the step together; abort after 50 consecutive skips.
- **cuDNN benchmark off** (input shapes vary every batch); TF32 on.
- **Evaluation:** official pycocotools COCOeval against
  `instances_val2017.json` (copied once to local disk); each rank infers its
  val partition, detections gathered to rank 0, de-duplicated by image_id
  (streaming pads partitions). 12 standard stats + per-class AP and AP50.
- **Checkpoints / resume:** `best.pt` (weights, EMA if enabled, + config),
  `last.pt` (model/optimizer/scheduler/scaler/EMA/step/best AP/W&B run id),
  refreshed every epoch AND every `--checkpoint-every 2000` optimizer steps.
  Mid-epoch resume positions the stream exactly via
  `StreamingDataset.load_state_dict({'epoch': e-1, 'sample_in_epoch':
  batches_done * batch * world, ...})` (streaming epochs are 0-based; the
  state lives in shared memory so persistent spawn workers pick it up).
  End-of-epoch resume also sets the streaming epoch so the shuffle order does
  not replay epoch 0 (a latent issue in the Lyft trainer). `--resume auto` =
  `<out>/last.pt` if it exists.
- **Outputs default to the UC Volume** (`coco/runs/<run>`) so weights survive
  cluster termination/wipes; streaming cache on
  `/local_disk0/mds_cache_coco`.

## 5. W&B logging design

Primary rank only; project `COCO object detection`, entity `yashchks87`,
run name = run directory name; `--wandb` on by default (`--no-wandb`,
`--wandb-mode offline|disabled`).
- All metrics use `trainer/global_step` as the x-axis (`define_metric`), so a
  resumed run that replays steps after its last checkpoint overlaps instead
  of having logs rejected; resume re-attaches to the same run id.
- Every 20 steps: `train/loss` + 4 components (classifier, box_reg,
  objectness, rpn_box_reg), averaged over the interval and across ranks;
  `train/lr`, `train/grad_norm`, `train/skipped_steps`, `train/amp_scale`
  (fp16 only), `perf/images_per_second`, `perf/data_wait_fraction`,
  `perf/step_time_ms`, `perf/gpu_mem_peak_gb`, `trainer/epoch` (fractional).
- Every epoch: `epoch/train_*`, `epoch/elapsed_minutes`, `val/AP, AP50, AP75,
  APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl`, `val/best_AP`,
  `val/eval_seconds`, `val/num_detections`, `val_per_class/AP_<class>` (80
  scalars), `val/per_class_table` + bar chart, and `val/predictions`: 16 fixed
  val images with toggleable prediction (score >= 0.4) and ground-truth box
  layers (padded to a shared canvas; the first smoke test showed W&B warns on
  mixed image sizes).
- Summaries: `val/AP*` max, `train/loss` min, `best_ap`, `best_epoch`,
  `parameters`. Optional `--wandb-log-model` uploads best.pt as an artifact.
  W&B system metrics (GPU util/mem) come for free.

## 6. Files created

| File | Purpose |
|---|---|
| `scripts/coco_data.py` | `COCODetectionDataset(StreamingDataset)`, `decode_rgb`, `clean_targets`, `hflip`, `collate_detection`, `validate_paths`, `load_dataset_meta` |
| `scripts/coco_models.py` | Model registry, `freeze_untrainable_batchnorm`, `parameter_groups` (norm wd), `ModelEMA`, backbone weight prefetch (rank 0 downloads first) |
| `scripts/coco_metrics.py` | `evaluate_coco` (pycocotools, per-class AP/AP50, empty-detection safe), `detections_to_array` (xyxy -> xywh, class idx -> COCO id) |
| `scripts/train_frcnn.py` | Trainer (`Trainer` class): DDP, AMP, grad accum, schedules, EMA, resumable checkpoints, COCO eval, W&B, `terminate_cluster()` |
| `scripts/run_training.sh` | Launcher: detached by default via `nohup setsid --fork`, pre-flight guards, log header (command + git sha), log copy, terminate-after-success |
| `scripts/test_pipeline.py` | 23 pytest tests (data cleaning/flip/decode modes, COCOeval perfect/wrong/empty, LR schedule, model forward train/eval, BN freezing, param groups, EMA, real val records) |
| `requirements.txt`, `AGENTS.md`, `.gitignore` (+ `wandb/`, caches, logs) | |

Launcher details:
- `DETACH=1` default (`DETACH=0` = foreground with tqdm). Why `setsid` and
  not just nohup: the Databricks ssh-tunnel SIGTERMs every process in its sshd
  session on shutdown ("No SSH clients for 10m0s"); nohup/disown only cover
  SIGHUP (this killed a Lyft run in its last epoch). The job gets its own
  session and reparents to init.
- Pre-flight refusals: venv lacks torch/torchvision/streaming/pycocotools;
  another `train_frcnn.py` already running (override `ALLOW_CONCURRENT=1`);
  `RUNS_DIR/<run>` exists without `--resume`; W&B enabled but not logged in;
  `--terminate-cluster` without Databricks env vars.
- Env overrides: `RUNS_DIR`, `CACHE_DIR`, `NUM_GPUS`, `PYTHON`, `LOCAL_LOG_DIR`.

## 7. Verification

- **Unit tests:** 23/23 pass (`python -m pytest scripts/test_pipeline.py -q -p no:warnings`).
  Note: torchvision 0.29 emits a DeprecationWarning for its image decoders
  (moving to TorchCodec); still fully functional.
- **GPU benchmark** (single A10G, v2 model, bf16, fwd+bwd+SGD, real val
  images at 800 short side; worst case = mixed extreme aspect ratios padded to
  ~1344x1344 with 60 GT boxes):

  | per-GPU batch | real peak | worst-case peak | throughput |
  |---|---|---|---|
  | 2 | 3.1 GB | 4.2 GB | 11.4 img/s |
  | 4 | 6.2 GB | 7.0 GB | 11.1 img/s |
  | 6 | 9.3 GB | 10.3 GB | 10.3 img/s |
  | 8 | 12.3 GB | 13.6 GB | 10.1 img/s |

  GPU is compute-saturated from bs=2; **bs=4 chosen** (4 x 4 GPUs = reference
  global batch 16, lr 0.02 unscaled, lots of memory headroom).
- **4-GPU smoke test** (foreground, W&B offline, `--ema`, 2 epochs x 60
  batches, val capped at 10 batches/rank): exit 0; setup, step logs, epoch
  logs, COCOeval, best.pt (175MB), last.pt (524MB, incl. optimizer + EMA),
  config.json, metrics.jsonl, `_TRAINING_SUCCESS` all produced. Loss 3.97 ->
  ~0.95; **~36 img/s aggregate -> ~55 min/epoch, ~24h for 26 epochs**.
  "no detections" at eval was expected: EMA weights after only 120 steps are
  still ~initial.
- Not yet verified: a real mid-epoch kill + `--resume auto` cycle (Yash
  declined the standalone stream-resume check; the resume path is exercised
  for the first time on a real resume - check for the `"event": "resume"` log
  line with the right epoch/step), and online W&B sync (smoke run was offline).

## 8. Yash runs the training himself

Yash asked for the command to run himself (no launch by Devin). Nothing was
launched. Leftovers from the smoke test that can be deleted:
`/local_disk0/tmp/smoke_runs` and the offline run under `wandb/`.

## 9. Plan docs and experiment note

- This session log was written on request (chat portion only, in `plans/`).
- `plans/exp001-frcnn-v2-baseline.md`: short note for the first experiment
  (model/arch table, data, hyperparameters, derived schedule numbers:
  7,393 steps/epoch, 192,218 total, LR drops at steps 118,288 / 162,646;
  43.66M params / 43.44M trainable; expectation ~40-42 AP (estimate), ~24 h;
  empty Results section to fill in).
- Yash launched `./scripts/run_training.sh frcnn_v2_001` himself (22:09 UTC,
  2026-09-25, WITHOUT `--terminate-cluster`) and committed/pushed the
  codebase (`2e2fd55 Added model training phase.`).

## 10. Auto-terminate for a run already started without the flag

**Request:** does the training turn the cluster off at the end? -> No (flag is
fixed at launch). Killing and relaunching would lose progress, so added
`scripts/terminate_when_done.sh <run_name>`: a detached watcher that waits for
the run's torchrun PID to exit, then for run_training.sh's
"Training exited with status N" line, sleeps a settle period (log copy), and
calls `terminate_cluster()` ONLY if status 0 AND `_TRAINING_SUCCESS` exists.
Guards: Databricks env vars set, run actually running (PID cmdline must
contain `train_frcnn.py`, avoids PID reuse), no duplicate watcher.
`POLL_SECONDS`/`SETTLE_SECONDS` overridable. Tested against a fake run with a
stub PYTHON (success -> terminate called; failure -> not called; not running ->
error). Yash asked whether to kill the current run: no, just start the
watcher; he did (watcher PID 53108). `AGENTS.md` gained the command.
Committed by Yash as `eabd340`.

## 11. Why GPU utilization / memory differ between GPUs

Sampled at 0.5 s: each GPU 81-100% util (~92% mean), 140-195 W, throughput
steady 36-37 img/s, CPU ~90% idle. Explanation given:
- nvidia-smi util = fraction of time any kernel ran in the sample window.
- DDP stragglers: every step waits in the gradient all-reduce for the slowest
  rank; ranks get different image sizes (aspect ratio + random multi-scale)
  and GT counts. Scaling: 11.1 img/s single GPU -> 36-37 on 4 (~83%), typical
  on PCIe A10G without NVLink.
- Short CPU-bound gaps in Faster R-CNN (anchor matching, sampling, NMS).
- Memory 12-20 GB vs 7 GB benchmark = caching-allocator RESERVED memory
  (fragmentation from varying shapes) + CUDA context + NCCL buffers; not a
  leak, not an OOM risk. Real peak = W&B `perf/gpu_mem_peak_gb`.
- Possible later speedups (not done): bf16 DDP gradient compression hook,
  remove the per-step host sync.

## 12. Skills: plan-update and setup-env

- `.agents/skills/plan-update/SKILL.md` (Yash's requested path/name): writes or
  updates a compact session log in `plans/` (verify facts via git/logs, one
  numbered section per request, final "Current state" section, no secrets,
  never commit). Frontmatter: `triggers: [user]`, allowed read/grep/glob,
  `permissions.allow` Write(plans/**), git status/log/diff, ls.
- Fixes along the way: placeholder was `SKILL.MD` -> renamed to `SKILL.md`
  (case-sensitive); the template block was fenced as ```markdown, which made
  the VS Code editor render it as the file's own headings -> changed to a
  4-backtick `text` fence; folder renamed `update_plans` -> `plan-update`
  because the slash command comes from the folder name.
- Not showing in the Skills panel: project skills load only from the
  workspace root, and the Devin workspace is `/root`, not
  `/root/object_detection`. Open `/root/object_detection` as the workspace
  (or ask for a global copy in `~/.config/devin/skills/`). In this session the
  skill is still registered under its old name `update_plans`, so the
  invocation was carried out by following the file's instructions manually.
- `.agents/skills/setup-env/SKILL.md`: runs, in this exact order from
  `/root`, in one persistent shell, stopping at the first failure:
  `sudo apt-get update`, `pyenv install 3.12`, `pyenv local 3.12`,
  `cd object_detection/`, `pip install -r requirements.txt`. If 3.12 is already
  installed, answer N to pyenv's prompt. Note: step 5 uses the pyenv pip, not
  `venv/` (the launcher defaults to `venv/bin/python`; pass `PYTHON=...` or add
  a venv step if needed). Both skills committed by Yash as `7a6bb5f`.

## 13. Incident: frcnn_v2_001 killed at 00:06:43 UTC (2026-09-26)

**Request:** why did the run crash? Not a code, GPU, OOM or data failure.

| Time (UTC) | Event |
|---|---|
| 22:09 | run started; epochs 1-2 completed (~53 min each) |
| 23:48:23 | IDE server: "The client has disconnected" (laptop asleep/network) |
| 00:00 | epoch 2 done; `last.pt` / `best.pt` published to the Volume |
| 00:06:43.169 | tunnel stdout: `"No SSH clients for 10m0s, shutting down..."` |
| 00:06:43 | torchrun "Received 15 death signal"; driver stdout: `Reaped child 49799 with status 15` (supervisor) and `Reaped child 53108 with status 15` (watcher) |
| 01:59 | Yash reconnected; Databricks started a new tunnel (new notebook process 243968) |

Root cause, from the tunnel bootstrap
`/Workspace/Users/yash.choksi@intusurg.com/.databricks/ssh-tunnel/0.295.0/<cluster>/ssh-server-bootstrap.py`:
- The Databricks ssh tunnel runs as a job inside a notebook Python process
  (`db_ipykernel_launcher`, here PID 4694, child of the driver JVM) that calls
  `prctl(PR_SET_CHILD_SUBREAPER)`. Anything detached with setsid/nohup/disown is
  re-parented to THAT process, not to init.
- When the ssh server exits (10 min without clients), `kill_all_children()`
  runs `pkill -P <notebook pid>` (SIGTERM) in a ~10 Hz loop until no children
  remain. It never finishes because interactive IDE shells survive SIGTERM,
  so the loop is STILL running on 4694 (endless "No child has changed state"
  in `/databricks/driver/logs/stdout`) and instantly kills any newly adopted
  orphan (this is also why test orphans died within 0.3 s).
- It killed the setsid'd supervisor and watcher (direct children), then
  torchrun once it was re-parented. The Lyft-era rationale for setsid
  ("new session escapes the teardown") was therefore incomplete.
- Evidence ruled out: no reboot (uptime), no OOM, no driver restart, same
  cgroup for all processes, the tunnel job itself kept running after 00:06.

Impact: `last.pt` = end of epoch 2 (step 14,786, `epoch_complete: True`); lost
~914 batches (~7 min; the mid-epoch checkpoint at step 16,000 was not reached).
W&B reported "Fatal error while uploading data" at shutdown (last minutes may
be unsynced; `wandb sync wandb/run-20260925_220913-6k8qpuzy`). The watcher died
too, so the cluster would never have been stopped.

| Epoch | global step | train loss | val AP | val AP50 |
|---|---|---|---|---|
| 1 | 7,393 | 0.794 | 0.197 | 0.383 |
| 2 | 14,786 | 0.650 | 0.255 | 0.450 |

## 14. Fix: shield-process launcher

- Rejected: `systemd-run` (PID 1 is real systemd 255, units start fine, see
  GPUs/env/netrc) because processes outside the notebook process tree get
  **"Operation not permitted" on `/Volumes`** (matches the bootstrap comment
  about losing wsfs/dbfs access when re-parented to PID 1).
- Adopted: two-level launch in `run_training.sh` and `terminate_when_done.sh`.
  1. **Shield:** `nohup env --ignore-signal=TERM,INT setsid --fork <script>`
     (`DETACHED_SESSION=shield`). This is the process the subreaper adopts. It
     starts with SIGTERM already ignored, so there is no startup race with the
     kill loop, and it just waits for its child.
  2. **Supervisor / watcher:** started by the shield with
     `env --default-signal=TERM,INT,HUP` (`DETACHED_SESSION=supervisor` or
     `watcher`), i.e. normal signal handling. It is the shield's child, so
     `pkill -P` never reaches it, and it stays inside the notebook tree (keeps
     `/Volumes` access). Stopping is unchanged: `kill $(cat <run>.pid)`.
  - Scripts now re-exec via an absolute `$SCRIPT` path; pre-flight checks run
    only when `DETACHED_SESSION` is empty. coreutils 9.4 `env` verified to
    set/clear the SIGTERM ignore bit.
- Test against the LIVE kill loop (shells launched under PID 4694):
  `shield_test` (2 epochs x 150 batches, W&B offline,
  `RUNS_DIR=/local_disk0/tmp/smoke_runs`, watcher with a stub PYTHON). Chain
  torchrun -> supervisor -> shield -> 4694; trained normally (~37 img/s),
  exited 0, `_TRAINING_SUCCESS` written, watcher logged "run succeeded" and
  called the (stub) terminate. A first attempt failed only because of a bad
  test argument (`--lr-steps 1` with `--epochs 1`); that run still verified
  the failure path (supervisor recorded status 1, watcher did NOT terminate).
- Residual risk (unverified): if the notebook process 4694 itself ever exits,
  the shield would be re-parented to PID 1 and might lose `/Volumes` access
  (checkpoint publishing would then fail and be reported at the end).

## 15. Resume

Yash ran `./scripts/run_training.sh frcnn_v2_001 --resume auto --terminate-cluster`
at 02:39:48 UTC (torchrun PID 306057; chain 306057 -> 305844 -> 305838 ->
4694). Log: `{"event": "resume", ..., "epoch": 3, "skip_batches": 0,
"global_step": 14786}`; first steps 14,800 / 14,820 at loss 0.64 (continuous
with epoch 2's 0.65), ~37 img/s. This is the first real exercise of
`--resume auto` and it behaved as designed (same W&B run id `6k8qpuzy`).

---

## Current state (end of session, 2026-09-26 ~02:45 UTC)

- **Running:** `frcnn_v2_001` epoch 3/26, resumed from step 14,786; ~24 epochs
  x ~55 min left (~21-22 h). `--terminate-cluster` is set, so after a
  successful run the log is copied to the Volume and then the cluster is
  stopped (not deleted). No separate watcher is needed or running.
- **Paths:** live log `/local_disk0/run_logs/frcnn_v2_001.log`; outputs
  `/Volumes/daai_ke_team/default/images/object_detection_datasets/coco/runs/frcnn_v2_001/`;
  W&B `yashchks87/COCO object detection/frcnn_v2_001` (run id `6k8qpuzy`).

```bash
cd /root/object_detection && source venv/bin/activate
tail -f /local_disk0/run_logs/frcnn_v2_001.log              # monitor
kill $(cat /local_disk0/run_logs/frcnn_v2_001.pid)          # graceful stop
./scripts/run_training.sh frcnn_v2_001 --resume auto --terminate-cluster   # resume
./scripts/terminate_when_done.sh <run_name>                 # auto-stop for a run started without the flag
```

- **Uncommitted:** `scripts/run_training.sh`, `scripts/terminate_when_done.sh`
  (shield launcher) and this plan update.
- **Leftovers to delete:** `/local_disk0/tmp/smoke_runs`,
  `/local_disk0/tmp/stub_python`, offline W&B runs under `wandb/`.
- **Open items:** `wandb sync wandb/run-20260925_220913-6k8qpuzy` to recover
  pre-crash minutes; open `/root/object_detection` as the Devin workspace so
  `/plan-update` and `/setup-env` load; fill in the Results section of
  `plans/exp001-frcnn-v2-baseline.md` when the run ends; residual risk from
  section 14 (unverified); the Lyft repo's `run_training.sh` has the same
  setsid weakness (not changed).
