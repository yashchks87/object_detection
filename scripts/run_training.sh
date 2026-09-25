#!/usr/bin/env bash
# Launch Faster R-CNN COCO training on all local GPUs with torchrun.
#
# Usage:
#   ./scripts/run_training.sh <run_name> [extra train_frcnn.py args...]
#
# Detached (DEFAULT; returns immediately, survives closing ssh / the IDE):
#   ./scripts/run_training.sh frcnn_v2_001
#   ./scripts/run_training.sh frcnn_v2_001 --epochs 26 --terminate-cluster
#
# Resume the same run (same W&B run, exact mid-epoch stream position):
#   ./scripts/run_training.sh frcnn_v2_001 --resume auto
#
# Foreground (holds the terminal, streams output + progress bars):
#   DETACH=0 ./scripts/run_training.sh smoke_001 --epochs 1 --max-train-batches 50 --no-wandb
#
# Environment overrides:
#   RUNS_DIR   run outputs (checkpoints/metrics/config/final log). Default is the
#              UC Volume so weights survive cluster termination / wipes.
#   CACHE_DIR  node-local streaming shard cache (default /local_disk0/mds_cache_coco)
#   NUM_GPUS   processes per node (default: all visible GPUs)
#   PYTHON     interpreter (default: <repo>/venv/bin/python)
#
# Detach mechanics: the run is re-executed under `nohup setsid --fork`. nohup
# makes it ignore SIGHUP; setsid puts it in a brand-new session with no
# controlling terminal, reparented to init. The second part matters on
# Databricks: the ssh-tunnel serving the IDE terminal SIGTERMs every process
# in its sshd session when it shuts down ("No SSH clients for 10m0s"), which
# nohup/disown alone do not survive (this killed a Lyft run in its last epoch).
#
# Logging: the live log is written ONLY to node-local disk while training runs
# (streaming small writes to a /Volumes FUSE mount can wedge the processes in
# uninterruptible I/O). When training ends the log is copied once to RUNS_DIR.
# Checkpoints are staged locally and published to RUNS_DIR by a background
# thread in the trainer, so a slow Volume never stalls DDP into an NCCL timeout.
#
# --terminate-cluster is handled HERE, not in the trainer, so the order is
# guaranteed: training ends -> log copied to RUNS_DIR -> cluster terminated
# (stop, not delete). Only after a fully successful run; any failure leaves the
# cluster up for debugging and `--resume auto`.
#
# Monitor / stop a detached run:
#   tail -f /local_disk0/run_logs/<run_name>.log
#   kill "$(cat /local_disk0/run_logs/<run_name>.pid)"   # graceful; resume later
set -euo pipefail

LAUNCH_ARGS=("$@")  # kept verbatim so a detached run can re-exec itself

RUN_NAME="${1:?usage: run_training.sh <run_name> [extra train_frcnn.py args...]}"
shift
if [[ "$RUN_NAME" == -* || "$RUN_NAME" == */* ]]; then
    echo "error: first argument must be a run name (no leading '-', no '/'), got '$RUN_NAME'" >&2
    exit 1
fi

# Intercept --terminate-cluster; everything else is passed to the trainer.
TERMINATE=0
RESUMING=0
USE_WANDB=1
TRAIN_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --terminate-cluster) TERMINATE=1; continue ;;
        --resume|--resume=*) RESUMING=1 ;;
        --no-wandb) USE_WANDB=0 ;;
        --wandb) USE_WANDB=1 ;;
    esac
    TRAIN_ARGS+=("$arg")
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/venv/bin/python}"
CACHE_DIR="${CACHE_DIR:-/local_disk0/mds_cache_coco}"
RUNS_DIR="${RUNS_DIR:-/Volumes/daai_ke_team/default/images/object_detection_datasets/coco/runs}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --list-gpus | wc -l)}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-/local_disk0/run_logs}"
LOG_FILE="$LOCAL_LOG_DIR/$RUN_NAME.log"
PID_FILE="$LOCAL_LOG_DIR/$RUN_NAME.pid"
DETACH="${DETACH:-1}"

# ---------------------------------------------------------------------------
# Pre-flight checks (only in the submitting process, where errors are visible).
# ---------------------------------------------------------------------------
if [[ "${DETACHED_SESSION:-0}" != "1" ]]; then
    if ! "$PYTHON" -c 'import torch, torchvision, streaming, pycocotools' >/dev/null 2>&1; then
        echo "error: $PYTHON lacks torch/torchvision/streaming/pycocotools;" \
             "pip install -r requirements.txt into the venv (or set PYTHON=...)." >&2
        exit 1
    fi
    if RUNNING="$(pgrep -f 'scripts/train_frcnn\.py' | tr '\n' ' ')" && [[ -n "$RUNNING" ]] \
            && [[ "${ALLOW_CONCURRENT:-0}" != "1" ]]; then
        echo "error: train_frcnn.py is already running (PID(s): $RUNNING); the GPUs are busy." >&2
        echo "       Stop it (kill \$(cat $LOCAL_LOG_DIR/<run>.pid)) or set ALLOW_CONCURRENT=1." >&2
        exit 1
    fi
    if [[ -e "$RUNS_DIR/$RUN_NAME" && "$RESUMING" == 0 ]]; then
        echo "error: $RUNS_DIR/$RUN_NAME already exists. Pick a new run name, or continue it" >&2
        echo "       with: $0 $RUN_NAME --resume auto" >&2
        exit 1
    fi
    if [[ "$USE_WANDB" == 1 && -z "${WANDB_API_KEY:-}" ]] \
            && ! grep -qs 'api.wandb.ai' "$HOME/.netrc"; then
        echo "error: W&B is enabled but not logged in; run 'wandb login' once (a detached" \
             "job cannot prompt), export WANDB_API_KEY, or pass --no-wandb." >&2
        exit 1
    fi
    if [[ "$TERMINATE" == 1 ]]; then
        for var in DATABRICKS_HOST DATABRICKS_TOKEN DATABRICKS_CLUSTER_ID; do
            if [[ -z "${!var:-}" ]]; then
                echo "error: $var is not set; cannot honour --terminate-cluster." >&2
                exit 1
            fi
        done
    fi
fi

mkdir -p "$RUNS_DIR" "$LOCAL_LOG_DIR"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1

COMMAND=("$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS"
         scripts/train_frcnn.py --cache "$CACHE_DIR" --out "$RUNS_DIR/$RUN_NAME" "${TRAIN_ARGS[@]}")

# Clear leftover shared memory from previously killed runs; a no-op otherwise.
clear_stale_shared_memory() {
    "$PYTHON" -c "from streaming.base.util import clean_stale_shared_memory; clean_stale_shared_memory()" \
        2>/dev/null || true
}

copy_log() {
    cp -f "$LOG_FILE" "$RUNS_DIR/$RUN_NAME.log" || true
}

finalize() {  # copy the log to RUNS_DIR first, only then terminate the cluster
    local status="$1"
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] Training exited with status $status; copying log to $RUNS_DIR." >> "$LOG_FILE"
    copy_log
    if [[ "$TERMINATE" == 1 ]]; then
        if [[ "$status" -eq 0 ]]; then
            "$PYTHON" -c "import sys; sys.path.insert(0, '.'); \
from scripts.train_frcnn import terminate_cluster; terminate_cluster()" >> "$LOG_FILE" 2>&1 || true
        else
            echo "Training failed (status $status); cluster left running for debugging/resume." >> "$LOG_FILE"
        fi
        copy_log  # refresh the snapshot so the termination outcome is captured too
    fi
}

log_header() {
    {
        echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] run '$RUN_NAME' on $(hostname), $NUM_GPUS GPU(s)"
        echo "command: ${COMMAND[*]}"
        echo "git: $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo n/a)" \
             "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null | grep -q . && echo '(dirty)')"
    } > "$LOG_FILE"
}

supervise() {  # run training, then finalize; safe to detach
    local status=0
    clear_stale_shared_memory
    log_header
    "${COMMAND[@]}" >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    wait $! || status=$?
    finalize "$status"
    return "$status"
}

if [[ "$DETACH" == "1" ]]; then
    COMMAND+=(--no-progress)  # progress bars are pointless in a file

    if [[ "${DETACHED_SESSION:-0}" == "1" ]]; then
        # Already re-executed into our own session: this process supervises training.
        status=0
        supervise > /dev/null 2>&1 || status=$?
        exit "$status"
    fi

    # Resolved settings are passed explicitly so the re-exec cannot disagree
    # with what is reported below.
    rm -f "$PID_FILE"  # never report a stale PID from an earlier run
    DETACH=1 DETACHED_SESSION=1 PYTHON="$PYTHON" CACHE_DIR="$CACHE_DIR" RUNS_DIR="$RUNS_DIR" \
        LOCAL_LOG_DIR="$LOCAL_LOG_DIR" NUM_GPUS="$NUM_GPUS" \
        nohup setsid --fork "$0" "${LAUNCH_ARGS[@]}" < /dev/null > /dev/null 2>&1

    deadline=$((SECONDS + 120))
    while [[ ! -s "$PID_FILE" && "$SECONDS" -lt "$deadline" ]]; do
        sleep 0.2
    done
    if [[ ! -s "$PID_FILE" ]]; then
        echo "Failed to submit run '$RUN_NAME'; see $LOG_FILE." >&2
        exit 1
    fi

    echo "Submitted run '$RUN_NAME' (torchrun PID $(cat "$PID_FILE")). Safe to close the terminal/ssh."
    echo "  live log:  tail -f $LOG_FILE"
    echo "  outputs:   $RUNS_DIR/$RUN_NAME/  (final log copied to $RUNS_DIR/$RUN_NAME.log)"
    echo "  stop:      kill \$(cat $PID_FILE)"
    echo "  resume:    $0 $RUN_NAME --resume auto"
    [[ "$TERMINATE" == 1 ]] && echo "  NOTE: the cluster TERMINATES automatically after a successful run."
    exit 0
fi

status=0
clear_stale_shared_memory
log_header
"${COMMAND[@]}" 2>&1 | tee -a "$LOG_FILE" || status=$?
finalize "$status"
exit "$status"
