#!/usr/bin/env bash
# Terminate (stop, NOT delete) the Databricks cluster once an already-running
# training run finishes successfully -- for runs launched WITHOUT
# --terminate-cluster. New runs should just pass --terminate-cluster to
# run_training.sh instead.
#
# Usage:
#   ./scripts/terminate_when_done.sh <run_name>
#
# Runs detached (nohup setsid --fork, like run_training.sh), so it survives
# closing ssh. It waits for the run's torchrun process to exit, then for
# run_training.sh to record the exit status and copy the log to RUNS_DIR, and
# terminates the cluster ONLY if the exit status is 0 AND
# <RUNS_DIR>/<run_name>/_TRAINING_SUCCESS exists. Any failure or interruption
# leaves the cluster running for debugging / --resume.
#
# Monitor / cancel:
#   cat /local_disk0/run_logs/<run_name>.terminate_watch.log
#   kill "$(cat /local_disk0/run_logs/<run_name>.terminate_watch.pid)"
set -euo pipefail

RUN_NAME="${1:?usage: terminate_when_done.sh <run_name>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/venv/bin/python}"
RUNS_DIR="${RUNS_DIR:-/Volumes/daai_ke_team/default/images/object_detection_datasets/coco/runs}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-/local_disk0/run_logs}"
LOG_FILE="$LOCAL_LOG_DIR/$RUN_NAME.log"
PID_FILE="$LOCAL_LOG_DIR/$RUN_NAME.pid"
WATCH_LOG="$LOCAL_LOG_DIR/$RUN_NAME.terminate_watch.log"
WATCH_PID="$LOCAL_LOG_DIR/$RUN_NAME.terminate_watch.pid"
POLL_SECONDS="${POLL_SECONDS:-60}"      # how often to check whether training is alive
SETTLE_SECONDS="${SETTLE_SECONDS:-60}"  # grace period for the final log copy to RUNS_DIR

is_training() {  # guards against PID reuse: the PID must still be our trainer
    [[ -r "/proc/$1/cmdline" ]] && tr '\0' ' ' < "/proc/$1/cmdline" | grep -q 'train_frcnn\.py'
}

note() {
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" >> "$WATCH_LOG"
}

watch_run() {
    echo $$ > "$WATCH_PID"
    local pid status
    pid="$(cat "$PID_FILE")"
    note "watching run '$RUN_NAME' (torchrun PID $pid)"
    while is_training "$pid"; do sleep "$POLL_SECONDS"; done
    note "torchrun exited; waiting for run_training.sh to record the exit status"
    for _ in $(seq 60); do
        grep -q 'Training exited with status' "$LOG_FILE" 2>/dev/null && break
        sleep "$(( POLL_SECONDS < 10 ? POLL_SECONDS : 10 ))"
    done
    status="$(sed -n 's/.*Training exited with status \([0-9]*\).*/\1/p' "$LOG_FILE" | tail -1)"
    sleep "$SETTLE_SECONDS"  # let run_training.sh finish copying the log to RUNS_DIR
    if [[ "$status" == "0" && -f "$RUNS_DIR/$RUN_NAME/_TRAINING_SUCCESS" ]]; then
        note "run succeeded; terminating the cluster"
        (cd "$REPO_ROOT" && "$PYTHON" -c "import sys; sys.path.insert(0, '.'); \
from scripts.train_frcnn import terminate_cluster; terminate_cluster()") >> "$WATCH_LOG" 2>&1 || true
    else
        note "run did not succeed (status '${status:-unknown}', _TRAINING_SUCCESS" \
             "$([[ -f "$RUNS_DIR/$RUN_NAME/_TRAINING_SUCCESS" ]] && echo present || echo missing));" \
             "cluster left running"
    fi
    rm -f "$WATCH_PID"
}

if [[ "${DETACHED_SESSION:-0}" == "1" ]]; then
    watch_run
    exit 0
fi

# Pre-flight (visible to the caller).
for var in DATABRICKS_HOST DATABRICKS_TOKEN DATABRICKS_CLUSTER_ID; do
    [[ -n "${!var:-}" ]] || { echo "error: $var is not set; cannot terminate the cluster." >&2; exit 1; }
done
[[ -s "$PID_FILE" ]] || { echo "error: no PID file $PID_FILE; is '$RUN_NAME' running?" >&2; exit 1; }
is_training "$(cat "$PID_FILE")" || { echo "error: run '$RUN_NAME' is not running (PID $(cat "$PID_FILE"))." >&2; exit 1; }
if [[ -s "$WATCH_PID" ]] && kill -0 "$(cat "$WATCH_PID")" 2>/dev/null; then
    echo "error: already watching '$RUN_NAME' (watcher PID $(cat "$WATCH_PID"))." >&2
    exit 1
fi

rm -f "$WATCH_PID"
DETACHED_SESSION=1 PYTHON="$PYTHON" RUNS_DIR="$RUNS_DIR" LOCAL_LOG_DIR="$LOCAL_LOG_DIR" \
    POLL_SECONDS="$POLL_SECONDS" SETTLE_SECONDS="$SETTLE_SECONDS" \
    nohup setsid --fork "$0" "$RUN_NAME" < /dev/null > /dev/null 2>&1
for _ in $(seq 50); do [[ -s "$WATCH_PID" ]] && break; sleep 0.2; done
[[ -s "$WATCH_PID" ]] || { echo "error: failed to start the watcher; see $WATCH_LOG" >&2; exit 1; }

echo "Watching run '$RUN_NAME' (watcher PID $(cat "$WATCH_PID")). Safe to close the terminal/ssh."
echo "  The cluster TERMINATES after the run finishes successfully; failures leave it running."
echo "  status: cat $WATCH_LOG"
echo "  cancel: kill \$(cat $WATCH_PID)"
