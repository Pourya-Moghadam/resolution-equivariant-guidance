#!/usr/bin/env bash
# Start scripts/run_grid.py for one job file in a detached tmux session "job" on this host.
# Meant to be called on every participating server (locally on reg1, over SSH on workers):
#
#   bash scripts/launch_grid.sh jobs/kill_or_go.jsonl kill_or_go            # REG_SERVER_ID / REG_NUM_SERVERS from env.sh
#   bash scripts/launch_grid.sh jobs/kill_or_go.jsonl kill_or_go 1 4        # explicit split (e.g. when a server is skipped)
#   ssh reg3 "bash -lc '~/reg/scripts/launch_grid.sh jobs/kill_or_go.jsonl kill_or_go 1 4'"
#
# Output dir is $REG_DATA/outputs/<name>; the log is <out>/log_<host>.txt. The tmux session is "job"
# (REG_JOB_SESSION overrides it for a side job next to a running grid). Refuses to start if that
# session already exists. Follow up with scripts/wait_jobs.sh on reg1 (it watches "job" sessions).

set -eu
cd "$(dirname "$0")/.."
. ./env.sh

JOBS=${1:?job file}
NAME=${2:?output name}
SID=${3:-${REG_SERVER_ID:?}}
NUM=${4:-$REG_NUM_SERVERS}
OUT=$REG_DATA/outputs/$NAME
mkdir -p "$OUT"

SESSION=${REG_JOB_SESSION:-job}
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "$(hostname -s): a '$SESSION' session is already running" >&2
  exit 1
fi
tmux new-session -d -s "$SESSION" \
  "cd $PWD && . ./env.sh && python scripts/run_grid.py --jobs $JOBS --out $OUT --server-id $SID --num-servers $NUM > $OUT/log_$(hostname -s).txt 2>&1"
echo "$(hostname -s): started $JOBS as server $SID/$NUM -> $OUT"
