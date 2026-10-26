#!/usr/bin/env bash
# Pull generated outputs from the worker servers into the hub (reg1).
# Run on reg1; safe to re-run (rsync is incremental). Workers write disjoint rows
# (idx % REG_NUM_SERVERS) and per-host log files, so nothing on reg1 is overwritten.
#
#   bash scripts/sync_results.sh                 # sync $REG_DATA/outputs from all workers -> reg1
#   bash scripts/sync_results.sh kill_or_go      # only one job directory
#
# Nightly on reg1:  0 3 * * * bash ~/reg/scripts/sync_results.sh >> ~/sync.log 2>&1

set -u
. "$(dirname "$0")/../env.sh"

SUB=${1:-}
DIR="$REG_DATA/outputs/${SUB:+$SUB/}"

mkdir -p "$DIR"
FAIL=0
for h in $REG_WORKERS; do
  echo "== $h -> $(hostname -s) : $DIR"
  rsync -a --info=stats1 --partial "$h:$DIR" "$DIR" || { echo "!! rsync from $h failed"; FAIL=1; }
done
exit $FAIL
