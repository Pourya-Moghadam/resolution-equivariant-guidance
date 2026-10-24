#!/usr/bin/env bash
# Pull generated outputs from the worker servers into the hub (reg1).
# Run from the laptop or from reg1 itself; safe to re-run (rsync is incremental).
#
#   bash scripts/sync_results.sh                 # sync /data/reg/outputs from reg2..reg5 -> reg1
#   bash scripts/sync_results.sh kill_or_go      # only one job directory
#
# Nightly on reg1:  0 3 * * * bash ~/reg/scripts/sync_results.sh >> ~/sync.log 2>&1

set -u

HUB=${REG_HUB:-reg1}
WORKERS=${REG_WORKERS:-"reg2 reg3 reg4 reg5"}
ROOT=${REG_DATA:-/data/reg}
SUB=${1:-}
SRC="$ROOT/outputs/${SUB:+$SUB/}"
DST="$ROOT/outputs/${SUB:+$SUB/}"

# If we are already on the hub, rsync directly; otherwise tell the hub to do it.
if [ "$(hostname -s)" = "$HUB" ] || [ "$(hostname)" = "$HUB" ]; then
  for h in $WORKERS; do
    echo "== $h -> $(hostname) : $SRC"
    mkdir -p "$DST"
    rsync -a --info=stats1 --partial --ignore-existing "$h:$SRC" "$DST" || echo "!! rsync from $h failed"
  done
else
  ssh -o BatchMode=yes "$HUB" "REG_WORKERS='$WORKERS' REG_DATA='$ROOT' bash ~/reg/scripts/sync_results.sh ${SUB}"
fi
