#!/usr/bin/env bash
# Run one shell command on all REG servers (or only workers) over SSH, sequentially,
# printing a header per host. Requires the reg1..reg5 aliases in ~/.ssh/config.
#
#   bash scripts/fanout.sh 'nvidia-smi --query-gpu=name --format=csv,noheader'
#   bash scripts/fanout.sh --workers 'cd ~/reg && git pull -q'
#   bash scripts/fanout.sh --parallel 'bash ~/reg/scripts/server_audit.sh'
#
# Override the host list with REG_HOSTS="reg1 reg2" if needed.

set -u

HOSTS=${REG_HOSTS:-"reg1 reg2 reg3 reg4 reg5"}
PARALLEL=0

while [ $# -gt 0 ]; do
  case "$1" in
    --workers)  HOSTS=${REG_WORKERS:-"reg2 reg3 reg4 reg5"}; shift ;;
    --parallel) PARALLEL=1; shift ;;
    --) shift; break ;;
    -*) echo "unknown flag $1" >&2; exit 2 ;;
    *) break ;;
  esac
done

if [ $# -eq 0 ]; then
  echo "usage: $0 [--workers] [--parallel] 'command'" >&2
  exit 2
fi
CMD="$*"

run_one() {
  local h=$1
  local out
  out=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" "bash -lc $(printf '%q' "$CMD")" 2>&1)
  local rc=$?
  printf '\n##### %s (exit %d) #####\n%s\n' "$h" "$rc" "$out"
  return $rc
}

FAIL=0
if [ "$PARALLEL" -eq 1 ]; then
  pids=()
  for h in $HOSTS; do run_one "$h" & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p" || FAIL=1; done
else
  for h in $HOSTS; do run_one "$h" || FAIL=1; done
fi
exit $FAIL
