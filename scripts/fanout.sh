#!/usr/bin/env bash
# Run one shell command on every REG server (or only the workers), printing a header
# per host. Run it on the hub (reg1): the hub runs the command locally, the workers
# over SSH via the reg2..reg5 aliases in ~/.ssh/config. On every host the command
# runs in a login shell with ~/reg/env.sh sourced, so REG_* variables and the venv
# are available.
#
#   bash scripts/fanout.sh 'nvidia-smi --query-gpu=name --format=csv,noheader'
#   bash scripts/fanout.sh --workers 'cd ~/reg && bash scripts/setup_env.sh'
#   bash scripts/fanout.sh --parallel 'bash ~/reg/scripts/server_audit.sh'
#
# Host lists come from env.sh (REG_HOSTS / REG_WORKERS, derived from REG_NUM_SERVERS).

set -u
. "$(dirname "$0")/../env.sh"

HUB=${REG_HUB:-reg1}
HOSTS=$REG_HOSTS
PARALLEL=0

while [ $# -gt 0 ]; do
  case "$1" in
    --workers)  HOSTS=$REG_WORKERS; shift ;;
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
CMD="[ -f ~/reg/env.sh ] && . ~/reg/env.sh; $*"

run_one() {
  local h=$1
  local out
  if [ "$h" = "$HUB" ]; then
    out=$(bash -lc "$CMD" 2>&1)
  else
    out=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" "bash -lc $(printf '%q' "$CMD")" 2>&1)
  fi
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
