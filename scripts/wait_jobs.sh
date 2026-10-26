#!/usr/bin/env bash
# Block until no server (or only the given ones) still has the tmux session "job", then pull
# outputs to reg1. Checks run in a login shell on each host because the workers' tmux lives
# in ~/.local/bin, which a plain `ssh host cmd` does not have on PATH.
#
#   bash scripts/wait_jobs.sh                 # all workers, then sync_results.sh
#   bash scripts/wait_jobs.sh reg3 reg4       # only these hosts
#   REG_WAIT_POLL=30 bash scripts/wait_jobs.sh
#
# reg1's own session is checked locally. Exit 0 when everything finished.

set -u
. "$(dirname "$0")/../env.sh"

HOSTS=${*:-"$REG_HOSTS"}
POLL=${REG_WAIT_POLL:-60}
t0=$(date +%s)

running() {  # prints the hosts that still have a "job" session
  local h
  for h in $HOSTS; do
    if [ "$h" = "${REG_HUB:-reg1}" ]; then
      tmux has-session -t job 2>/dev/null && echo "$h"
    else
      ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" 'bash -lc "tmux has-session -t job 2>/dev/null"' && echo "$h"
    fi
  done
}

while :; do
  live=$(running | tr '\n' ' ')
  [ -z "$live" ] && break
  echo "$(date +%H:%M) still running: $live"
  sleep "$POLL"
done
echo "all job sessions finished after $(( $(date +%s) - t0 )) s"
bash "$(dirname "$0")/sync_results.sh" 2>&1 | grep -E '^==|Number of regular files transferred|!!'
