#!/usr/bin/env bash
# Detached handoff (tmux "handoff27" on reg1): two grids back to back on four cards, one chain per host run in parallel:
#   1. jobs/reg_6k_grid.jsonl (100 rows, output reg_6k_grid) as shards 0/4 reg3, 1/4 reg4, 2/4 reg5, 3/4 reg1;
#   2. jobs/qwen_drawbench_3k.jsonl (400 rows, output qwen_drawbench_3k), same shard per host, once that host's 6K shard ends.
# reg1 starts only after the upscale-baseline scoring has logged its completion line. Session names matched exactly.
# reg2 is not used (lent out). Pull and scoring: scripts/handoff_pull_score.sh (handoff28/29), by record count.
D=/data1/moghadamp/reg_data/outputs
LOG=$D/handoff_log.txt
cd /data1/moghadamp/reg-research
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
state() { if [ "$1" = reg1 ]; then tmux has-session -t "=$2" 2>/dev/null && echo BUSY || echo IDLE; else ssh -o BatchMode=yes -o ConnectTimeout=20 "$1" "bash -lc \"tmux has-session -t =$2 2>/dev/null && echo BUSY || echo IDLE\"" 2>/dev/null | tail -n 1; fi; }
ended() { [ "$(state "$1" "$2")" = IDLE ] && sleep 60 && [ "$(state "$1" "$2")" = IDLE ]; }
launch() { local h=$1; shift; if [ "$h" = reg1 ]; then bash scripts/launch_grid.sh "$@" 2>&1; else ssh -o BatchMode=yes -o ConnectTimeout=20 "$h" "bash -lc '~/reg/scripts/launch_grid.sh $*'" 2>&1; fi; }
start() { local h=$1; shift; local out; until out=$(launch "$h" "$@") && echo "$out" | grep -q started; do say "$h launch failed ($out), retry in 5 min"; sleep 300; done; say "$out"; }
chain() {
  local h=$1 sid=$2
  if [ "$h" = reg1 ]; then until grep -q "Upscale baseline at 4K complete and scored" "$LOG"; do sleep 300; done; fi
  until ended "$h" job; do sleep 600; done
  start "$h" jobs/reg_6k_grid.jsonl reg_6k_grid "$sid" 4
  sleep 900
  until ended "$h" job; do sleep 600; done
  start "$h" jobs/qwen_drawbench_3k.jsonl qwen_drawbench_3k "$sid" 4
}
chain reg3 0 & chain reg4 1 & chain reg5 2 & chain reg1 3 &
wait
say "handoff27: all four hosts have started their Qwen DrawBench shards"
