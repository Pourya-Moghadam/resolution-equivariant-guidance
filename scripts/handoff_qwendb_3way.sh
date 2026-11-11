#!/usr/bin/env bash
# Detached handoff (tmux "handoff30" on reg1), replaces handoff27's second step: jobs/qwen_drawbench_3k.jsonl (400 rows,
# output qwen_drawbench_3k) as shards 0/3 reg3, 1/3 reg4, 2/3 reg1, each launched once that host's 6K shard (session
# "job") has ended. reg5 is handed to its other user after its 6K shard (agreed by email 2026-11-11) and gets nothing.
# reg2 is not used. Session names matched exactly; pull and scoring by handoff29 (record count).
D=/data1/moghadamp/reg_data/outputs
LOG=$D/handoff_log.txt
cd /data1/moghadamp/reg-research
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
state() { if [ "$1" = reg1 ]; then tmux has-session -t "=$2" 2>/dev/null && echo BUSY || echo IDLE; else ssh -o BatchMode=yes -o ConnectTimeout=20 "$1" "bash -lc \"tmux has-session -t =$2 2>/dev/null && echo BUSY || echo IDLE\"" 2>/dev/null | tail -n 1; fi; }
ended() { [ "$(state "$1" "$2")" = IDLE ] && sleep 60 && [ "$(state "$1" "$2")" = IDLE ]; }
launch() { local h=$1; shift; if [ "$h" = reg1 ]; then bash scripts/launch_grid.sh "$@" 2>&1; else ssh -o BatchMode=yes -o ConnectTimeout=20 "$h" "bash -lc '~/reg/scripts/launch_grid.sh $*'" 2>&1; fi; }
chain() {
  local h=$1 sid=$2 out
  until ended "$h" job; do sleep 600; done
  until out=$(launch "$h" jobs/qwen_drawbench_3k.jsonl qwen_drawbench_3k "$sid" 3) && echo "$out" | grep -q started; do say "$h launch failed ($out), retry in 5 min"; sleep 300; done
  say "$out"
}
chain reg3 0 & chain reg4 1 & chain reg1 2 &
wait
say "handoff30: Qwen DrawBench 3K running on reg3, reg4 and reg1 (3 shards); reg5 left to its other user"
