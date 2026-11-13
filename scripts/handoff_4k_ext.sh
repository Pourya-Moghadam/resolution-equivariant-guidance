#!/usr/bin/env bash
# Detached handoff (tmux "handoff31" on reg1): jobs/reg_4k_ext.jsonl (150 rows, output reg_4k_ext) in four shards.
# reg5 (back from its other user, Fri 2026-11-13) runs shards 0, 1 and 2 one after another, starting now; reg3 takes
# shard 3 once its Qwen DrawBench shard has ended. Pull and scoring: handoff_pull_score.sh reg_4k_ext 150 refs (handoff32).
D=/data1/moghadamp/reg_data/outputs
LOG=$D/handoff_log.txt
cd /data1/moghadamp/reg-research
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
state() { if [ "$1" = reg1 ]; then tmux has-session -t "=$2" 2>/dev/null && echo BUSY || echo IDLE; else ssh -o BatchMode=yes -o ConnectTimeout=20 "$1" "bash -lc \"tmux has-session -t =$2 2>/dev/null && echo BUSY || echo IDLE\"" 2>/dev/null | tail -n 1; fi; }
ended() { [ "$(state "$1" "$2")" = IDLE ] && sleep 60 && [ "$(state "$1" "$2")" = IDLE ]; }
launch() { local h=$1; shift; if [ "$h" = reg1 ]; then bash scripts/launch_grid.sh "$@" 2>&1; else ssh -o BatchMode=yes -o ConnectTimeout=20 "$h" "bash -lc '~/reg/scripts/launch_grid.sh $*'" 2>&1; fi; }
start() { local h=$1; shift; local out; until out=$(launch "$h" "$@") && echo "$out" | grep -q started; do say "$h launch failed ($out), retry in 5 min"; sleep 300; done; say "$out"; }
chain5() { for sid in 0 1 2; do until ended reg5 job; do sleep 600; done; start reg5 jobs/reg_4k_ext.jsonl reg_4k_ext "$sid" 4; sleep 900; done; }
chain3() { until ended reg3 job; do sleep 600; done; start reg3 jobs/reg_4k_ext.jsonl reg_4k_ext 3 4; }
chain5 & chain3 & wait
say "handoff31: all four reg_4k_ext shards have been started (reg5 x3, reg3 x1)"
