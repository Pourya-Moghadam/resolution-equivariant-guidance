#!/usr/bin/env bash
# Detached handoff (tmux "handoff43" on reg1): when every host's reg_4k_yarn shard has ended, start
# jobs/reg_4k_baselines.jsonl (300 rows: hires fix at two strengths, APG in two variants) in four shards.
# Pull and score: handoff_pull_score.sh reg_4k_baselines 300 refs (handoff44).
LOG=/data1/moghadamp/reg_data/outputs/handoff_log.txt
cd /data1/moghadamp/reg-research
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
state() { if [ "$1" = reg1 ]; then tmux has-session -t "=$2" 2>/dev/null && echo BUSY || echo IDLE; else ssh -o BatchMode=yes -o ConnectTimeout=20 "$1" "bash -lc \"tmux has-session -t =$2 2>/dev/null && echo BUSY || echo IDLE\"" 2>/dev/null | tail -n 1; fi; }
ended() { [ "$(state "$1" "$2")" = IDLE ] && sleep 60 && [ "$(state "$1" "$2")" = IDLE ]; }
launch() { local h=$1; shift; if [ "$h" = reg1 ]; then bash scripts/launch_grid.sh "$@" 2>&1; else ssh -o BatchMode=yes -o ConnectTimeout=20 "$h" "bash -lc '~/reg/scripts/launch_grid.sh $*'" 2>&1; fi; }
start() { local h=$1; shift; local out; until out=$(launch "$h" "$@") && echo "$out" | grep -q started; do say "$h launch failed ($out), retry in 5 min"; sleep 300; done; say "$out"; }
one() { until ended $1 job; do sleep 300; done; start $1 jobs/reg_4k_baselines.jsonl reg_4k_baselines $2 4; }
one reg1 0 & one reg3 1 & one reg4 2 & one reg5 3 & wait
say "handoff43: reg_4k_baselines started on four cards"
