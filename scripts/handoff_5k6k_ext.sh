#!/usr/bin/env bash
# Detached handoff (tmux "handoff35" on reg1), 2026-11-18, all hosts on the NVMe scratch:
#   jobs/reg_5k6k_ext.jsonl (200 rows, outputs/reg_5k6k_ext) shards 0/1/2 on reg1/reg3/reg4 now; reg5 runs
#   jobs/geneval_1k.jsonl (553 rows, native 1K reference, ~5 h) first and takes shard 3 when that has ended.
# Pull/score: handoff_pull_score.sh geneval_1k 553 norefs cat (handoff36, then handoff_geneval_1k_score.sh) and
# handoff_pull_score.sh reg_5k6k_ext 200 refs (handoff37).
LOG=/data1/moghadamp/reg_data/outputs/handoff_log.txt
cd /data1/moghadamp/reg-research
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
state() { if [ "$1" = reg1 ]; then tmux has-session -t "=$2" 2>/dev/null && echo BUSY || echo IDLE; else ssh -o BatchMode=yes -o ConnectTimeout=20 "$1" "bash -lc \"tmux has-session -t =$2 2>/dev/null && echo BUSY || echo IDLE\"" 2>/dev/null | tail -n 1; fi; }
ended() { [ "$(state "$1" "$2")" = IDLE ] && sleep 60 && [ "$(state "$1" "$2")" = IDLE ]; }
launch() { local h=$1; shift; if [ "$h" = reg1 ]; then bash scripts/launch_grid.sh "$@" 2>&1; else ssh -o BatchMode=yes -o ConnectTimeout=20 "$h" "bash -lc '~/reg/scripts/launch_grid.sh $*'" 2>&1; fi; }
start() { local h=$1; shift; local out; until out=$(launch "$h" "$@") && echo "$out" | grep -q started; do say "$h launch failed ($out), retry in 5 min"; sleep 300; done; say "$out"; }
start reg1 jobs/reg_5k6k_ext.jsonl reg_5k6k_ext 0 4
start reg3 jobs/reg_5k6k_ext.jsonl reg_5k6k_ext 1 4
start reg4 jobs/reg_5k6k_ext.jsonl reg_5k6k_ext 2 4
start reg5 jobs/geneval_1k.jsonl geneval_1k 0 1
sleep 900; until ended reg5 job; do sleep 600; done
start reg5 jobs/reg_5k6k_ext.jsonl reg_5k6k_ext 3 4
say "handoff35: all four reg_5k6k_ext shards started (reg5 after geneval_1k)"
