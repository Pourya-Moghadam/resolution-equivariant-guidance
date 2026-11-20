#!/usr/bin/env bash
# Detached handoff (tmux "handoff41" on reg1), 2026-11-20: jobs/reg_4k_yarn.jsonl (250 rows, outputs/reg_4k_yarn) in four
# shards on reg1, reg3, reg4, reg5 (all idle). Pull and score: handoff_pull_score.sh reg_4k_yarn 250 refs (handoff42).
LOG=/data1/moghadamp/reg_data/outputs/handoff_log.txt
cd /data1/moghadamp/reg-research
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
launch() { local h=$1; shift; if [ "$h" = reg1 ]; then bash scripts/launch_grid.sh "$@" 2>&1; else ssh -o BatchMode=yes -o ConnectTimeout=20 "$h" "bash -lc '~/reg/scripts/launch_grid.sh $*'" 2>&1; fi; }
start() { local h=$1; shift; local out; until out=$(launch "$h" "$@") && echo "$out" | grep -q started; do say "$h launch failed ($out), retry in 5 min"; sleep 300; done; say "$out"; }
i=0; for h in reg1 reg3 reg4 reg5; do start $h jobs/reg_4k_yarn.jsonl reg_4k_yarn $i 4; i=$((i+1)); done
say "handoff41: reg_4k_yarn started on four cards"
