#!/usr/bin/env bash
# Detached handoff (tmux "handoff47" on reg1), 2026-11-23: jobs/reg_4k_baselines2.jsonl (300 rows, outputs/reg_4k_baselines2) in four
# shards on reg1, reg3, reg4, reg5 (all idle). Pull and score: handoff_pull_score.sh reg_4k_baselines2 300 refs (handoff48).
LOG=/data1/moghadamp/reg_data/outputs/handoff_log.txt
cd /data1/moghadamp/reg-research
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
launch() { local h=$1; shift; if [ "$h" = reg1 ]; then bash scripts/launch_grid.sh "$@" 2>&1; else ssh -o BatchMode=yes -o ConnectTimeout=20 "$h" "bash -lc '~/reg/scripts/launch_grid.sh $*'" 2>&1; fi; }
start() { local h=$1; shift; local out; until out=$(launch "$h" "$@") && echo "$out" | grep -q started; do say "$h launch failed ($out), retry in 5 min"; sleep 300; done; say "$out"; }
i=0; for h in reg1 reg3 reg4 reg5; do start $h jobs/reg_4k_baselines2.jsonl reg_4k_baselines2 $i 4; i=$((i+1)); done
say "handoff47: reg_4k_baselines2 started on four cards"
