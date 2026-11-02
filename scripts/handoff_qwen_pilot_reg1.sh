#!/usr/bin/env bash
# Detached handoff (tmux "handoff12" on reg1): after the 3K rule curve completes on reg1 (handoff11), run shard 1/2 of the
# Qwen pilot there (shard 0/2 goes on reg4 by hand once the port check passes).
D=/data1/moghadamp/reg_data/outputs
LOG=$D/handoff_log.txt
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
until grep -q "3K rule curve complete" $LOG && ! tmux has-session -t job 2>/dev/null; do sleep 600; done
say "$(bash ~/reg/scripts/launch_grid.sh jobs/qwen_pilot.jsonl qwen_pilot 1 2 2>&1)"
