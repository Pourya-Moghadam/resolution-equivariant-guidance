#!/usr/bin/env bash
# Detached handoff (tmux "handoff11" on reg1): reg5's shard 0/2 of the 3K rule curve never ran (run_grid's 20 GB free-memory
# guard timed out beside the stale 12 GB process), so run it on reg1 after reg1's own shard 1/2.
D=/data1/moghadamp/reg_data/outputs
LOG=$D/handoff_log.txt
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
until ! tmux has-session -t job 2>/dev/null; do sleep 600; done
say "reg1 3K curve shard 1/2 done: $(ls $D/reg_pyramid_grid/reg_s12_3072_w[28]_*.json | wc -l)/200 w2/w8 rows"
say "$(bash ~/reg/scripts/launch_grid.sh jobs/reg_anchor_grid_s12_3k_w28.jsonl reg_pyramid_grid 0 2 2>&1)"
until ! tmux has-session -t job 2>/dev/null; do sleep 600; done
say "3K rule curve complete: $(ls $D/reg_pyramid_grid/reg_s12_3072_w[28]_*.json | wc -l)/200 w2/w8 rows"
