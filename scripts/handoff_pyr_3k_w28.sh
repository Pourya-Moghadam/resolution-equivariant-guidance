#!/usr/bin/env bash
# Detached handoff (tmux "handoff5" on reg1): pyramid-only 3K w2 (100 rows) on reg4 (shard 0/2, after its pyr_3k w4 shard)
# and reg1 (shard 1/2, after the pyr w8 run). Pulls reg4's finished shard first. reg3 already runs pyr_3k w8 (launched by hand).
D=/data1/moghadamp/reg_data/outputs
LOG=$D/handoff_log.txt
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
busy() { ssh -o BatchMode=yes "$1" 'bash -lc "tmux has-session -t job 2>/dev/null"'; }
pull() { mkdir -p $D/$2; rsync -a --exclude 'log_*' $1:$D/$2/ $D/$2/; say "$1 $2 pulled (handoff5): $(ls $D/$2/*.json | wc -l) rows"; }
( until ! busy reg4; do sleep 600; done; pull reg4 reg_pyramid_grid
  say "$(ssh reg4 "bash -lc '~/reg/scripts/launch_grid.sh jobs/reg_pyramid_grid_pyr_3k_w2.jsonl reg_pyramid_grid 0 2'" 2>&1)" ) &
( until ! tmux has-session -t job 2>/dev/null; do sleep 600; done
  say "reg1 pyr w8 done: $(ls $D/reg_pyramid_grid/reg_pyr_4096_w8_*.json | wc -l) rows"
  say "$(bash ~/reg/scripts/launch_grid.sh jobs/reg_pyramid_grid_pyr_3k_w2.jsonl reg_pyramid_grid 1 2 2>&1)" ) &
wait
say "pyramid-only 3K w2 launches done"
