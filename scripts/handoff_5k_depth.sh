#!/usr/bin/env bash
# Detached handoff (tmux "handoff6" on reg1): when reg5 finishes its pyr_3k w4 shard, pull it and launch the 5K pyramid-depth pilot.
D=/data1/moghadamp/reg_data/outputs
LOG=$D/handoff_log.txt
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
busy() { ssh -o BatchMode=yes "$1" 'bash -lc "tmux has-session -t job 2>/dev/null"'; }
pull() { mkdir -p $D/$2; rsync -a --exclude 'log_*' $1:$D/$2/ $D/$2/; say "$1 $2 pulled (handoff6): $(ls $D/$2/*.json | wc -l) rows"; }
until ! busy reg5; do sleep 600; done; pull reg5 reg_pyramid_grid
say "pyr 3k w4 complete: $(ls $D/reg_pyramid_grid/reg_pyr_3072_w4_*.json | wc -l)/100 rows"
say "$(ssh reg5 "bash -lc '~/reg/scripts/launch_grid.sh jobs/reg_5k_depth_pilot.jsonl reg_pag_pilot 0 1'" 2>&1)"
say "5K depth pilot launch done"
