#!/usr/bin/env bash
# Detached handoff (tmux "handoff4" on reg1): the new default (pyramid-only anchor) at 3K w4 (100 rows, reg4+reg5 after the
# PAG-scale and 5K pilots) and at 4K w2 (50 rows, reg2 after its YaRN shard). Pulls the finished job itself before launching.
D=/data1/moghadamp/reg_data/outputs
LOG=$D/handoff_log.txt
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
busy() { ssh -o BatchMode=yes "$1" 'bash -lc "tmux has-session -t job 2>/dev/null"'; }
pull() { mkdir -p $D/$2; rsync -a --exclude 'log_*' $1:$D/$2/ $D/$2/; say "$1 $2 pulled (handoff4): $(ls $D/$2/*.json | wc -l) rows"; }
for i in 0 1; do
  h=$([ $i = 0 ] && echo reg4 || echo reg5)
  ( until ! busy $h; do sleep 600; done; pull $h reg_pag_pilot
    say "$(ssh $h "bash -lc '~/reg/scripts/launch_grid.sh jobs/reg_pyramid_grid_pyr_3k.jsonl reg_pyramid_grid $i 2'" 2>&1)" ) &
done
( until ! busy reg2; do sleep 600; done; pull reg2 reg_generality_grid
  say "$(ssh reg2 "bash -lc '~/reg/scripts/launch_grid.sh jobs/reg_pyramid_grid_pyr_w2.jsonl reg_pyramid_grid 0 1'" 2>&1)" ) &
wait
say "pyramid-default launches done"
