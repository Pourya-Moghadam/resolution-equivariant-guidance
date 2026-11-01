#!/usr/bin/env bash
# Detached handoff (tmux "handoff7" on reg1): the candidate rule's 4K default (pyramid [2] + super 8 = w(r-2), w4) at grid
# scale, 50 rows: shard 0/2 on reg2 after its pyramid-only 4K w2 run (pull it first), shard 1/2 on reg1 after the 6K probe.
D=/data1/moghadamp/reg_data/outputs
LOG=$D/handoff_log.txt
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
busy() { ssh -o BatchMode=yes "$1" 'bash -lc "tmux has-session -t job 2>/dev/null"'; }
pull() { mkdir -p $D/$2; rsync -a --exclude 'log_*' $1:$D/$2/ $D/$2/; say "$1 $2 pulled (handoff7): $(ls $D/$2/*.json | wc -l) rows"; }
( until ! busy reg2; do sleep 600; done; pull reg2 reg_pyramid_grid
  say "pyr 4K w2 complete: $(ls $D/reg_pyramid_grid/reg_pyr_4096_w2_*.json | wc -l)/50 rows"
  say "$(ssh reg2 "bash -lc '~/reg/scripts/launch_grid.sh jobs/reg_pyramid_grid_pyr_s8.jsonl reg_pyramid_grid 0 2'" 2>&1)" ) &
( until ! tmux has-session -t job 2>/dev/null; do sleep 600; done
  say "reg1 6K probe done: $(grep -l '"status": "ok"' $D/reg_pag_pilot/*6144*.json 2>/dev/null | wc -l)/2 ok"
  say "$(bash ~/reg/scripts/launch_grid.sh jobs/reg_pyramid_grid_pyr_s8.jsonl reg_pyramid_grid 1 2 2>&1)" ) &
wait
say "pyr+s8 4K launches done"
