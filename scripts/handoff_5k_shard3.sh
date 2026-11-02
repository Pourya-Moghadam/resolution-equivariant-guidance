#!/usr/bin/env bash
# Detached handoff (tmux "handoff9" on reg1): reg5 is shared with another user's 12 GB job, so its 5K shard (25 rows) OOMed.
# Re-run those rows (jobs/reg_5k_grid_shard3.jsonl) on reg2/reg3/reg4 after their own shards (pull first), and the second
# half of the 3K rule curve (jobs/reg_anchor_grid_s12_3k_w28.jsonl, shard 1/2) on reg1 after its 3K w4 cell. reg5 runs shard 0/2.
D=/data1/moghadamp/reg_data/outputs
LOG=$D/handoff_log.txt
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
busy() { ssh -o BatchMode=yes "$1" 'bash -lc "tmux has-session -t job 2>/dev/null"'; }
pull() { mkdir -p $D/$2; rsync -a --exclude 'log_*' $1:$D/$2/ $D/$2/; say "$1 $2 pulled (handoff9): $(ls $D/$2/*.json | wc -l) rows"; }
i=0
for h in reg2 reg3 reg4; do
  ( until ! busy $h; do sleep 600; done; pull $h reg_5k_grid
    say "$(ssh $h "bash -lc '~/reg/scripts/launch_grid.sh jobs/reg_5k_grid_shard3.jsonl reg_5k_grid $i 3'" 2>&1)" ) &
  i=$((i+1))
done
( until ! tmux has-session -t job 2>/dev/null; do sleep 600; done
  say "reg1 3K single anchor + super 12 w4 done: $(ls $D/reg_pyramid_grid/reg_s12_3072_w4_*.json | wc -l)/100 rows"
  say "$(bash ~/reg/scripts/launch_grid.sh jobs/reg_anchor_grid_s12_3k_w28.jsonl reg_pyramid_grid 1 2 2>&1)" ) &
wait
say "5K shard3 + 3K curve launches done"
