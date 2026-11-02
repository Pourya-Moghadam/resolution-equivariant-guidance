#!/usr/bin/env bash
# Detached handoff (tmux "handoff10" on reg1): 5K plain w4/w8 (100 rows) over reg3 (shard 0/3, launched by hand), reg2 and
# reg4 (shards 1/3, 2/3 after their shard3 rows; pull first); the 5K pyramid-only ablation (50 rows) on reg5 once the stale
# 12 GB process (pid 2186845, another user's, admin notified) is gone and nothing else is running there.
D=/data1/moghadamp/reg_data/outputs
LOG=$D/handoff_log.txt
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
busy() { ssh -o BatchMode=yes "$1" 'bash -lc "tmux has-session -t job 2>/dev/null"'; }
pull() { mkdir -p $D/$2; rsync -a --exclude 'log_*' $1:$D/$2/ $D/$2/; say "$1 $2 pulled (handoff10): $(ls $D/$2/*.json | wc -l) rows"; }
i=1
for h in reg2; do
  ( until ! busy $h; do sleep 600; done; pull $h reg_5k_grid
    say "$(ssh $h "bash -lc '~/reg/scripts/launch_grid.sh jobs/reg_5k_plain_w48.jsonl reg_5k_grid $i 3'" 2>&1)" ) &
  i=$((i+1))
done
( until ! busy reg4; do sleep 600; done; pull reg4 reg_5k_grid; say "reg4 idle: held for the Qwen port test (plain shard 2/3 launched by hand afterwards)" ) &
( until ! busy reg5 && ! ssh -o BatchMode=yes reg5 'ps -p 2186845 >/dev/null 2>&1'; do sleep 900; done
  say "reg5 free: stale process gone, $(ssh -o BatchMode=yes reg5 'nvidia-smi --query-gpu=memory.used --format=csv,noheader') used"
  say "$(ssh reg5 "bash -lc '~/reg/scripts/launch_grid.sh jobs/reg_5k_pyr_only.jsonl reg_5k_grid 0 1'" 2>&1)" ) &
wait
say "5K plain curve + pyramid-only launches done"
