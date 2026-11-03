#!/usr/bin/env bash
# Detached handoff (tmux "handoff15" on reg1): run the Qwen 3K plain-w16 baseline (jobs/qwen_grid_3k_w16.jsonl) on the two
# 32 GB cards that free tonight: shard 0/2 on reg3 after its 5K pyramid-only shard, shard 1/2 on reg2 after its 5K plain
# shard. Pulls each card's 5K rows first. reg5 is skipped (another user's job leaves < 28 GB, which the Qwen 3K guard needs).
D=/data1/moghadamp/reg_data/outputs
LOG=$D/handoff_log.txt
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
busy() { ssh -o BatchMode=yes "$1" 'bash -lc "tmux has-session -t job 2>/dev/null"'; }
pull5k() { rsync -a --include='*.json' --include='*.png' --exclude='*' "$1":$D/reg_5k_grid/ $D/reg_5k_grid/; say "$1 reg_5k_grid pulled: $(ls $D/reg_5k_grid/*.json | wc -l) rows on reg1"; }
launch() { say "$(ssh -o BatchMode=yes "$1" "bash -lc '~/reg/scripts/launch_grid.sh jobs/qwen_grid_3k_w16.jsonl qwen_grid $2 2'" 2>&1)"; }
r3=0; r2=0
while [ $r3 = 0 ] || [ $r2 = 0 ]; do
  if [ $r3 = 0 ] && ! busy reg3; then pull5k reg3; launch reg3 0; r3=1; fi
  if [ $r2 = 0 ] && ! busy reg2; then pull5k reg2; launch reg2 1; r2=1; fi
  sleep 600
done
say "Qwen w16 baseline launches done"
