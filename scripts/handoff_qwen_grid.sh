#!/usr/bin/env bash
# Detached handoff (tmux "handoff14" on reg1): when the Qwen plain-weight pilot shards end on reg1 and reg4, pull reg4's rows and
# launch the Qwen 3K grid (jobs/qwen_grid_3k.jsonl) as shard 1/2 on reg1 and 0/2 on reg4.
D=/data1/moghadamp/reg_data/outputs
LOG=$D/handoff_log.txt
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
busy() { ssh -o BatchMode=yes "$1" 'bash -lc "tmux has-session -t job 2>/dev/null"'; }
while tmux has-session -t job 2>/dev/null || busy reg4; do sleep 600; done
rsync -a --include='*.json' --include='*.png' --exclude='*' reg4:$D/qwen_pilot/ $D/qwen_pilot/
say "Qwen plain-weight pilot done: $(ls $D/qwen_pilot/*.json | wc -l) pilot rows on reg1"
say "$(bash ~/reg/scripts/launch_grid.sh jobs/qwen_grid_3k.jsonl qwen_grid 1 2 2>&1)"
say "$(ssh -o BatchMode=yes reg4 'bash -lc "~/reg/scripts/launch_grid.sh jobs/qwen_grid_3k.jsonl qwen_grid 0 2"' 2>&1)"
say "Qwen grid launches done"
