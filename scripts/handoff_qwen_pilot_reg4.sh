#!/usr/bin/env bash
# Detached handoff (tmux "handoff13" on reg1): when reg4's Qwen pilot shard 0/2 (the first 24 rows) ends, pull its rows and
# relaunch the same shard on reg4 so it picks up the appended 3K schedule-control rows (24-47). reg1's shard 1/2 (handoff12)
# reads the extended file when it starts, so it needs nothing.
D=/data1/moghadamp/reg_data/outputs
LOG=$D/handoff_log.txt
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
busy() { ssh -o BatchMode=yes reg4 'bash -lc "tmux has-session -t job 2>/dev/null"'; }
while busy; do sleep 600; done
rsync -a --include='*.json' --include='*.png' --exclude='*' reg4:$D/qwen_pilot/ $D/qwen_pilot/
say "reg4 Qwen pilot first shard done: $(ls $D/qwen_pilot/*.json | wc -l) rows pulled"
say "$(ssh -o BatchMode=yes reg4 'bash -lc "~/reg/scripts/launch_grid.sh jobs/qwen_pilot.jsonl qwen_pilot 0 2"' 2>&1)"
