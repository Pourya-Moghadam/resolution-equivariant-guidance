#!/usr/bin/env bash
# Detached handoff (tmux "handoff16" on reg1): when the Qwen 3K plain-w16 shards end on reg3 and reg2, pull their rows and
# launch the Qwen 4K grid shards 2/4 (reg3) and 3/4 (reg2); shards 0/4 (reg4) and 1/4 (reg1) started Wed 11:00.
D=/data1/moghadamp/reg_data/outputs
LOG=$D/handoff_log.txt
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
busy() { ssh -o BatchMode=yes "$1" 'bash -lc "tmux has-session -t job 2>/dev/null"'; }
pull() { rsync -a --include='*.json' --include='*.png' --exclude='*' "$1":$D/qwen_grid/ $D/qwen_grid/; say "$1 qwen_grid pulled: $(ls $D/qwen_grid/*.json | wc -l) rows on reg1, w16 ok: $(grep -l '"status": "ok"' $D/qwen_grid/qcfg_3072_w16_*.json 2>/dev/null | wc -l)"; }
launch() { say "$(ssh -o BatchMode=yes "$1" "bash -lc '~/reg/scripts/launch_grid.sh jobs/qwen_grid_4k.jsonl qwen_grid $2 4'" 2>&1)"; }
r3=0; r2=0
while [ $r3 = 0 ] || [ $r2 = 0 ]; do
  if [ $r3 = 0 ] && ! busy reg3; then pull reg3; launch reg3 2; r3=1; fi
  if [ $r2 = 0 ] && ! busy reg2; then pull reg2; launch reg2 3; r2=1; fi
  sleep 600
done
say "Qwen w16 baseline done, 4K shards 2/4 and 3/4 launched"
