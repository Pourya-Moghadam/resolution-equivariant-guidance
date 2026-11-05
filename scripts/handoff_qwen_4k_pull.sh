#!/usr/bin/env bash
# Detached handoff (tmux "handoff17" on reg1): pull the Qwen 4K grid rows from reg4, reg3 and reg2 as each shard ends and
# log when all four shards (reg1's included) are in; the local shard 1/4 ends with a "done:" line in log_bri-p7-1.txt.
D=/data1/moghadamp/reg_data/outputs
LOG=$D/handoff_log.txt
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
busy() { ssh -o BatchMode=yes "$1" 'bash -lc "tmux has-session -t job 2>/dev/null"'; }
pull() { rsync -a --include='*.json' --include='*.png' --exclude='*' "$1":$D/qwen_grid/ $D/qwen_grid/; say "$1 qwen_grid pulled: 4K ok: $(grep -l '"status": "ok"' $D/qwen_grid/q*_4096_*.json 2>/dev/null | wc -l), 4K error: $(grep -l '"status": "error"' $D/qwen_grid/q*_4096_*.json 2>/dev/null | wc -l)"; }
r4=0; r3=0; r2=0; r1=0
while [ $r4 = 0 ] || [ $r3 = 0 ] || [ $r2 = 0 ] || [ $r1 = 0 ]; do
  if [ $r4 = 0 ] && ! busy reg4; then pull reg4; r4=1; fi
  if [ $r3 = 0 ] && ! busy reg3; then pull reg3; r3=1; fi
  if [ $r2 = 0 ] && ! busy reg2; then pull reg2; r2=1; fi
  if [ $r1 = 0 ] && grep -q "done:" $D/qwen_grid/log_bri-p7-1.txt; then say "reg1 4K shard 1/4 done"; r1=1; fi
  sleep 600
done
say "Qwen 4K grid complete: $(grep -l '"status": "ok"' $D/qwen_grid/q*_4096_*.json | wc -l) ok rows on reg1"
