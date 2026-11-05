#!/usr/bin/env bash
# Detached handoff (tmux "handoff17" on reg1): pull the Qwen 4K grid rows from reg4, reg3 and reg2 as each shard ends and
# log when all four shards (reg1's included) are in; the local shard 1/4 ends with a "done:" line in log_bri-p7-1.txt.
# A worker counts as ended only when ssh answers IDLE twice, 60 s apart: a failed ssh (broken pipe) is not "ended"
# (the first version read it so and marked all three workers done on Wed night), and a pull is retried until rsync exits 0.
D=/data1/moghadamp/reg_data/outputs
LOG=$D/handoff_log.txt
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
state() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$1" 'bash -lc "tmux has-session -t job 2>/dev/null && echo BUSY || echo IDLE"' 2>/dev/null | tail -n 1; }
ended() { [ "$(state "$1")" = IDLE ] && sleep 60 && [ "$(state "$1")" = IDLE ]; }
pull() {
  until rsync -a --include='*.json' --include='*.png' --exclude='*' "$1":$D/qwen_grid/ $D/qwen_grid/; do say "$1 rsync failed, retry in 5 min"; sleep 300; done
  say "$1 qwen_grid pulled: 4K ok: $(grep -l '"status": "ok"' $D/qwen_grid/q*_4096_*.json 2>/dev/null | wc -l), 4K error: $(grep -l '"status": "error"' $D/qwen_grid/q*_4096_*.json 2>/dev/null | wc -l)"
}
r4=0; r3=0; r2=0; r1=0
while [ $r4 = 0 ] || [ $r3 = 0 ] || [ $r2 = 0 ] || [ $r1 = 0 ]; do
  if [ $r4 = 0 ] && ended reg4; then pull reg4; r4=1; fi
  if [ $r3 = 0 ] && ended reg3; then pull reg3; r3=1; fi
  if [ $r2 = 0 ] && ended reg2; then pull reg2; r2=1; fi
  if [ $r1 = 0 ] && grep -q "done:" $D/qwen_grid/log_bri-p7-1.txt; then say "reg1 4K shard 1/4 done"; r1=1; fi
  sleep 600
done
say "Qwen 4K grid complete: $(ls $D/qwen_grid/q*_4096_*.png | wc -l) images on reg1"
