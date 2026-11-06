#!/usr/bin/env bash
# Detached handoff (tmux "handoff20" on reg1): pull the Qwen 4K texture-ablation rows (jobs/qwen_4k_abl.jsonl, output
# qwen_4k_abl) from reg4, reg3 and reg2 as each shard ends (explicit IDLE twice; rsync retried); when reg1's shard has
# ended too, score the rows on reg1's GPU (results/qwen_4k_abl_{pref,eval,speckle}.csv) and log the completion line.
D=/data1/moghadamp/reg_data/outputs
O=$D/qwen_4k_abl
LOG=$D/handoff_log.txt
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
state() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$1" 'bash -lc "tmux has-session -t job 2>/dev/null && echo BUSY || echo IDLE"' 2>/dev/null | tail -n 1; }
ended() { [ "$(state "$1")" = IDLE ] && sleep 60 && [ "$(state "$1")" = IDLE ]; }
pull() {
  until rsync -a --include='*.json' --include='*.png' --exclude='*' "$1":$O/ $O/; do say "$1 rsync failed, retry in 5 min"; sleep 300; done
  say "$1 qwen_4k_abl pulled: ok: $(grep -l '"status": "ok"' $O/*.json 2>/dev/null | wc -l), error: $(grep -l '"status": "error"' $O/*.json 2>/dev/null | wc -l)"
}
r4=0; r3=0; r2=0; r1=0
while [ $r4 = 0 ] || [ $r3 = 0 ] || [ $r2 = 0 ] || [ $r1 = 0 ]; do
  if [ $r4 = 0 ] && ended reg4; then pull reg4; r4=1; fi
  if [ $r3 = 0 ] && ended reg3; then pull reg3; r3=1; fi
  if [ $r2 = 0 ] && ended reg2; then pull reg2; r2=1; fi
  if [ $r1 = 0 ] && grep -q "done:" $O/log_bri-p7-1.txt && ! tmux has-session -t job 2>/dev/null; then say "reg1 ablation shard 1/4 done"; r1=1; fi
  sleep 600
done
cd /data1/moghadamp/reg-research
export REG_DATA=/data1/moghadamp/reg_data
S=$O/score_logs; mkdir -p $S
/data1/moghadamp/reg_pref_venv/bin/python scripts/run_pref.py --out $O --group-by size method cfg --csv results/qwen_4k_abl_pref.csv > $S/pref.log 2>&1
( . ./env.sh
  python scripts/run_eval.py --out $O --refs $REG_DATA/datasets/aesthetic_4k/eval/size_2048/images --group-by size method cfg --skip ir --csv results/qwen_4k_abl_eval.csv > $S/eval.log 2>&1
  python scripts/speckle.py --out $O --group-by size method cfg --csv results/qwen_4k_abl_speckle.csv > $S/speckle.log 2>&1 )
say "Qwen 4K ablation complete and scored: $(ls $O/*.png | wc -l) images on reg1"
