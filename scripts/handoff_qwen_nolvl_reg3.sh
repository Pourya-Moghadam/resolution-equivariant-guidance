#!/usr/bin/env bash
# Detached handoff (tmux "handoff24" on reg1): reg3 runs jobs/qwen_4k_mu_nolvl.jsonl (38 rows, output qwen_4k_abl) first
# (its DrawBench shard 1/3 was stopped for it Mon 10:00); when that ends: pull, score qwen_4k_abl on reg1's GPU, and put
# reg3 back on jobs/drawbench_4k.jsonl shard 1/3. Session names matched exactly.
D=/data1/moghadamp/reg_data/outputs
O=$D/qwen_4k_abl
LOG=$D/handoff_log.txt
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
state() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$1" "bash -lc \"tmux has-session -t =$2 2>/dev/null && echo BUSY || echo IDLE\"" 2>/dev/null | tail -n 1; }
ended() { [ "$(state "$1" "$2")" = IDLE ] && sleep 60 && [ "$(state "$1" "$2")" = IDLE ]; }
until sleep 600 && ended reg3 job; do :; done
until rsync -a --include='*.json' --include='*.png' --exclude='*' reg3:$O/ $O/; do say "reg3 rsync failed, retry in 5 min"; sleep 300; done
say "reg3 qwen_4k_abl pulled: mu235nolvl ok: $(grep -l '"status": "ok"' $O/qreg_4096_w4_mu235nolvl_*.json | wc -l), error: $(grep -l '"status": "error"' $O/qreg_4096_w4_mu235nolvl_*.json | wc -l)"
until out=$(ssh -o BatchMode=yes -o ConnectTimeout=20 reg3 "bash -lc '~/reg/scripts/launch_grid.sh jobs/drawbench_4k.jsonl drawbench_4k 1 3'" 2>&1) && echo "$out" | grep -q started; do say "reg3 launch failed ($out), retry in 5 min"; sleep 300; done
say "$out"
cd /data1/moghadamp/reg-research
export REG_DATA=/data1/moghadamp/reg_data
S=$O/score_logs; mkdir -p $S
/data1/moghadamp/reg_pref_venv/bin/python scripts/run_pref.py --out $O --group-by size method cfg --csv results/qwen_4k_abl_pref.csv > $S/pref.log 2>&1
( . ./env.sh
  python scripts/run_eval.py --out $O --refs $REG_DATA/datasets/aesthetic_4k/eval/size_2048/images --group-by size method cfg --skip ir --csv results/qwen_4k_abl_eval.csv > $S/eval.log 2>&1
  python scripts/speckle.py --out $O --group-by size method cfg --csv results/qwen_4k_abl_speckle.csv > $S/speckle.log 2>&1 )
say "Qwen 4K rule-as-written cell (mu 2.35, no 2048 anchor) complete and scored"
