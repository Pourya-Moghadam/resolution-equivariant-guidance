#!/usr/bin/env bash
# Detached handoff (tmux "handoff19" on reg1): the 4K texture ablation (jobs/qwen_4k_abl.jsonl, 48 rows) runs as shards
# 0/4 reg4, 2/4 reg3, 3/4 reg2 (launched by hand Fri 10:40, when those workers had sat idle behind reg1's last w16 row).
# Once reg1's w16 shard has ended (handoff18's completion line), this scores the qwen_grid rows on reg1's free GPU
# (resume-safe: only the 50 new plain-w16 rows are computed; CSVs under results/) and then launches reg1's shard 1/4.
D=/data1/moghadamp/reg_data/outputs
LOG=$D/handoff_log.txt
S=$D/qwen_grid/score_logs; mkdir -p $S
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
until grep -q "Qwen 4K w16 baseline complete" "$LOG" && ! tmux has-session -t job 2>/dev/null; do sleep 120; done
cd /data1/moghadamp/reg-research
export REG_DATA=/data1/moghadamp/reg_data
/data1/moghadamp/reg_pref_venv/bin/python scripts/run_pref.py --out $D/qwen_grid --group-by size method cfg --csv results/qwen_grid_pref.csv > $S/pref.log 2>&1
( . ./env.sh
  python scripts/run_eval.py --out $D/qwen_grid --refs $REG_DATA/datasets/aesthetic_4k/eval/size_2048/images --group-by size method cfg --skip ir --csv results/qwen_grid_eval.csv > $S/eval.log 2>&1
  python scripts/speckle.py --out $D/qwen_grid --group-by size method cfg --csv results/qwen_grid_speckle.csv > $S/speckle.log 2>&1 )
say "qwen_grid scored with the 4K w16 rows (results/qwen_grid_{pref,eval,speckle}.csv)"
say "$(bash scripts/launch_grid.sh jobs/qwen_4k_abl.jsonl qwen_4k_abl 1 4 2>&1)"
