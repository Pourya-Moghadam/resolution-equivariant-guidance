#!/usr/bin/env bash
# Detached handoff (tmux "handoff25" on reg1): DrawBench at 4K on Lumina runs on reg1 (plain w4), reg5 (plain w16),
# reg4 and reg3 (REG rows, parts a and b). Completion is judged by records, not by tmux session names (reg3 and reg4
# start their parts later, from other handoffs): every 20 min the worker records are copied to reg1, and when all 600
# ids have a record the images are pulled and the set is scored (HPS, ImageReward, CLIP, duplication; overall and per
# DrawBench category). Error records count towards the 600 and are reported in the log line. reg2 is not used.
D=/data1/moghadamp/reg_data/outputs
O=$D/drawbench_4k
LOG=$D/handoff_log.txt
N=600
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
count() { find $O -maxdepth 1 -name 'db_*.json' | wc -l; }
until [ "$(count)" -ge $N ]; do
  sleep 1200
  for h in reg3 reg4 reg5; do rsync -a -e 'ssh -o BatchMode=yes -o ConnectTimeout=20' --include='db_*.json' --exclude='*' $h:$O/ $O/ 2>/dev/null; done
done
for h in reg3 reg4 reg5; do
  until rsync -a -e 'ssh -o BatchMode=yes -o ConnectTimeout=20' --include='db_*.json' --include='db_*.png' --exclude='*' $h:$O/ $O/; do say "$h drawbench rsync failed, retry in 5 min"; sleep 300; done
done
say "drawbench_4k pulled: records $(count), ok $(grep -l '"status": "ok"' $O/db_*.json | wc -l), error $(grep -l '"status": "error"' $O/db_*.json | wc -l), images $(find $O -maxdepth 1 -name 'db_*.png' | wc -l)"
cd /data1/moghadamp/reg-research
export REG_DATA=/data1/moghadamp/reg_data
S=$O/score_logs; mkdir -p $S
P=/data1/moghadamp/reg_pref_venv/bin/python
$P scripts/run_pref.py --out $O --group-by method cfg --csv results/drawbench_4k_pref.csv > $S/pref.log 2>&1
$P scripts/run_pref.py --out $O --group-by category method cfg --csv results/drawbench_4k_pref_cat.csv > $S/pref_cat.log 2>&1
( . ./env.sh
  python scripts/run_eval.py --out $O --group-by method cfg --skip ir fid --csv results/drawbench_4k_eval.csv > $S/eval.log 2>&1
  python scripts/run_eval.py --out $O --group-by category method cfg --skip ir fid --csv results/drawbench_4k_eval_cat.csv > $S/eval_cat.log 2>&1
  python scripts/speckle.py --out $O --group-by method cfg --csv results/drawbench_4k_speckle.csv > $S/speckle.log 2>&1 )
sed -i 's/\r$//' results/drawbench_4k_*.csv
say "DrawBench 4K complete and scored"
