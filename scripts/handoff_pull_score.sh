#!/usr/bin/env bash
# Generic pull-and-score handoff, judged by record count rather than by tmux sessions (hosts join a grid at different times):
#   bash scripts/handoff_pull_score.sh NAME N [refs|norefs] [cat]
# Every 20 min copy the worker records of output NAME to reg1; when N records exist, pull the images, then score on reg1:
# run_pref (HPS, ImageReward), run_eval (CLIP, patch-CLIP, duplicates; patch-FID/KID against the 2K references when "refs"),
# speckle; with "cat", per-category tables as well (DrawBench). Error records count towards N and are reported in the log
# line. Hosts: $REG_WORKERS from env.sh (reg2 excluded there).
set -u
NAME=${1:?output name}; N=${2:?record count}; REFS=${3:-refs}; CAT=${4:-}
cd /data1/moghadamp/reg-research
. ./env.sh
D=$REG_DATA/outputs; O=$D/$NAME; LOG=$D/handoff_log.txt
mkdir -p "$O"
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
count() { find "$O" -maxdepth 1 -name '*.json' | wc -l; }
until [ "$(count)" -ge "$N" ]; do
  sleep 1200
  for h in $REG_WORKERS; do rsync -a -e 'ssh -o BatchMode=yes -o ConnectTimeout=20' --include='*.json' --exclude='*' "$h:$O/" "$O/" 2>/dev/null; done
done
for h in $REG_WORKERS; do
  until rsync -a -e 'ssh -o BatchMode=yes -o ConnectTimeout=20' --include='*.json' --include='*.png' --exclude='*' "$h:$O/" "$O/"; do say "$h $NAME rsync failed, retry in 5 min"; sleep 300; done
done
say "$NAME pulled: records $(count), ok $(grep -l '"status": "ok"' "$O"/*.json | wc -l), error $(grep -l '"status": "error"' "$O"/*.json | wc -l), images $(find "$O" -maxdepth 1 -name '*.png' | wc -l)"
S=$O/score_logs; mkdir -p "$S"
P=/data1/moghadamp/reg_pref_venv/bin/python
$P scripts/run_pref.py --out "$O" --group-by size method cfg --csv "results/${NAME}_pref.csv" > "$S/pref.log" 2>&1
[ -n "$CAT" ] && $P scripts/run_pref.py --out "$O" --group-by category method cfg --csv "results/${NAME}_pref_cat.csv" > "$S/pref_cat.log" 2>&1
if [ "$REFS" = refs ]; then
  python scripts/run_eval.py --out "$O" --refs "$REG_DATA/datasets/aesthetic_4k/eval/size_2048/images" --group-by size method cfg --skip ir --csv "results/${NAME}_eval.csv" > "$S/eval.log" 2>&1
else
  python scripts/run_eval.py --out "$O" --group-by size method cfg --skip ir fid --csv "results/${NAME}_eval.csv" > "$S/eval.log" 2>&1
  [ -n "$CAT" ] && python scripts/run_eval.py --out "$O" --group-by category method cfg --skip ir fid --csv "results/${NAME}_eval_cat.csv" > "$S/eval_cat.log" 2>&1
fi
python scripts/speckle.py --out "$O" --group-by size method cfg --csv "results/${NAME}_speckle.csv" > "$S/speckle.log" 2>&1
sed -i 's/\r$//' results/${NAME}_*.csv
say "$NAME complete and scored"
