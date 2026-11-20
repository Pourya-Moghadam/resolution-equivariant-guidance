#!/usr/bin/env bash
# Score the merged 5K/6K n = 100 cells (scripts/merge_5k6k_n100.py) and the paired differences (tmux "score56").
cd /data1/moghadamp/reg-research; . ./env.sh; P=/data1/moghadamp/reg_pref_venv/bin/python
LOG=/data1/moghadamp/reg_data/outputs/handoff_log.txt; say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
for k in 5k 6k; do N=lumina_${k}_n100; O=$REG_DATA/outputs/$N; S=$O/score_logs; mkdir -p $S
  $P scripts/run_pref.py --out $O --group-by size method cfg --csv results/${N}_pref.csv > $S/pref.log 2>&1
  python scripts/run_eval.py --out $O --refs $REG_DATA/datasets/aesthetic_4k/eval/size_2048/images --group-by size method cfg --skip ir --csv results/${N}_eval.csv > $S/eval.log 2>&1
  python scripts/speckle.py --out $O --group-by size method cfg --csv results/${N}_speckle.csv > $S/speckle.log 2>&1
done
for m in hps ir clip; do
  python scripts/reg_paired.py --metric $m --size 5120 --ref anchor_base_pyr_s12 --ref-cfg 4 --dirs lumina_5k_n100 --csv results/reg_paired_5k_n100_$m.csv
  python scripts/reg_paired.py --metric $m --size 6144 --ref anchor_base_pyr_s12 --ref-cfg 4 --dirs lumina_6k_n100 --csv results/reg_paired_6k_n100_$m.csv
done > $REG_DATA/outputs/lumina_6k_n100/score_logs/paired.log 2>&1
sed -i 's/\r$//' results/lumina_[56]k_n100_*.csv results/reg_paired_[56]k_n100_*.csv
say "score56: lumina_5k_n100 and lumina_6k_n100 scored"
