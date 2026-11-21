#!/usr/bin/env bash
# Score the merged 4K DyPE-YaRN n = 100 cells (scripts/merge_4k_yarn_n100.py) and the paired differences against REG.
cd /data1/moghadamp/reg-research; . ./env.sh; P=/data1/moghadamp/reg_pref_venv/bin/python
N=lumina_4k_yarn_n100; O=$REG_DATA/outputs/$N; S=$O/score_logs; mkdir -p $S
$P scripts/run_pref.py --out $O --group-by size method cfg --csv results/${N}_pref.csv > $S/pref.log 2>&1
python scripts/run_eval.py --out $O --refs $REG_DATA/datasets/aesthetic_4k/eval/size_2048/images --group-by size method cfg --skip ir --csv results/${N}_eval.csv > $S/eval.log 2>&1
python scripts/speckle.py --out $O --group-by size method cfg --csv results/${N}_speckle.csv > $S/speckle.log 2>&1
for m in hps ir clip; do
  python scripts/reg_paired.py --metric $m --size 4096 --ref reg_yarn --ref-cfg 4 --dirs $N --csv results/reg_paired_4k_yarn_n100_$m.csv
  python scripts/reg_paired.py --metric $m --size 4096 --ref cfg_yarn --ref-cfg 16 --dirs $N --csv results/cfg16_paired_4k_yarn_n100_$m.csv
done > $S/paired.log 2>&1
sed -i 's/\r$//' results/${N}_*.csv results/*_paired_4k_yarn_n100_*.csv
echo "$(date '+%a %H:%M') score_yarn: lumina_4k_yarn_n100 scored" | tee -a /data1/moghadamp/reg_data/outputs/handoff_log.txt
