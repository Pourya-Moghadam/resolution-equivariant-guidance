#!/usr/bin/env bash
# Detached handoff (tmux "handoff34" on reg1): once handoff33 logs "geneval_4k complete and scored" (HPS/CLIP/speckle),
# run the official GenEval evaluator on reg1's GPU (free by then), then bring the SSD copy up to date so the
# switch to /ssd2 (scripts/ssd_switch.sh) is a short step.
cd /data1/moghadamp/reg-research; source env.sh
LOG=$REG_DATA/outputs/handoff_log.txt
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
until grep -q "geneval_4k complete and scored" "$LOG"; do sleep 1800; done
until grep -q "ssdcopy: initial copy done" "$LOG"; do sleep 600; done
say "handoff34: running the official GenEval evaluator on geneval_4k"
if python scripts/run_geneval.py --out $REG_DATA/outputs/geneval_4k --csv results/geneval_4k_geneval.csv > $REG_DATA/outputs/geneval_4k/log_geneval_eval.txt 2>&1; then
  say "handoff34: GenEval scored -> results/geneval_4k_geneval.csv ($(wc -l < results/geneval_4k_geneval.csv) lines)"
else say "handoff34: run_geneval.py FAILED, see outputs/geneval_4k/log_geneval_eval.txt"; fi
until ionice -c3 nice -n 19 rsync -a --delete /data1/moghadamp/reg_data/ /ssd2/moghadamp/reg_data/; do say "handoff34: delta rsync failed, retry in 5 min"; sleep 300; done
say "handoff34: SSD delta synced, ready to switch (scripts/ssd_switch.sh)"
