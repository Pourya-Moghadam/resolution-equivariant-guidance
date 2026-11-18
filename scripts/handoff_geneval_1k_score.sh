#!/usr/bin/env bash
# Detached (tmux "handoff38" on reg1): once "geneval_1k complete and scored" is logged (handoff36), run the official
# GenEval evaluator on the native-1K reference set, then the paired comparison 4K REG vs 1K native is a script call.
cd /data1/moghadamp/reg-research; source env.sh
LOG=/data1/moghadamp/reg_data/outputs/handoff_log.txt
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
until grep -q "geneval_1k complete and scored" "$LOG"; do sleep 1800; done
say "handoff38: running the official GenEval evaluator on geneval_1k"
if python scripts/run_geneval.py --out $REG_DATA/outputs/geneval_1k --csv results/geneval_1k_geneval.csv > $REG_DATA/outputs/geneval_1k/log_geneval_eval.txt 2>&1; then
  say "handoff38: GenEval 1K scored -> results/geneval_1k_geneval.csv"; else say "handoff38: run_geneval.py FAILED on geneval_1k"; fi
