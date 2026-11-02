#!/usr/bin/env bash
# Detached handoff (tmux "handoff8" on reg1): pull the 5K grid shards from reg2-reg5 as each finishes and report reg1's 3K rule cell.
D=/data1/moghadamp/reg_data/outputs
LOG=$D/handoff_log.txt
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
busy() { ssh -o BatchMode=yes "$1" 'bash -lc "tmux has-session -t job 2>/dev/null"'; }
pull() { mkdir -p $D/$2; rsync -a --exclude 'log_*' $1:$D/$2/ $D/$2/; say "$1 $2 pulled (handoff8): $(ls $D/$2/*.json | wc -l) rows"; }
for h in reg2 reg3 reg4 reg5; do ( until ! busy $h; do sleep 600; done; pull $h reg_5k_grid ) & done
( until ! tmux has-session -t job 2>/dev/null; do sleep 600; done; say "reg1 3K single anchor + super 12 done: $(ls $D/reg_pyramid_grid/reg_s12_3072_w4_*.json | wc -l)/100 rows" ) &
wait
say "5K grid pulls done: $(ls $D/reg_5k_grid/*.json | wc -l)/100 rows; status: $(grep -ho '"status": "[a-z]*"' $D/reg_5k_grid/*.json | sort | uniq -c | tr '\n' ' ')"
