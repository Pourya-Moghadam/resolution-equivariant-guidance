#!/usr/bin/env bash
# Detached handoff (tmux "handoff3" on reg1): pull the Saturday-evening jobs as each worker finishes
#   reg2/reg3 -> reg_generality_grid (YaRN at grid scale); reg4 -> reg_pag_pilot (PAG scale 1/0.5); reg5 -> reg_pag_pilot (5K sweep)
D=/data1/moghadamp/reg_data/outputs
LOG=$D/handoff_log.txt
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
busy() { ssh -o BatchMode=yes "$1" 'bash -lc "tmux has-session -t job 2>/dev/null"'; }
pull() { mkdir -p $D/$2; rsync -a --exclude 'log_*' $1:$D/$2/ $D/$2/; say "$1 $2 done and pulled: $(ls $D/$2/*.json | wc -l) rows; status: $(grep -h -o '"status": "[a-z]*"' $D/$2/*.json | sort | uniq -c | tr '\n' ' ')"; }
for h in reg2 reg3; do ( until ! busy $h; do sleep 600; done; pull $h reg_generality_grid ) & done
for h in reg4 reg5; do ( until ! busy $h; do sleep 600; done; pull $h reg_pag_pilot ) & done
( while tmux has-session -t job 2>/dev/null; do sleep 600; done; say "reg1 pyr w8 done: $(ls $D/reg_pyramid_grid/reg_pyr_4096_w8_*.json 2>/dev/null | wc -l)/50 rows" ) &
wait
say "saturday-evening pulls done"
