#!/usr/bin/env bash
# Detached handoff (tmux "handoff" on reg1): pull each worker's reg_pyramid_grid rows (3K REG on reg2/reg3, REG w2/w8 on reg4/reg5) when its job ends.
D=/data1/moghadamp/reg_data/outputs
LOG=$D/handoff_log.txt
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
busy() { ssh -o BatchMode=yes "$1" 'bash -lc "tmux has-session -t job 2>/dev/null"'; }
for h in reg2 reg3 reg4 reg5; do
  ( until ! busy $h; do sleep 600; done
    rsync -a --exclude 'log_*' $h:$D/reg_pyramid_grid/ $D/reg_pyramid_grid/
    say "$h pyramid_grid job done and pulled: $(ls $D/reg_pyramid_grid/*.json | wc -l) rows" ) &
done
( while tmux has-session -t job 2>/dev/null; do sleep 600; done; say "reg1 w28 shard done" ) &
wait
say "pyr 3k/w28 handoffs done: $(ls $D/reg_pyramid_grid/*.json | wc -l)/350 rows"
