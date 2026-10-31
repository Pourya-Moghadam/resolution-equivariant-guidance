#!/usr/bin/env bash
# Detached handoffs for 2026-10-30 night (run inside tmux on reg1 so a Claude session restart cannot kill them):
#   reg2/reg5: when the w2/w8 shard ends -> pull rows -> launch the generality pilot (server 0/2, 1/2)
#   reg3:      when the PAG/5K pilot ends -> pull rows -> launch the ablation's second shard (1/2)
#   reg1:      report when the local w28 shard ends
D=/data1/moghadamp/reg_data/outputs
LOG=$D/handoff_log.txt
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
busy() { ssh -o BatchMode=yes "$1" 'bash -lc "tmux has-session -t job 2>/dev/null"'; }
for i in 0 1; do
  h=$([ $i = 0 ] && echo reg2 || echo reg5)
  ( until ! busy $h; do sleep 600; done
    rsync -a --exclude 'log_*' $h:$D/reg_anchor_grid/ $D/reg_anchor_grid/
    say "$h w28 shard done and pulled: $(ls $D/reg_anchor_grid/*.json | wc -l) rows in reg_anchor_grid"
    say "$(ssh $h "bash -lc '~/reg/scripts/launch_grid.sh jobs/reg_generality_pilot.jsonl reg_generality_pilot $i 2'" 2>&1)" ) &
done
( until ! busy reg3; do sleep 600; done
  mkdir -p $D/reg_pag_pilot
  rsync -a --exclude 'log_*' reg3:$D/reg_pag_pilot/ $D/reg_pag_pilot/
  say "PAG/5K pilot done on reg3 and pulled: $(ls $D/reg_pag_pilot/*.json | wc -l)/15 rows; status: $(grep -h -o '"status": "[a-z]*"' $D/reg_pag_pilot/*.json | sort | uniq -c | tr '\n' ' ')"
  say "$(ssh reg3 "bash -lc '~/reg/scripts/launch_grid.sh jobs/reg_pyramid_ablation.jsonl reg_pyramid_grid 1 2'" 2>&1)" ) &
( while tmux has-session -t job 2>/dev/null; do sleep 600; done
  say "reg1 w28 shard done: $(ls $D/reg_anchor_grid/*.json | wc -l) rows in reg_anchor_grid; by weight: $(grep -h -o '"cfg": [0-9.]*' $D/reg_anchor_grid/*.json | sort | uniq -c | tr '\n' ' ')" ) &
wait
say "all handoffs done"
