#!/usr/bin/env bash
# Detached handoff (tmux "handoff2" on reg1): when reg2/reg3 finish the 3K REG rows, pull them and launch the YaRN
# generality grid (server 0/2, 1/2); when reg4 finishes its REG w2/w8 shard, pull it and launch the reduced-PAG pilot; when reg5 finishes, pull it and launch the 5K super-native sweep.
D=/data1/moghadamp/reg_data/outputs
LOG=$D/handoff_log.txt
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
busy() { ssh -o BatchMode=yes "$1" 'bash -lc "tmux has-session -t job 2>/dev/null"'; }
for i in 0 1; do
  h=$([ $i = 0 ] && echo reg2 || echo reg3)
  ( until ! busy $h; do sleep 600; done
    rsync -a --exclude 'log_*' $h:$D/reg_pyramid_grid/ $D/reg_pyramid_grid/
    say "$h 3K REG rows pulled: $(ls $D/reg_pyramid_grid/*.json | wc -l) rows in reg_pyramid_grid"
    say "$(ssh $h "bash -lc '~/reg/scripts/launch_grid.sh jobs/reg_generality_grid.jsonl reg_generality_grid $i 2'" 2>&1)" ) &
done
( until ! busy reg4; do sleep 600; done
  rsync -a --exclude 'log_*' reg4:$D/reg_pyramid_grid/ $D/reg_pyramid_grid/
  say "reg4 REG w28 shard pulled: $(ls $D/reg_pyramid_grid/*.json | wc -l) rows"
  say "$(ssh reg4 "bash -lc '~/reg/scripts/launch_grid.sh jobs/reg_pag_scale_pilot.jsonl reg_pag_pilot 0 1'" 2>&1)" ) &
( until ! busy reg5; do sleep 600; done
  rsync -a --exclude 'log_*' reg5:$D/reg_pyramid_grid/ $D/reg_pyramid_grid/
  say "reg5 REG w28 shard pulled: $(ls $D/reg_pyramid_grid/*.json | wc -l) rows"
  say "$(ssh reg5 "bash -lc '~/reg/scripts/launch_grid.sh jobs/reg_5k_pilot.jsonl reg_pag_pilot 0 1'" 2>&1)" ) &
wait
say "yarn/pag2 launches done"
