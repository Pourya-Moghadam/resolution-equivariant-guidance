#!/usr/bin/env bash
# Detached handoff (tmux "handoff40" on reg1), 2026-11-19: reg5's reg_5k6k_ext shard (all 50 6K REG rows) lost 46 rows to
# CUDA OOM from Wed 16:31 while another user was on the shared card. Delete those error records (reg5 first, then the hub
# copy, so handoff37 cannot re-pull them), rerun jobs/reg_5k6k_ext_redo.jsonl (the 46 rows) into the same output in four
# shards: reg1, reg3, reg5 now, reg4 after its own shard ends. When all have ended, any row that failed again is deleted
# and rerun once on reg1. handoff37 (count 200) then pulls and scores as before.
D=/ssd2/moghadamp/reg_data/outputs; O=$D/reg_5k6k_ext; LOG=/data1/moghadamp/reg_data/outputs/handoff_log.txt
cd /data1/moghadamp/reg-research
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
state() { if [ "$1" = reg1 ]; then tmux has-session -t "=$2" 2>/dev/null && echo BUSY || echo IDLE; else ssh -o BatchMode=yes -o ConnectTimeout=20 "$1" "bash -lc \"tmux has-session -t =$2 2>/dev/null && echo BUSY || echo IDLE\"" 2>/dev/null | tail -n 1; fi; }
ended() { [ "$(state "$1" "$2")" = IDLE ] && sleep 60 && [ "$(state "$1" "$2")" = IDLE ]; }
launch() { local h=$1; shift; if [ "$h" = reg1 ]; then bash scripts/launch_grid.sh "$@" 2>&1; else ssh -o BatchMode=yes -o ConnectTimeout=20 "$h" "bash -lc '~/reg/scripts/launch_grid.sh $*'" 2>&1; fi; }
start() { local h=$1; shift; local out; until out=$(launch "$h" "$@") && echo "$out" | grep -q started; do say "$h launch failed ($out), retry in 5 min"; sleep 300; done; say "$out"; }
errs() { python3 -c "import json,glob,os; print(' '.join(os.path.basename(f) for f in glob.glob('$1/*.json') if json.load(open(f)).get('error')))"; }
# keep the first-pass logs (launch_grid overwrites log_<host>.txt)
for h in reg5 reg3; do ssh -o BatchMode=yes $h "cd $O && for f in log_*.txt; do mv \$f \${f%.txt}_pass1.txt; done"; done
( cd $O && for f in log_*.txt; do [[ $f == *_pass1.txt ]] || mv $f ${f%.txt}_pass1.txt; done )
ssh -o BatchMode=yes reg5 "cd $O && python3 -c \"import json,glob,os; [os.remove(f) for f in glob.glob('*.json') if json.load(open(f)).get('error')]\""
E=$(errs $O); ( cd $O && rm -f $E ); say "handoff40: deleted $(echo $E | wc -w) reg_5k6k_ext error records (OOM on reg5), relaunching them"
start reg1 jobs/reg_5k6k_ext_redo.jsonl reg_5k6k_ext 0 4
start reg3 jobs/reg_5k6k_ext_redo.jsonl reg_5k6k_ext 1 4
start reg5 jobs/reg_5k6k_ext_redo.jsonl reg_5k6k_ext 2 4
until ended reg4 job; do sleep 600; done
ssh -o BatchMode=yes reg4 "cd $O && mv log_bri-p7-4.txt log_bri-p7-4_pass1.txt"
start reg4 jobs/reg_5k6k_ext_redo.jsonl reg_5k6k_ext 3 4
sleep 900
for h in reg1 reg3 reg5 reg4; do until ended $h job; do sleep 600; done; done
for h in reg3 reg5 reg4; do rsync -a "$h:$O/*.json" "$h:$O/*.png" $O/ 2>/dev/null; done
E=$(errs $O)
if [ -n "$E" ]; then
  for h in reg3 reg5 reg4; do ssh -o BatchMode=yes $h "cd $O && rm -f $E"; done; ( cd $O && rm -f $E )
  say "handoff40: $(echo $E | wc -w) rows failed again, rerunning on reg1"; start reg1 jobs/reg_5k6k_ext_redo.jsonl reg_5k6k_ext 0 1
else say "handoff40: all 46 redo rows ok"; fi
