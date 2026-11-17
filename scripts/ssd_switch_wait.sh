#!/usr/bin/env bash
# Detached waiter (tmux "ssdswitch" on each host): once the host's generation job has ended (IDLE twice, 60 s apart)
# and the switch preconditions hold, run scripts/ssd_switch.sh. Retries every 30 min while a precondition fails.
cd /data1/moghadamp/reg-research
LOG=/data1/moghadamp/reg_data/outputs/handoff_log.txt
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
idle() { ! tmux has-session -t =job 2>/dev/null; }
while :; do until idle && sleep 60 && idle; do sleep 600; done
  out=$(bash scripts/ssd_switch.sh 2>&1); rc=$?
  if [ $rc -eq 0 ]; then say "ssd_switch_wait: switch done on $(hostname -s)"; break; fi
  say "ssd_switch_wait: $(hostname -s) not switched yet ($(echo "$out" | tail -n 1)), retry in 30 min"; sleep 1800
done
