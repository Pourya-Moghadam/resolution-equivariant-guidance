#!/usr/bin/env bash
# Detached handoff (tmux "handoff23" on reg1): DrawBench at 4K on Lumina (600 rows, output drawbench_4k). Split since
# Mon 14:40: reg1 = shard 0/3 of jobs/drawbench_4k.jsonl (plain w4), reg5 = shard 1/3 (plain w16), and the REG rows
# (shard 2/3) as two job files: part a (160 rows) on reg4, launched here once its plain-w4 Qwen run (session "job2")
# has ended, and part b (40 rows) on reg3 after its Qwen cell (handoff24). Session names are matched exactly
# ("=name"): a bare "-t job" also matches "job2", which stalled handoff21 for a day. reg2 is not used (lent out).
D=/data1/moghadamp/reg_data/outputs
LOG=$D/handoff_log.txt
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
state() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$1" "bash -lc \"tmux has-session -t =$2 2>/dev/null && echo BUSY || echo IDLE\"" 2>/dev/null | tail -n 1; }
ended() { [ "$(state "$1" "$2")" = IDLE ] && sleep 60 && [ "$(state "$1" "$2")" = IDLE ]; }
until ended reg4 job2 && ended reg4 job; do sleep 600; done
until out=$(ssh -o BatchMode=yes -o ConnectTimeout=20 reg4 "bash -lc '~/reg/scripts/launch_grid.sh jobs/drawbench_4k_reg_a.jsonl drawbench_4k 0 1'" 2>&1) && echo "$out" | grep -q started; do say "reg4 launch failed ($out), retry in 5 min"; sleep 300; done
say "$out"
