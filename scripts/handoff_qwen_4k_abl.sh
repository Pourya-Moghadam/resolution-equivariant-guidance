#!/usr/bin/env bash
# Detached handoff (tmux "handoff19" on reg1): once handoff18 has pulled the whole Qwen 4K plain-w16 baseline, launch the
# 4K texture ablation (jobs/qwen_4k_abl.jsonl, 48 rows) as shards 0/4 reg4, 1/4 reg1, 2/4 reg3, 3/4 reg2.
D=/data1/moghadamp/reg_data/outputs
LOG=$D/handoff_log.txt
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
until grep -q "Qwen 4K w16 baseline complete" "$LOG"; do sleep 300; done
cd /data1/moghadamp/reg-research
say "$(bash scripts/launch_grid.sh jobs/qwen_4k_abl.jsonl qwen_4k_abl 1 4 2>&1)"
for p in "reg4 0" "reg3 2" "reg2 3"; do set -- $p
  until out=$(ssh -o BatchMode=yes -o ConnectTimeout=20 "$1" "bash -lc '~/reg/scripts/launch_grid.sh jobs/qwen_4k_abl.jsonl qwen_4k_abl $2 4'" 2>&1) && echo "$out" | grep -q started; do say "$1 launch failed ($out), retry in 5 min"; sleep 300; done
  say "$out"
done
say "Qwen 4K ablation launched on reg1-reg4"
