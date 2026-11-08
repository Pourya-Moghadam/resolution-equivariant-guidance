#!/usr/bin/env bash
# Detached handoff (tmux "handoff22" on reg1): plain CFG 4 at the capped shift (jobs/qwen_4k_mu_w4.jsonl, 50 rows, output
# qwen_4k_mu_w4) runs on reg4 alone (session "job2", all 50 rows) once reg4's qwen_4k_mu shard (session "job") has ended;
# "job2" keeps handoff21's idle check of session "job" true. (A reg2 shard started Sat 21:50 was stopped 10 min later: the
# card's borrower had in fact been running on it all Saturday.) Pulls reg4 when job2 ends, scores on reg1's GPU.
D=/data1/moghadamp/reg_data/outputs
O=$D/qwen_4k_mu_w4
LOG=$D/handoff_log.txt
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
state() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$1" "bash -lc \"tmux has-session -t $2 2>/dev/null && echo BUSY || echo IDLE\"" 2>/dev/null | tail -n 1; }
ended() { [ "$(state "$1" "$2")" = IDLE ] && sleep 60 && [ "$(state "$1" "$2")" = IDLE ]; }
pull() {
  until rsync -a --include='*.json' --include='*.png' --exclude='*' "$1":$O/ $O/; do say "$1 rsync failed, retry in 5 min"; sleep 300; done
  say "$1 qwen_4k_mu_w4 pulled: ok: $(grep -l '"status": "ok"' $O/*.json 2>/dev/null | wc -l), error: $(grep -l '"status": "error"' $O/*.json 2>/dev/null | wc -l)"
}
until ended reg4 job; do sleep 600; done
until out=$(ssh -o BatchMode=yes -o ConnectTimeout=20 reg4 "bash -lc 'REG_JOB_SESSION=job2 ~/reg/scripts/launch_grid.sh jobs/qwen_4k_mu_w4.jsonl qwen_4k_mu_w4 0 1'" 2>&1) && echo "$out" | grep -q started; do say "reg4 launch failed ($out), retry in 5 min"; sleep 300; done
say "$out"
until sleep 600 && ended reg4 job2; do :; done
pull reg4
cd /data1/moghadamp/reg-research
export REG_DATA=/data1/moghadamp/reg_data
S=$O/score_logs; mkdir -p $S
/data1/moghadamp/reg_pref_venv/bin/python scripts/run_pref.py --out $O --group-by size method cfg --csv results/qwen_4k_mu_w4_pref.csv > $S/pref.log 2>&1
( . ./env.sh
  python scripts/run_eval.py --out $O --refs $REG_DATA/datasets/aesthetic_4k/eval/size_2048/images --group-by size method cfg --skip ir --csv results/qwen_4k_mu_w4_eval.csv > $S/eval.log 2>&1
  python scripts/speckle.py --out $O --group-by size method cfg --csv results/qwen_4k_mu_w4_speckle.csv > $S/speckle.log 2>&1 )
say "Qwen 4K plain-w4 capped-shift run complete and scored: $(ls $O/*.png | wc -l) images on reg1"
