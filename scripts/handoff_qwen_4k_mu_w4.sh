#!/usr/bin/env bash
# Detached handoff (tmux "handoff22" on reg1): plain CFG 4 at the capped shift (jobs/qwen_4k_mu_w4.jsonl, 50 rows, output
# qwen_4k_mu_w4) runs as shard 0/2 on reg2 (launched Sat 22:00, tmux session "job2": reg2 came back when the colleague's
# queued run never started) and shard 1/2 on reg4 once reg4's qwen_4k_mu shard (session "job") has ended; "job2" keeps
# handoff21's idle check of session "job" true. Pulls both shards when their job2 sessions end, scores on reg1's GPU.
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
until out=$(ssh -o BatchMode=yes -o ConnectTimeout=20 reg4 "bash -lc 'REG_JOB_SESSION=job2 ~/reg/scripts/launch_grid.sh jobs/qwen_4k_mu_w4.jsonl qwen_4k_mu_w4 1 2'" 2>&1) && echo "$out" | grep -q started; do say "reg4 launch failed ($out), retry in 5 min"; sleep 300; done
say "$out"
r2=0; r4=0
while [ $r2 = 0 ] || [ $r4 = 0 ]; do
  sleep 600
  if [ $r2 = 0 ] && ended reg2 job2; then pull reg2; r2=1; fi
  if [ $r4 = 0 ] && ended reg4 job2; then pull reg4; r4=1; fi
done
cd /data1/moghadamp/reg-research
export REG_DATA=/data1/moghadamp/reg_data
S=$O/score_logs; mkdir -p $S
/data1/moghadamp/reg_pref_venv/bin/python scripts/run_pref.py --out $O --group-by size method cfg --csv results/qwen_4k_mu_w4_pref.csv > $S/pref.log 2>&1
( . ./env.sh
  python scripts/run_eval.py --out $O --refs $REG_DATA/datasets/aesthetic_4k/eval/size_2048/images --group-by size method cfg --skip ir --csv results/qwen_4k_mu_w4_eval.csv > $S/eval.log 2>&1
  python scripts/speckle.py --out $O --group-by size method cfg --csv results/qwen_4k_mu_w4_speckle.csv > $S/speckle.log 2>&1 )
say "Qwen 4K plain-w4 capped-shift run complete and scored: $(ls $O/*.png | wc -l) images on reg1"
