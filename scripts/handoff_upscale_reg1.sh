#!/usr/bin/env bash
# Detached handoff (tmux "handoff26" on reg1): the generate-then-upscale baseline at 4K (scripts/upscale_baseline.py,
# output upscale_4k: the 50 4K grid prompts, 1K plain CFG 4 images from kill_or_go upscaled 4x by Lanczos and by
# Real-ESRGAN x4plus). Lanczos rows were written on CPU Tue 23:10; the Real-ESRGAN rows wait for reg1's DrawBench
# shard (session "job") to end and for the upscaler venv to import, then the cell is scored on reg1's GPU with the
# same calls as every other 4K cell. Session names are matched exactly. reg2 is not used.
D=/data1/moghadamp/reg_data/outputs
O=$D/upscale_4k
LOG=$D/handoff_log.txt
P=/data1/moghadamp/reg_upscale_venv/bin/python
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
busy() { tmux has-session -t =job 2>/dev/null; }
until ! busy && sleep 60 && ! busy; do sleep 600; done
until $P -c "import realesrgan, basicsr" 2>/dev/null; do say "upscaler venv not ready, retry in 10 min"; sleep 600; done
cd /data1/moghadamp/reg-research
export REG_DATA=/data1/moghadamp/reg_data
S=$O/score_logs; mkdir -p $S
$P scripts/upscale_baseline.py --methods realesrgan > $S/realesrgan.log 2>&1
say "upscale_4k: $(grep -c ' ok ' $S/realesrgan.log) Real-ESRGAN rows written ($(tail -n 1 $S/realesrgan.log))"
/data1/moghadamp/reg_pref_venv/bin/python scripts/run_pref.py --out $O --group-by size method cfg --csv results/upscale_4k_pref.csv > $S/pref.log 2>&1
( . ./env.sh
  python scripts/run_eval.py --out $O --refs $REG_DATA/datasets/aesthetic_4k/eval/size_2048/images --group-by size method cfg --skip ir --csv results/upscale_4k_eval.csv > $S/eval.log 2>&1
  python scripts/speckle.py --out $O --group-by size method cfg --csv results/upscale_4k_speckle.csv > $S/speckle.log 2>&1 )
sed -i 's/\r$//' results/upscale_4k_*.csv
say "Upscale baseline at 4K complete and scored"
