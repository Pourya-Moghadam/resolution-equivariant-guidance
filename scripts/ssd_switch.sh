#!/usr/bin/env bash
# One-off switch of reg1 to the NVMe scratch (run by hand on reg1 after handoff34 logs "ready to switch"):
#  1. refuses while a generation job runs on reg1 or the copy is not up to date;
#  2. final delta rsync HDD -> SSD;
#  3. the three scoring venvs and the project .venv become symlinks to their SSD copies (their shebangs hard-code the
#     /data1 paths, so every script keeps working unchanged); the HDD originals are kept as *.hdd until verified;
#  4. the SSD copy's handoff log becomes a symlink to the /data1 log (one log for old and new handoff scripts);
#  5. marks the SSD copy active (env.sh then resolves REG_DATA to it) and starts the 6-hourly archive loop.
set -u
S=/ssd2/moghadamp/reg_data; H=/data1/moghadamp/reg_data; LOG=$H/outputs/handoff_log.txt; V=/ssd2/moghadamp/venvs
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
tmux has-session -t =job 2>/dev/null && { echo "a generation job is running on reg1, not switching"; exit 1; }
grep -q "handoff34: SSD delta synced, ready to switch" "$LOG" || { echo "handoff34 has not reported the delta sync"; exit 1; }
[ -f $S/.active ] && { echo "already switched"; exit 0; }
say "ssd_switch: final delta rsync"
rsync -a --delete "$H/" "$S/" || { say "ssd_switch: final rsync FAILED"; exit 1; }
cd /data1/moghadamp
for v in reg_pref_venv reg_geneval_venv reg_upscale_venv; do
  [ -L $v ] && continue
  rsync -a --delete "$v/" "$V/$v/" && mv "$v" "$v.hdd" && ln -s "$V/$v" "$v" || { say "ssd_switch: $v FAILED"; exit 1; }
  "$v/bin/python" -c "import torch, sys; print('$v ok', torch.__version__)" || { say "ssd_switch: $v import FAILED"; exit 1; }
done
cd /data1/moghadamp/reg-research
if [ ! -L .venv ]; then
  rsync -a --delete .venv/ "$V/main_venv/" && mv .venv .venv.hdd && ln -s "$V/main_venv" .venv || { say "ssd_switch: .venv FAILED"; exit 1; }
  .venv/bin/python -c "import torch, diffusers; print('main venv ok', torch.__version__, diffusers.__version__)" || { say "ssd_switch: .venv import FAILED"; exit 1; }
fi
rm -f "$S/outputs/handoff_log.txt"; ln -s "$LOG" "$S/outputs/handoff_log.txt"
touch "$S/.active"
( unset REG_DATA; . ./env.sh; echo "REG_DATA now $REG_DATA"; [ "$REG_DATA" = "$S" ] ) || { say "ssd_switch: env.sh does not resolve to the SSD"; exit 1; }
tmux has-session -t =archive 2>/dev/null || tmux new-session -d -s archive "bash scripts/archive_to_hdd.sh loop"
say "ssd_switch: reg1 now runs from $S ($(df -h $S | awk 'NR==2{print $4}') free); venvs symlinked; archive loop started"
