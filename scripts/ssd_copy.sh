#!/usr/bin/env bash
# Detached (tmux "ssdcopy" on reg1): initial copy of reg_data and the four venvs from the 12 TB HDD to the new NVMe
# (/ssd2/moghadamp, mounted by IT 2026-11-16). Read-only on /data1, low I/O priority, safe beside running jobs.
# The switch (REG_DATA -> /ssd2, venv symlinks) happens after geneval_4k is scored: scripts/ssd_switch.sh.
LOG=/data1/moghadamp/reg_data/outputs/handoff_log.txt
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
T=/ssd2/moghadamp; mkdir -p $T/venvs
say "ssdcopy: initial copy to $T started"
r() { until ionice -c3 nice -n 19 rsync -a --delete "$@"; do say "ssdcopy: rsync $* failed, retry in 5 min"; sleep 300; done; }
r /data1/moghadamp/reg_data/ $T/reg_data/
for v in reg_pref_venv reg_geneval_venv reg_upscale_venv; do r /data1/moghadamp/$v/ $T/venvs/$v/; done
r /data1/moghadamp/reg-research/.venv/ $T/venvs/main_venv/
say "ssdcopy: initial copy done ($(du -sh $T | cut -f1) on $T)"
