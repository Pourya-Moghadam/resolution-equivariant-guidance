#!/usr/bin/env bash
# Copy the NVMe working set (/ssd2/moghadamp/reg_data, single disk, no redundancy) back to the 12 TB HDD archive at
# /data1/moghadamp/reg_data. Additive (no --delete), skips tmp/ and the handoff log (which lives on /data1 and is
# symlinked from the SSD copy). "loop" runs it every 6 h (tmux "archive" on reg1, started by ssd_switch.sh).
#   bash scripts/archive_to_hdd.sh [loop]
S=/ssd2/moghadamp/reg_data; H=/data1/moghadamp/reg_data; LOG=$H/outputs/handoff_log.txt
say() { echo "$(date '+%a %H:%M') $*" | tee -a "$LOG"; }
once() { [ -f $S/.active ] || { say "archive: $S is not the active data root, nothing done"; return 1; }
  ionice -c3 nice -n 19 rsync -a --exclude tmp/ --exclude .active --exclude outputs/handoff_log.txt "$S/" "$H/" \
    && say "archive: SSD -> HDD synced ($(du -sh $H | cut -f1) on /data1)" || say "archive: rsync FAILED (rc $?)"; }
if [ "${1:-}" = loop ]; then while :; do once; sleep 21600; done; else once; fi
