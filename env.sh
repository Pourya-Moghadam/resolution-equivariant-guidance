# Source before working on any server:  source env.sh
# scripts/fanout.sh, push_code.sh and sync_results.sh source it too, so this file is
# the single description of the cluster; nothing cluster-specific lives in ~/.profile.
#
# Servers: reg1 = bri-p7-1 (hub, where everything is run), reg2..reg5 = bri-p7-2..5 (workers).

export REG_NUM_SERVERS=${REG_NUM_SERVERS:-5}
export REG_HOSTS=$(seq -s ' ' -f 'reg%g' 1 "$REG_NUM_SERVERS")
export REG_WORKERS=$(seq -s ' ' -f 'reg%g' 2 "$REG_NUM_SERVERS")

# Job rows are split by  idx % REG_NUM_SERVERS == REG_SERVER_ID.  bri-p7-N -> N-1.
# Preset REG_SERVER_ID only on a machine whose hostname does not follow that pattern.
_h=$(hostname -s)
case "$_h" in
  bri-p7-[1-9]) : "${REG_SERVER_ID:=$(( ${_h##*-} - 1 ))}" ;;
esac
unset _h
[ -n "${REG_SERVER_ID:-}" ] && export REG_SERVER_ID

# Data root: weights, datasets, text-embedding caches, outputs (never in git).
# Every server's large disk is /data1; the same path is used everywhere.
export REG_DATA=${REG_DATA:-/data1/moghadamp/reg_data}
export HF_HOME=$REG_DATA/weights/hf
export HF_XET_HIGH_PERFORMANCE=1    # fast downloads; replaces the deprecated HF_HUB_ENABLE_HF_TRANSFER in huggingface_hub 2.x

# Project venv (built by scripts/setup_env.sh); silently skipped on a server where it isn't built yet.
_venv="$(dirname "${BASH_SOURCE[0]}")/.venv/bin/activate"
[ -f "$_venv" ] && . "$_venv"
unset _venv
