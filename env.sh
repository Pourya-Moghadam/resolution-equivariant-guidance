# Source before working:  source env.sh
# Data root holds weights, datasets, text-embedding caches and outputs (never in git).
# SERVER_SETUP.md uses /data/reg; on bri-p7-3 the large disk is /data1, so default there.
export REG_DATA=${REG_DATA:-/data1/moghadamp/reg_data}
export HF_HOME=$REG_DATA/weights/hf
export HF_HUB_ENABLE_HF_TRANSFER=1
source "$(dirname "${BASH_SOURCE[0]}")/.venv/bin/activate"
