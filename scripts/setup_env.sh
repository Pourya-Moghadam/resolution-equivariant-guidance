#!/usr/bin/env bash
# Phase 0, step 2: build the project venv (.venv) from requirements.lock.
# Run from the repo root on each server:  bash scripts/setup_env.sh
# Uses plain venv + pip (conda/uv are not installed on the servers).

set -eu

cd "$(dirname "$0")/.."

if [ ! -x .venv/bin/python ]; then
  # Ubuntu's python3 may lack ensurepip (python3.12-venv not installed): create without pip, then bootstrap it.
  python3 -m venv .venv 2>/dev/null || { rm -rf .venv; python3 -m venv --without-pip .venv; }
fi
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  curl -sSL https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
fi

.venv/bin/python -m pip install -q --upgrade pip wheel setuptools
.venv/bin/python -m pip install -q --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements.lock
.venv/bin/python -m pip install -q -e . --no-deps

.venv/bin/python -c "import torch,diffusers;print(torch.__version__,diffusers.__version__,torch.cuda.get_device_name())"
