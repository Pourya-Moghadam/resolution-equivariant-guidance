#!/usr/bin/env bash
# Phase 0, step 1: hardware/software audit for one REG server.
# Usage (one server):   bash scripts/server_audit.sh | tee audit_$(hostname).txt
# Usage (all, on reg1): bash scripts/fanout.sh 'bash ~/reg/scripts/server_audit.sh' > audit_all.txt
# Paste outputs into 06-compute-and-datasets.md.
# Read-only: prints information, changes nothing.

set -u

section() { printf '\n=== %s ===\n' "$1"; }

section "HOST"
hostname
date -Is
uname -r
(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME") || echo "os-release: n/a"

section "GPU (expect: RTX 5000 Ada Generation, 32 GB)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap,pcie.link.gen.max,pcie.link.width.max \
             --format=csv
  echo "-- current utilisation / other users:"
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
else
  echo "nvidia-smi NOT FOUND: driver missing or not on PATH"
fi

section "CUDA toolkit"
if command -v nvcc >/dev/null 2>&1; then nvcc --version | tail -n 2; else echo "nvcc not on PATH (OK if using conda/pip CUDA wheels)"; fi
ls -d /usr/local/cuda* 2>/dev/null || echo "no /usr/local/cuda*"

section "CPU / RAM"
lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Socket' || true
free -h | head -n 2

section "DISK (need ~1-2 TB free for outputs + weights)"
df -hT -x tmpfs -x devtmpfs -x squashfs 2>/dev/null | sort -k7 -h || df -h

section "SHARED STORAGE (NFS/CIFS mounts, if any)"
mount | grep -E 'nfs|cifs|smb|lustre|beegfs' || echo "no network filesystems mounted"

section "PYTHON / ENV MANAGERS"
for b in python3 python conda mamba uv pip; do
  if command -v "$b" >/dev/null 2>&1; then printf '%-7s %s  ' "$b" "$(command -v "$b")"; "$b" --version 2>&1 | head -n 1; fi
done

section "TORCH (if already installed in default python)"
python3 - <<'PY' 2>/dev/null || echo "torch not importable in default python (expected before Phase 0 step 2)"
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("device", p.name, "VRAM_GB", round(p.total_memory/2**30, 1), "sm", f"{p.major}{p.minor}")
PY

section "TOOLS"
for b in git tmux rsync ssh htop nvtop; do
  if command -v "$b" >/dev/null 2>&1; then echo "ok   $b"; else echo "MISSING $b"; fi
done

section "NETWORK (for rsync between servers / HF downloads)"
hostname -I 2>/dev/null || ip -4 addr show scope global | grep inet || true
(curl -sS -m 5 -o /dev/null -w 'huggingface.co HTTP %{http_code}\n' https://huggingface.co) || echo "huggingface.co: unreachable or curl missing"

section "DONE"
echo "Paste this output into 06-compute-and-datasets.md under 'Hardware'."
