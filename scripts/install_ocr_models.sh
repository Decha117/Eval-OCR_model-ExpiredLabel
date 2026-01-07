#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip

# Doctr (fast_base, linknet_resnet34)
# ติดตั้ง torch ที่รองรับ CUDA ให้เรียบร้อยก่อนรันระบบ
python -m pip install "python-doctr[torch]"
