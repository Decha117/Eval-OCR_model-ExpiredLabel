#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip

# RapidOCR
python -m pip install rapidocr-onnxruntime

# Doctr (fast_base, linknet_resnet34)
python -m pip install "python-doctr[torch]"
