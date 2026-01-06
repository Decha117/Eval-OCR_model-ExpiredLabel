#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip

# RapidOCR
python -m pip install rapidocr-onnxruntime

# PP-OCRv3 Mobile (PaddleOCR + PaddlePaddle CPU)
python -m pip install paddlepaddle paddleocr>=2.7.0.3

# Doctr (fast_base, linknet_resnet34)
python -m pip install "python-doctr[torch]"
