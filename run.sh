#!/usr/bin/env bash
# Start the ndaseal-util OCR sidecar (RapidOCR/ONNXRuntime + VietOCR).
#
# Usage:  ./run.sh            # listens on 127.0.0.1:8090
#         PORT=9000 ./run.sh  # custom port
set -euo pipefail
cd "$(dirname "$0")"

exec .venv/bin/uvicorn main:app --host 127.0.0.1 --port "${PORT:-8090}"
