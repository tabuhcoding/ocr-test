#!/usr/bin/env bash
# Start the ndaseal-util OCR sidecar (RapidOCR/ONNXRuntime + VietOCR).
#
# Usage:  ./run.sh            # listens on 127.0.0.1:8090
#         PORT=9000 ./run.sh  # custom port
#         OCR_HOST=0.0.0.0 ./run.sh
set -euo pipefail
cd "$(dirname "$0")"

exec .venv/bin/python -m uvicorn ndaseal_util.api:app \
  --host "${OCR_HOST:-127.0.0.1}" \
  --port "${PORT:-${OCR_PORT:-8090}}"
