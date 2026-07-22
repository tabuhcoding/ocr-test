#!/usr/bin/env bash
# Start the ndaseal-util OCR sidecar.
#
# Single-threaded CPU inference (OMP_NUM_THREADS=1, cpu_threads=1 in main.py)
# avoids the sporadic paddle 3.0 macOS crash
#   RuntimeError: (NotFound) No allocator found for the place, Place(undefined:0)
#
# Usage:  ./run.sh            # listens on 127.0.0.1:8090
#         PORT=9000 ./run.sh  # custom port
set -euo pipefail
cd "$(dirname "$0")"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export FLAGS_allocator_strategy="${FLAGS_allocator_strategy:-naive_best_fit}"

exec .venv/bin/uvicorn main:app --host 127.0.0.1 --port "${PORT:-8090}"
