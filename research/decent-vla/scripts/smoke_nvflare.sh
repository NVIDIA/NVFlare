#!/usr/bin/env bash
# CPU smoke: 2-round FedAvg with the mock backend and 2 clients, through the real NVFlare simulator.
# Requires: pip install -e ".[nvflare]"
# The simulator starts the server and client processes itself, driven by `decent-vla run`.
set -euo pipefail
cd "$(dirname "$0")/.."
# One BLAS thread is enough for the mock model and avoids exhausting process limits on shared hosts.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
CONFIG="${1:-$PWD/examples/mock_fl_nvflare.yaml}"
# Fresh output dir per smoke: the runtime auto-resumes from checkpoints/ in fl.output_dir, and a
# leftover server_round_0002.pt would turn the smoke into a "nothing to do" run. Only the default
# config's output dir is cleaned; a custom config owns its own output_dir.
if [ $# -eq 0 ]; then rm -rf outputs/mock_fl_nvflare; fi
python -m decent_vla.cli.main run --config "$CONFIG"
echo "nvflare smoke OK"
