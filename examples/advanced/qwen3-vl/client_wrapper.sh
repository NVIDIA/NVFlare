#!/bin/bash
# Wrapper script for running the FL training script (e.g. client_sft_runner.py).
# Same pattern as examples/advanced/llm_hf: NVFlare runs this script with
# SCRIPT_PATH and SCRIPT_ARGS; we launch the script (single-process or future torchrun).
# For multi-node SLURM, extend this to use srun + torchrun like llm_hf/client_wrapper.sh.

SCRIPT_PATH="$1"
shift

echo "========================================="
echo "Qwen3-VL FL Training Wrapper"
echo "========================================="
echo "Script: $SCRIPT_PATH"
echo "Args: $*"
echo "========================================="

exec python "$SCRIPT_PATH" "$@"
