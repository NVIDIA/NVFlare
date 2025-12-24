#!/usr/bin/env bash
# TensorFlow Environment Configuration
# This script sets up environment variables for TensorFlow training
# 
# Usage:
#   source ./set_tf_env_vars.sh
#   or
#   . ./set_tf_env_vars.sh

# Set PYTHONPATH to include custom files
export PYTHONPATH=${PYTHONPATH}:${PWD}/src

# GPU Memory Management
# Prevent TensorFlow from allocating full GPU memory at once
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_asyncp

# Suppress TensorFlow info and warning messages
# 0=all logs, 1=filter INFO, 2=filter INFO+WARNING, 3=filter INFO+WARNING+ERROR
export TF_CPP_MIN_LOG_LEVEL=2

# Suppress Python warnings (NumPy, Keras, etc.)
export PYTHONWARNINGS="ignore"

echo "✓ TensorFlow environment variables configured:"
echo "  - PYTHONPATH=${PYTHONPATH}"
echo "  - TF_FORCE_GPU_ALLOW_GROWTH=${TF_FORCE_GPU_ALLOW_GROWTH}"
echo "  - TF_GPU_ALLOCATOR=${TF_GPU_ALLOCATOR}"
echo "  - TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL}"
echo "  - PYTHONWARNINGS=${PYTHONWARNINGS}"
