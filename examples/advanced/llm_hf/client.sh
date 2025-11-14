#!/bin/bash
# Wrapper script for multi-node distributed training
# This script is called by the NVFlare client and uses srun to launch training across nodes

# Get the training script and arguments from command line
SCRIPT_PATH="$1"
shift  # Remove first argument
SCRIPT_ARGS="$@"

# Get SLURM environment variables (should be set by the SLURM job)
NNODES=${SLURM_JOB_NUM_NODES:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
JOB_ID=${SLURM_JOB_ID:-0}

echo "========================================="
echo "Multi-Node Training Launcher"
echo "========================================="
echo "Nodes: $NNODES"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Job ID: $JOB_ID"
echo "Script: $SCRIPT_PATH"
echo "Args: $SCRIPT_ARGS"
echo "========================================="

# Determine number of GPUs per node
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    # Count comma-separated GPU IDs
    NGPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    # Default to number of GPUs detected
    NGPUS=$(nvidia-smi --list-gpus | wc -l)
fi

echo "GPUs per node: $NGPUS"

if [ "$NNODES" -eq "1" ]; then
    # Single node - just use torchrun directly
    echo "Running single-node training..."
    # Set environment variables for NCCL
    export NCCL_DEBUG=INFO
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    torchrun \
        --nproc_per_node=$NGPUS \
        --master_port=$MASTER_PORT \
        $SCRIPT_PATH $SCRIPT_ARGS
else
    # Multi-node - use srun to launch torchrun on each node
    echo "Running multi-node training across $NNODES nodes..."
    srun --nodes=$NNODES --ntasks-per-node=1 bash -c "
        # Set environment variables for NCCL to reduce warnings
        export NCCL_DEBUG=INFO
        export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
        export NCCL_IB_DISABLE=0
        export NCCL_SOCKET_IFNAME=^lo,docker0
        
        torchrun \
            --nnodes=$NNODES \
            --nproc_per_node=$NGPUS \
            --node_rank=\$SLURM_NODEID \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            --rdzv_backend=c10d \
            --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
            --rdzv_id=$JOB_ID \
            $SCRIPT_PATH $SCRIPT_ARGS
    "
fi

EXIT_CODE=$?
echo "========================================="
echo "Training completed with exit code: $EXIT_CODE"
echo "========================================="
exit $EXIT_CODE
