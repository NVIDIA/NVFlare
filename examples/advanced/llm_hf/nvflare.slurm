#!/bin/bash
#SBATCH -A [ACCOUNT_NAME]
#SBATCH --partition=[PARTITION_NAME1,PARTITION_NAME2,...]
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=0:30:00
#SBATCH --job-name=fl_job
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Create logs directory if it doesn't exist
mkdir -p slurm_logs

# Print job information
echo "Job started at: $(date)"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"

# Get the master node address (first node in the allocation)
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
echo "Master node: $MASTER_ADDR:$MASTER_PORT"

# Set up distributed training environment variables
export WORLD_SIZE=$SLURM_NTASKS
export NNODES=$SLURM_JOB_NUM_NODES

# Activate venv
echo "Activating venv"
source ${VENV_DIR}/bin/activate
echo "nvflare path: $(which nvflare)"
echo $(nvflare --version)

# start server on first node (if using production mode)
echo "Starting NVFlare server..."
${NVFLARE_PROJECT}/server/startup/start.sh

# wait for server to be ready
sleep 10

# start client on first node (if using production mode)
echo "Starting NVFlare client..."
${NVFLARE_PROJECT}/dolly/startup/start.sh

# wait for client to start up (NVFlare will handle connection retry)
sleep 20
echo "Server and client should be ready now"

# Launch the FL job ONLY on the master node (first node)
# The FL client will then use torchrun to coordinate training across all allocated nodes
# Do NOT use srun here - we only want ONE FL client, not one per node
echo "Launching FL job on master node: $MASTER_ADDR"

# Login to WandB if API key is set
if [ -n "$WANDB_API_KEY" ]; then
    echo "Logging in to WandB..."
    wandb login $WANDB_API_KEY
else
    echo "WANDB_API_KEY not set, skipping WandB login (training will proceed without WandB tracking)"
fi

sleep 10
echo "Job submission starting at: $(date)"

python3 -u job.py \
    --client_ids dolly \
    --data_path ${PWD}/dataset \
    --workspace_dir ${PWD}/workspace/dolly_fl_multi_node \
    --job_dir ${PWD}/workspace/jobs/dolly_fl_multi_node \
    --ports 7777 \
    --gpu "[0,1,2,3,4,5,6,7]" \
    --multi_node \
    --startup_kit_location ${NVFLARE_PROJECT}/admin@nvidia.com \
    --wandb_project nvflare-llm-multinode \
    --wandb_run_name "2node_16gpu_dolly_${SLURM_JOB_ID}"

echo "Job completed at: $(date)"
echo "Check FL client results and logs at: ${NVFLARE_PROJECT}"
