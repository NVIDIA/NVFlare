# Multi-Node Training Solution Summary

## Quick Reference

This quick reference describes how to run NVIDIA FLARE in an SLURM-managed cluster environment.

### Key Files
- **`nvflare.slurm`** - SLURM batch script
- **`job.py`** - Job configuration (uses wrapper script)
- **`client_wrapper.sh`** - Wrapper script for multi-node coordination
- **`client.py`** - Training script (fixed rank vs local_rank)

### Main Principles for Multi-node Training
1. **Use global rank 0 for NVFlare operations** (not local_rank)
2. **Use local_rank for GPU device selection** (0-7 per node)
3. **Only master node runs FL job launcher** (no srun in SLURM script)
4. **Wrapper script handles multi-node srun coordination**

---

## Problem Evolution and Solutions

This document captures all the issues encountered and solutions implemented for multi-node distributed training with NVFlare and PyTorch DDP on SLURM.

### Issues Encountered (in order)

1. **"flare.init timeout" error** - Multiple FL clients trying to initialize on different nodes
2. **"missing job on client" error** - FL client couldn't execute the training command
3. **Environment variable scope** - Variables set in SLURM script weren't available in FL client process
4. **"Invalid device ordinal" error** - Wrong CUDA device mapping (global rank vs local rank)
5. **"Connection refused" error** - All processes trying to connect to NVFlare client
6. **NCCL warnings** - Process group initialization warnings

## Final Solution: Wrapper Script Approach

### Architecture

```
SLURM Job (2 nodes allocated)
  │
  ├─> Master Node
  │   ├─> NVFlare Server
  │   └─> NVFlare Client (dolly)
  │       └─> Receives training task
  │           └─> Executes: bash client_wrapper.sh client.py --args...
  │               └─> Wrapper script uses srun to launch across nodes
  │                   ├─> Node 0: torchrun --node_rank=0 (spawns 8 processes)
  │                   └─> Node 1: torchrun --node_rank=1 (spawns 8 processes)
  │
  └─> Worker Node (idle until srun activates it)
```

### Key Components

#### 1. **SLURM Script** (`nvflare.slurm`)
- Allocates 2 nodes
- Sets up environment variables (`MASTER_ADDR`, `MASTER_PORT`, etc.)
- Starts NVFlare server and client on master node
- Runs `job.py` to submit the FL job
- **Does NOT use srun** to run the job script (only ONE FL client)

#### 2. **Job Configuration** (`job.py`)
- Uses `FedAvgRecipe` with `per_site_config` for multi-node setup
- When `--multi_node` flag is set:
  - Sets command in per_site_config: `"command": "bash custom/client_wrapper.sh"`
  - Adds wrapper script to job: `recipe.job.to("client_wrapper.sh", site_name)`
- Script arguments passed via `"train_args"` in per_site_config
- No need to handle environment variables in Python

#### 3. **Wrapper Script** (`client_wrapper.sh`)
- Receives training script path and arguments
- Reads SLURM environment variables
- Automatically detects single-node vs multi-node
- Uses `srun` to launch `torchrun` on each node for multi-node
- Uses `torchrun` directly for single-node
- Uses `CUDA_VISIBLE_DEVICES` to set GPUs. Assumes that they are set as comma-separated list, e.g. "0,1,2,3,4,5,6,7".

**Why this works:**
- The wrapper script is included in the FL job package via `recipe.job.to("client_wrapper.sh", site_name)`
- It's placed in the `custom/` subdirectory of the job workspace
- Command is set to `bash custom/client_wrapper.sh` in the per_site_config
- It runs in the same environment as the SLURM job (has access to `srun` and SLURM variables)
- It handles all the complexity of multi-node coordination

#### 4. **Training Script** (`client.py`)
- Proper distributed training support with rank/local_rank distinction
- Reads environment variables set by `torchrun`
- **Only global rank 0 communicates with FL server**
- Uses local_rank for GPU selection (0-7 on each node)
- Uses global rank for NVFlare operations (only rank 0)

## How It Works

### Step-by-Step Flow

1. **SLURM allocates 2 nodes** (both reserved, worker node idle)

2. **SLURM script runs on master node**:
   ```bash
   export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
   export MASTER_PORT=29500
   # Start NVFlare server and client
   python3 job.py --client_ids dolly --data_path ${PWD}/dataset \
       --gpu "[0,1,2,3,4,5,6,7]" --multi_node \
       --workspace_dir ${PWD}/workspace --job_dir ${PWD}/jobs
   ```

3. **`job.py` creates and exports FL job**:
   - Uses `FedAvgRecipe` to configure the federated learning job
   - For multi-node mode (`--multi_node` flag):
     - Sets command via `per_site_config`: `"command": "bash custom/client_wrapper.sh"`
     - Adds wrapper script: `recipe.job.to("client_wrapper.sh", site_name)`
   - Includes `client.py` training script automatically via recipe
   - Exports job configuration to specified directory

4. **NVFlare client receives training task**:
   - Extracts job files to workspace (including `custom/client_wrapper.sh`)
   - Executes command from per_site_config: `bash custom/client_wrapper.sh`
   - Wrapper script receives training script path and arguments

5. **Wrapper script detects multi-node setup**:
   ```bash
   NNODES=$SLURM_JOB_NUM_NODES  # = 2
   # Multi-node detected, use srun
   srun --nodes=2 --ntasks-per-node=1 bash -c "torchrun --nnodes=2 --node_rank=\$SLURM_NODEID ..."
   ```

6. **`srun` launches torchrun on each node**:
   - Node 0: `SLURM_NODEID=0` → `torchrun --node_rank=0` → spawns 8 processes (ranks 0-7)
   - Node 1: `SLURM_NODEID=1` → `torchrun --node_rank=1` → spawns 8 processes (ranks 8-15)

7. **Training proceeds**:
   - All 16 processes train together using PyTorch DDP
   - Only rank 0 calls `flare.receive()` and `flare.send()`
   - Model updates synchronized across all processes

## Current Solution Advantages

✅ **Recipe Pattern**: Uses `FedAvgRecipe` for maintainable configuration  
✅ **Separation of concerns**: Job creation vs execution  
✅ **Environment isolation**: Wrapper script runs in correct environment  
✅ **Flexibility**: Works for both single-node and multi-node via `--multi_node` flag  
✅ **Per-Site Configuration**: Different commands and arguments per client via `per_site_config`  
✅ **Simplicity**: No complex string escaping or variable expansion  
✅ **Debugging**: Wrapper script provides clear logging  
✅ **Portability**: Easy to modify for different cluster setups

## Job Configuration Arguments

The `job.py` script accepts several arguments relevant to multi-node training:

**Required:**
- `--client_ids`: Client/site names, space-separated (e.g., `dolly` or `hospital-1`). Used directly as site names.
- `--data_path`: Root directory containing client datasets
- `--multi_node`: Flag to enable multi-node training mode

**Optional:**
- `--gpu`: GPU assignments (e.g., `"[0,1,2,3,4,5,6,7]"` for 8 GPUs)
- `--num_rounds`: Number of FL rounds (default: 3)
- `--local_epoch`: Local training epochs per round (default: 1)
- `--lr_scheduler`: Learning rate scheduler (default: "constant")
- `--train_mode`: Training mode - `SFT` or `PEFT` (default: "SFT")
- `--message_mode`: Communication format - `numpy` or `tensor` (default: "numpy")
- `--quantize_mode`: Model quantization for communication (e.g., `float16`, `blockwise8`)
- `--wandb_project`: WandB project name for experiment tracking
- `--wandb_run_name`: WandB run name
- `--use_tracking`: Enable TensorBoard tracking

**Example with all key arguments:**
```bash
python3 job.py \
    --client_ids dolly \
    --data_path ${PWD}/dataset \
    --multi_node \
    --gpu "[0,1,2,3,4,5,6,7]" \
    --num_rounds 5 \
    --local_epoch 2 \
    --lr_scheduler cosine_with_restarts \
    --train_mode SFT \
    --message_mode tensor \
    --wandb_project my_llm_project \
    --workspace_dir ${PWD}/workspace \
    --job_dir ${PWD}/jobs
```

## Testing

### Single Node (8 GPUs)
When testing with a single node, you can either:

**Option 1: Without `--multi_node` flag** (uses torchrun directly, no wrapper):
```bash
python3 job.py --client_ids dolly --data_path ./dataset \
    --gpu "[0,1,2,3,4,5,6,7]" \
    --workspace_dir ./workspace --job_dir ./jobs
```

**Option 2: With `--multi_node` flag** (uses wrapper script):
```bash
sbatch --nodes=1 --gpus-per-node=8 nvflare.slurm
```
Wrapper script detects `NNODES=1` and uses torchrun directly (no srun).

### Multi-Node (2 nodes, 16 GPUs)
**Required: Must use `--multi_node` flag**
```bash
sbatch --nodes=2 --gpus-per-node=8 nvflare.slurm
```
Wrapper script detects `NNODES=2` and uses srun + torchrun.

**The `--multi_node` flag is critical** - it tells `job.py` to:
- Use `client_wrapper.sh` instead of direct torchrun
- Include the wrapper script in the job package
- Set the correct command in the job configuration

## Troubleshooting

### Check wrapper script execution
Look for this in FL client logs:
```
=========================================
Multi-Node Training Launcher
=========================================
Nodes: 2
Master: node001:29500
...
```

### Check torchrun on each node
Look for:
```
Node 0 starting...
Node 1 starting...
```

### Check training processes
```bash
# On each node during training
nvidia-smi  # Should show 8 processes
ps aux | grep python  # Should show torchrun + 8 training processes
```

## Success Indicators

When everything is working correctly, you should see:

### 1. FL Client Initialization
```
Rank 0: NVFlare client initialized
Rank 1 (local_rank 1): Skipping NVFlare init
...
Rank 8 (local_rank 0): Skipping NVFlare init  ← Node 1
...
Rank 15 (local_rank 7): Skipping NVFlare init
```

### 2. NCCL Communication Setup
```
[rank0] NCCL INFO Channel 00/0 : 0[0] -> 1[1] [send] via P2P/CUMEM
[rank8] NCCL INFO Channel 00/0 : 8[0] -> 9[1] [send] via NET/IB/GDRDMA
...
Connected all rings, use ring PXN 0 GDR 1
```

**What this means:**
- **P2P/CUMEM**: Within-node GPU communication (fast!)
- **NET/IB/GDRDMA**: Cross-node via InfiniBand with GPUDirect RDMA (very fast!)
- **GDR 1**: GPUDirect RDMA enabled ✅

### 3. Training Progress
```
current_round=0
Training local epoch 1/1
{'loss': 2.5, 'learning_rate': 0.0005, 'epoch': 0.1}
...
```

### 4. Model Exchange
```
Rank 0: Received model from FL server
In total X params to be sent to server.
```

## Key Principles for Multi-Node NVFlare + PyTorch DDP

1. **One FL Client Per Multi-node Cluster**: Only one NVFlare client process is needed per cluster, and it should run on the SLURM master node.
2. **Use `--multi_node` Flag**: Must be set in `job.py` to enable wrapper script and correct command configuration
3. **Recipe-Based Configuration**: Use `FedAvgRecipe` with `per_site_config` for flexible job setup
4. **Rank 0 Only for FL Operations**: Only global rank 0 talks to FL server
5. **Local Rank for GPU Selection**: Use local_rank (0-7) for `cuda:X` device mapping
6. **Global Rank for FL Communication**: Use rank (0-15) for NVFlare API calls
7. **Broadcast Coordination**: Rank 0 broadcasts FL state to all other ranks
8. **Shared Filesystem**: Only rank 0 saves checkpoints (avoid conflicts)
9. **Wrapper Script Pattern**: Separate job creation from execution environment via `client_wrapper.sh`
