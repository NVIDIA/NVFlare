# FedHCA2: Federated Learning with Hierarchical Cross-Attention Aggregation

This directory contains the implementation of FedHCA2 algorithm using NVIDIA FLARE framework.

## Features

- **Conflict-Averse Encoder Aggregation**: Handles parameter conflicts between clients
- **Cross-Attention Decoder Aggregation**: Personalized model adaptation per client  
- **Learnable Hyperweights**: Dynamic client contribution weighting
- **Heterogeneous Clients**: Supports both single-task and multi-task clients
- **Standard NVFLARE Pattern**: Follows established NVFLARE research project conventions

## Quick Start

### 1. Setup Environment

```bash
# Activate NVFLARE environment
conda activate fedhca2_nvflare

# Verify NVFLARE installation
nvflare --version
```

### 2. Run Experiment

```bash
# Run FedHCA2 simulation with 6 clients (5 single-task + 1 multi-task)
nvflare simulator jobs/fedhca2_pascal_nyud -w /tmp/fedhca2_workspace -n 6 -t 6 --gpu 0
```

### 3. Monitor Results

```bash
# View tensorboard logs  
tensorboard --logdir=/tmp/fedhca2_workspace/server/simulate_job/tb_events
```

## Configuration

### Client-Specific Hyperparameters

Each client type can have different training parameters:

**Single-Task Clients (Pascal Context):**
```json
"ST_Datasets": [{
  "dataname": "pascalcontext",
  "learning_rate": 0.0001,
  "weight_decay": 0.0001,
  "local_epochs": 1,
  "batch_size": 4,
  "warmup_epochs": 5,
  "optimizer": "adamw",
  "nworkers": 4,
  "fp16": true
}]
```

**Multi-Task Clients (NYU Depth):**
```json
"MT_Datasets": [{
  "dataname": "nyud", 
  "learning_rate": 0.0001,  // Can be different from ST clients
  "weight_decay": 0.0001,
  "local_epochs": 1,
  "batch_size": 4,
  // ... other parameters
}]
```

### Global Algorithm Settings

```json
"algorithm": {
  "encoder_agg": "conflict_averse",
  "decoder_agg": "cross_attention", 
  "ca_c": 0.4,
  "enc_alpha_init": 0.1,
  "dec_beta_init": 0.1
}
```

## File Structure

```
research/fedhca2/
├── jobs/fedhca2_pascal_nyud/           # NVFLARE job directory
│   ├── app/config/
│   │   ├── config_fed_client.json     # Client executor config
│   │   ├── config_fed_server.json     # Server aggregator config  
│   │   └── config_train.json          # Training hyperparameters
│   ├── app/custom/
│   │   ├── fedhca2_learner.py         # Client-side executor
│   │   ├── fedhca2_aggregator.py      # Server-side aggregator
│   │   └── fedhca2_core/              # Algorithm implementation
│   └── meta.json                      # NVFLARE job metadata
├── data/                              # Datasets (Pascal Context, NYU Depth)
└── README.md                          # This file
```

## Key Advantages

1. **Standard NVFLARE Pattern**: Uses direct `nvflare simulator` command
2. **No Manual Config Copying**: NVFLARE handles all config distribution automatically
3. **Client-Specific Parameters**: Different hyperparameters per client type  
4. **Clean Separation**: Algorithm, model, and training configs are well organized
5. **No Parameter Conflicts**: Each parameter appears in exactly one place

## Experimental Setup

- **5 Pascal Context clients**: Each handles one task (semseg, human_parts, normals, edge, sal)
- **1 NYU Depth client**: Handles multiple tasks (semseg, normals, edge, depth)
- **Heterogeneous training**: Different datasets, tasks, and potentially different hyperparameters
- **FedHCA2 aggregation**: Conflict-averse encoder + cross-attention decoder with learnable hyperweights

## Citation

Please cite the original FedHCA2 paper if you use this implementation.