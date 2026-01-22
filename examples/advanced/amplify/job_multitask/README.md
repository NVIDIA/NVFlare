# AMPLIFY Multi-Task Federated Fine-tuning

This directory contains the job configuration for federated multi-task fine-tuning of AMPLIFY, where each client trains on a different downstream task while jointly fine-tuning the shared AMPLIFY trunk.

## Overview

In this scenario:
- **6 clients** (one per task): aggregation, binding, expression, immunogenicity, polyreactivity, thermostability
- Each client trains their **own private regression head** for their specific task
- All clients **jointly fine-tune the AMPLIFY trunk** using FedAvg
- Regression heads are kept private (not shared with server) using the `ExcludeParamsFilter`

## Quick Start

### 1. Prepare Data

First, clone the FLAb repository and prepare the data:

```bash
cd ..
git clone https://github.com/Graylab/FLAb.git

# Combine and prepare data for each task
for task in "aggregation" "binding" "expression" "immunogenicity" "polyreactivity" "thermostability" 
do
    echo "Combining $task CSV data"
    python src/combine_data.py --input_dir ./FLAb/data/${task} --output_dir ./FLAb/data_fl/${task}
done
```

### 2. Run Local Training (Baseline)

Train each client independently without federated learning (1 round only):

```bash
python job.py \
    --num_rounds 1 \
    --local_epochs 600 \
    --pretrained_model "chandar-lab/AMPLIFY_120M" \
    --layer_sizes "128,64,32" \
    --exp_name "local_singletask" \
    --sim_gpus "0,1,2,0,1,2"
```

### 3. Run Federated Learning

Train using FedAvg with 600 rounds:

```bash
python job.py \
    --num_rounds 600 \
    --local_epochs 1 \
    --pretrained_model "chandar-lab/AMPLIFY_120M" \
    --layer_sizes "128,64,32" \
    --exp_name "fedavg_multitask" \
    --sim_gpus "0,1,2,0,1,2"
```

## Command Line Arguments

- `--num_rounds`: Number of federated learning rounds (default: 600)
- `--local_epochs`: Number of local training epochs per round (default: 1)
- `--pretrained_model`: AMPLIFY model to use (default: "chandar-lab/AMPLIFY_120M")
- `--layer_sizes`: Comma-separated MLP layer sizes (default: "128,64,32")
- `--data_root`: Root directory for data (default: "../FLAb/data_fl")
- `--exp_name`: Experiment name (default: "fedavg_multitask")
- `--sim_gpus`: GPU indices for simulation (default: "0")
- `--batch_size`: Training batch size (default: 64)
- `--trunk_lr`: Learning rate for AMPLIFY trunk (default: 1e-4)
- `--regressor_lr`: Learning rate for regression layers (default: 1e-2)
- `--name`: Custom job name (optional)

## Results

Results are saved to `/tmp/nvflare/AMPLIFY/multitask/<job_name>/`. You can visualize metrics using TensorBoard:

```bash
tensorboard --logdir /tmp/nvflare/AMPLIFY/multitask
```

## Key Components

- **job.py**: Main job configuration using `FedAvgRecipe`
- **../client.py**: Shared client training script (used by both scenarios)
- **../src/model.py**: AmplifyRegressor model definition
- **../src/filters.py**: ExcludeParamsFilter to keep regressors private
