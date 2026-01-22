# AMPLIFY All-Tasks Federated Fine-tuning

This directory contains the job configuration for federated all-tasks fine-tuning of AMPLIFY, where each client trains on all tasks using heterogeneous data distributions.

## Overview

In this scenario:
- **Multiple clients** (default: 6) each train on **all tasks**
- Data is split heterogeneously across clients using Dirichlet distribution
- Two modes available:
  1. **Shared regressors**: All regression heads are jointly trained and shared
  2. **Private regressors**: Regression heads remain private to each client (using `--private_regressors`)

## Quick Start

### 1. Prepare Data

First, clone the FLAb repository and prepare the heterogeneous data split:

```bash
cd ..
git clone https://github.com/Graylab/FLAb.git

# Combine and split data across 6 clients with alpha=1.0
for task in "aggregation" "binding" "expression" "immunogenicity" "polyreactivity" "thermostability" 
do
    echo "Combining $task CSV data"
    python src/combine_data.py \
        --input_dir ./FLAb/data/${task} \
        --output_dir ./FLAb/data_fl/${task} \
        --num_clients=6 \
        --alpha=1.0
done
```

### 2. Run Local Training (Baseline)

Train each client independently without federated learning (1 round only):

```bash
python job.py \
    --num_clients 6 \
    --num_rounds 1 \
    --local_epochs 600 \
    --pretrained_model "chandar-lab/AMPLIFY_120M" \
    --layer_sizes "128,64,32" \
    --exp_name "local_alltasks" \
    --sim_gpus "0,1,2,0,1,2"
```

### 3. Run Federated Learning with Shared Regressors

All clients jointly train both the AMPLIFY trunk and all regression heads:

```bash
python job.py \
    --num_clients 6 \
    --num_rounds 300 \
    --local_epochs 2 \
    --pretrained_model "chandar-lab/AMPLIFY_120M" \
    --layer_sizes "128,64,32" \
    --exp_name "fedavg_alltasks" \
    --sim_gpus "0,1,2,0,1,2"
```

### 4. Run Federated Learning with Private Regressors

Keep regression heads private while jointly training the AMPLIFY trunk:

```bash
python job.py \
    --num_clients 6 \
    --num_rounds 300 \
    --local_epochs 2 \
    --pretrained_model "chandar-lab/AMPLIFY_120M" \
    --layer_sizes "128,64,32" \
    --exp_name "fedavg_alltasks_private_regressors" \
    --private_regressors \
    --sim_gpus "0,1,2,0,1,2"
```

## Command Line Arguments

- `--num_clients`: Number of federated learning clients (default: 6)
- `--num_rounds`: Number of federated learning rounds (default: 300)
- `--local_epochs`: Number of local training epochs per round (default: 2)
- `--pretrained_model`: AMPLIFY model to use (default: "chandar-lab/AMPLIFY_120M")
- `--layer_sizes`: Comma-separated MLP layer sizes (default: "128,64,32")
- `--data_root`: Root directory for data (default: "../FLAb/data_fl")
- `--exp_name`: Experiment name (default: "fedavg_alltasks")
- `--private_regressors`: Keep regressors private (flag, default: False)
- `--sim_gpus`: GPU indices for simulation (default: "0")
- `--batch_size`: Training batch size (default: 64)
- `--trunk_lr`: Learning rate for AMPLIFY trunk (default: 1e-4)
- `--regressor_lr`: Learning rate for regression layers (default: 1e-2)
- `--name`: Custom job name (optional)

## Results

Results are saved to `/tmp/nvflare/AMPLIFY/alltasks/<job_name>/`. You can visualize metrics using TensorBoard:

```bash
tensorboard --logdir /tmp/nvflare/AMPLIFY/alltasks
```

## Key Components

- **job.py**: Main job configuration using `FedAvgRecipe`
- **../client.py**: Shared client training script (used by both scenarios)
- **../src/model.py**: AmplifyRegressor model definition
- **../src/filters.py**: ExcludeParamsFilter to keep regressors private (when `--private_regressors` is used)
