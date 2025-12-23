# FedAvg Example

This directory implements [Federated Averaging (FedAvg)](https://arxiv.org/abs/1602.05629) for CIFAR-10 classification using NVFlare's FL simulator.

## Overview

FedAvg is the foundational federated learning algorithm that trains models across multiple clients by:
1. Broadcasting the global model to selected clients
2. Each client performs local training on their private data
3. Server aggregates client model updates by weighted averaging
4. Process repeats for multiple rounds

**Key Features:**
- Simple and efficient federated learning baseline
- Weighted aggregation based on client dataset sizes
- Supports various data heterogeneity levels (controlled by `alpha` parameter)
- Uses NVFlare's `FedAvgRecipe` for easy configuration

## Usage

### Basic Usage

Run FedAvg with 8 clients, 50 rounds, and uniform data distribution (α=1.0):

```bash
python job.py --n_clients 8 --num_rounds 50 --alpha 1.0
```

### Command-line Arguments

#### Federated Learning Parameters
- `--n_clients` - Number of FL clients (default: `8`)
- `--num_rounds` - Number of FL rounds (default: `50`)
- `--alpha` - Data heterogeneity parameter (default: `0.5`)
  - Higher values (e.g., 1.0) = more uniform/IID data distribution
  - Lower values (e.g., 0.1) = more heterogeneous/non-IID distribution

#### Training Parameters
- `--aggregation_epochs` - Local epochs per round (default: `4`)
- `--lr` - Learning rate (default: `0.01`)
- `--batch_size` - Training batch size (default: `64`)
- `--num_workers` - Data loading workers (default: `2`)

#### Learning Rate Scheduler
- `--no_lr_scheduler` - Disable cosine annealing LR scheduler (enabled by default)
- `--cosine_lr_eta_min_factor` - Minimum LR as factor of initial LR (default: `0.01`)

#### Other Options
- `--train_idx_root` - Root directory for data splits (default: `/tmp/cifar10_splits`)
- `--evaluate_local` - Evaluate local model on validation set after each epoch
- `--name` - Custom job name (default: auto-generated)

## Examples

### Uniform Data Distribution (α=1.0)

```bash
python job.py --n_clients 8 --num_rounds 50 --alpha 1.0
```

**Expected Result:** ~89.4% validation accuracy

### Moderate Heterogeneity (α=0.5)

```bash
python job.py --n_clients 8 --num_rounds 50 --alpha 0.5
```

**Expected Result:** ~88.5% validation accuracy

### High Heterogeneity (α=0.1)

```bash
python job.py --n_clients 8 --num_rounds 50 --alpha 0.1
```

**Expected Result:** ~80.7% validation accuracy

### Custom Learning Rate

```bash
python job.py --n_clients 8 --num_rounds 50 --alpha 0.1 --lr 0.05
```

### More Clients, Shorter Rounds

```bash
python job.py --n_clients 16 --num_rounds 100 --alpha 0.5 --aggregation_epochs 2
```

## Implementation Details

### Algorithm

FedAvg performs weighted averaging of client model updates:

```
# Server side (each round):
1. Send global model w_t to selected clients
2. Receive model updates Δw_i from each client i
3. Aggregate: w_{t+1} = w_t + Σ(n_i / n_total) * Δw_i
   where n_i is the number of training samples at client i
```

### Client Training

Each client (`client.py`):
- Loads the global model from the server
- Trains locally for `aggregation_epochs` epochs
- Computes model difference: Δw = w_local - w_global
- Sends Δw back to the server with metadata (number of training steps)

### Server Aggregation

The server uses NVFlare's default Aggregator:
- Weights each client's contribution by their training steps
- Applies weighted average to all model parameters
- Broadcasts the updated global model

### Data Heterogeneity

The `alpha` parameter controls data heterogeneity via Dirichlet sampling:
- **α = 1.0**: Near-uniform distribution (IID)
- **α = 0.5**: Moderate non-IID distribution
- **α = 0.3**: High non-IID distribution
- **α = 0.1**: Very high non-IID distribution

Lower alpha values make the federated learning task more challenging.

## Performance Comparison

| Alpha | Val Accuracy | Description |
|-------|-------------|-------------|
| 1.0 | 89.4% | Uniform distribution, best performance |
| 0.5 | 88.5% | Moderate heterogeneity |
| 0.3 | 85.1% | High heterogeneity |
| 0.1 | 80.7% | Very high heterogeneity, challenging |

*Results from 8 clients, 50 rounds, 4 local epochs per round.*

## Viewing Results

After running a job, view the training curves with TensorBoard:

```bash
tensorboard --logdir=/tmp/nvflare/simulation
```

Then open `http://localhost:6006` in your browser.

## Comparison with Other FL Algorithms

For highly heterogeneous data (α=0.1), consider more advanced algorithms:
- **[FedProx](../cifar10_fedprox/README.md)**: Adds proximal regularization
- **[FedOpt](../cifar10_fedopt/README.md)**: Server-side adaptive optimization
- **[SCAFFOLD](../cifar10_scaffold/README.md)**: Control variates for variance reduction

See the [main README](../README.md) for detailed comparisons.

## References

- [FedAvg Paper](https://arxiv.org/abs/1602.05629) - McMahan et al., 2017
- [NVFlare Documentation](https://nvflare.readthedocs.io/)
- [Dirichlet Distribution for Non-IID Data](https://arxiv.org/abs/2002.06440) - Wang et al., 2020

