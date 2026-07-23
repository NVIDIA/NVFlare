# FedProx Example

This directory implements [FedProx](https://arxiv.org/abs/1812.06127) for CIFAR-10 classification using NVFlare's FL simulator.

> **Main branch note:** `FedProxRecipe` is introduced for NVFlare 2.9.0. Until that package is published, install
> NVFlare from this repository with `python -m pip install -e .` from the repository root, then install the
> remaining simulator requirements separately.

## Overview

FedProx extends FedAvg by adding a **proximal regularization term** to the local training objective. This prevents client models from drifting too far from the global model, which is particularly beneficial when:
- Clients have heterogeneous data distributions (non-IID)
- Clients have varying computational capabilities
- Local training requires many steps

**Key Algorithm:**
Instead of minimizing `L(w)`, each client minimizes:
```
L_proximal(w) = L(w) + (μ/2) * ||w - w_global||²
```
where μ controls the strength of the proximal term.

## Usage

### Basic Usage

Run FedProx with 8 clients, 50 rounds, and high data heterogeneity (α=0.1):

```bash
python job.py --n_clients 8 --num_rounds 50 --alpha 0.1 --fedprox_mu 0.01
```

### Command-line Arguments

#### FedProx-Specific Parameters
- `--fedprox_mu` - Proximal term coefficient μ (default: `0.01`)
  - Useful values are task-dependent and may span several orders of magnitude
  - Higher values = stronger regularization (less client drift)
  - Must be finite and positive; use the sibling FedAvg example for no proximal regularization

#### Federated Learning Parameters
- `--n_clients` - Number of FL clients (default: `8`)
- `--num_rounds` - Number of FL rounds (default: `50`)
- `--alpha` - Data heterogeneity parameter (default: `0.5`)
  - Lower values = more heterogeneous/non-IID data

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

### Standard FedProx with High Heterogeneity

```bash
python job.py --n_clients 8 --num_rounds 50 --alpha 0.1 --fedprox_mu 0.01
```

### Stronger Proximal Regularization

```bash
python job.py --n_clients 8 --num_rounds 50 --alpha 0.1 --fedprox_mu 0.1
```

### Compare with FedAvg

```bash
# Run from the cifar10_fedprox directory
python ../cifar10_fedavg/job.py --n_clients 8 --num_rounds 50 --alpha 0.1
```

### Different Heterogeneity Levels

```bash
# Moderate heterogeneity
python job.py --n_clients 8 --num_rounds 50 --alpha 0.5 --fedprox_mu 0.01

# Very high heterogeneity
python job.py --n_clients 8 --num_rounds 50 --alpha 0.1 --fedprox_mu 0.01
```

## Implementation Details

### Proximal Loss

`FedProxRecipe` sends the configured coefficient in every received model. The raw PyTorch client reads and
validates that metadata, snapshots the newly received global model, and creates `PTFedProxLoss` for that round:

```python
input_model = flare.receive()
mu = input_model.meta[AlgorithmConstants.FEDPROX_MU]
model.load_state_dict(input_model.params)
global_model = copy.deepcopy(model)
fedprox_loss = PTFedProxLoss(mu=mu)

loss = criterion(outputs, labels)
loss += fedprox_loss(model, global_model)
```

Reading μ on every round honors controller-side coefficient changes. Missing, invalid, or non-positive metadata
is a client-contract error: a raw client that ignores the metadata is not FedProx-compatible.

### When to Use FedProx

FedProx is most beneficial when:
1. **High data heterogeneity**: Clients have very different data distributions
2. **Many local steps**: Each client trains for many epochs before synchronization
3. **System heterogeneity**: Clients have different computational capabilities

### Tuning the μ Parameter

Start with the recipe default of μ = 0.01, then tune over several orders of magnitude. Smaller values approach
FedAvg behavior, while larger values constrain client drift more strongly and may eventually limit local learning.

## Performance Comparison

### High Heterogeneity (α=0.1)

| Algorithm | μ | Val Accuracy |
|-----------|---|-------------|
| FedAvg | 0.0 | 80.7% |
| FedProx | 1e-5 | 80.5% |

*Results from 8 clients, 50 rounds, 4 local epochs per round.*

**Note:** In this example, FedProx shows similar performance to FedAvg. The proximal term's effectiveness varies depending on:
- Data heterogeneity level
- Number of local epochs
- Model architecture
- Learning rate

For better performance on highly heterogeneous data, consider:
- **[FedOpt](../cifar10_fedopt/README.md)**: Server-side adaptive optimization (81.2% accuracy)
- **[SCAFFOLD](../cifar10_scaffold/README.md)**: Control variates (83.0% accuracy)

## Viewing Results

After running a job, view the training curves with TensorBoard:

```bash
tensorboard --logdir=/tmp/nvflare/simulation
```

Then open `http://localhost:6006` in your browser.

## Algorithm Comparison

See the [main README](../README.md) for detailed comparisons with other federated learning algorithms.

## References

- [FedProx Paper](https://arxiv.org/abs/1812.06127) - Li et al., 2020
- [NVFlare Documentation](https://nvflare.readthedocs.io/)
- [PTFedProxLoss API](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_opt.pt.fedproxloss.html)
