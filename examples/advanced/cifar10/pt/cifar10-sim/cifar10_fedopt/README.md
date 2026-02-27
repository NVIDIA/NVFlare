# FedOpt Example

This directory implements [FedOpt](https://arxiv.org/abs/2003.00295) for CIFAR-10 classification using NVFlare's FL simulator.

## Overview

FedOpt improves upon FedAvg by applying **adaptive optimization algorithms on the server side** when aggregating client updates. Instead of simple weighted averaging, the server uses optimizers like SGD with momentum, Adam, Yogi, or Adagrad to update the global model.

**Key Innovation:**
- **FedAvg**: `w_{t+1} = w_t + Σ(n_i / n_total) * Δw_i`
- **FedOpt**: `w_{t+1} = ServerOptimizer(w_t, Σ(n_i / n_total) * Δw_i)`

This allows the global model to benefit from momentum and adaptive learning rates, leading to better convergence, especially with heterogeneous data.

**This example uses:** Server-side SGD with momentum (0.9)

## Usage

### Basic Usage

Run FedOpt with 8 clients, 50 rounds, and high data heterogeneity (α=0.1):

```bash
python job.py --n_clients 8 --num_rounds 50 --alpha 0.1
```

### Command-line Arguments

#### Federated Learning Parameters
- `--n_clients` - Number of FL clients (default: `8`)
- `--num_rounds` - Number of FL rounds (default: `50`)
- `--alpha` - Data heterogeneity parameter (default: `0.5`)
  - Lower values = more heterogeneous/non-IID data

#### Training Parameters
- `--aggregation_epochs` - Local epochs per round (default: `4`)
- `--lr` - Client learning rate (default: `0.01`)
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

### Standard FedOpt with High Heterogeneity

```bash
python job.py --n_clients 8 --num_rounds 50 --alpha 0.1
```

**Expected Result:** ~81.2% validation accuracy

### Different Heterogeneity Levels

```bash
# Uniform data distribution
python job.py --n_clients 8 --num_rounds 50 --alpha 1.0

# Moderate heterogeneity
python job.py --n_clients 8 --num_rounds 50 --alpha 0.5

# Very high heterogeneity
python job.py --n_clients 8 --num_rounds 50 --alpha 0.1
```

### Custom Client Learning Rate

```bash
python job.py --n_clients 8 --num_rounds 50 --alpha 0.1 --lr 0.05
```

### More Clients

```bash
python job.py --n_clients 16 --num_rounds 100 --alpha 0.1 --aggregation_epochs 2
```

## Implementation Details

### Server-Side Optimization

FedOpt is configured in `job.py` using `FedOptRecipe` with `optimizer_args` and optional `lr_scheduler_args`:

```python
from nvflare.app_opt.pt.recipes.fedopt import FedOptRecipe

recipe = FedOptRecipe(
    name="cifar10_fedopt",
    # ... other parameters ...
    optimizer_args={"class_path": "torch.optim.SGD", "args": {"lr": 1.0, "momentum": 0.6}},
    lr_scheduler_args={
        "class_path": "torch.optim.lr_scheduler.CosineAnnealingLR",
        "args": {"T_max": num_rounds, "eta_min": 0.9},
    },
)
```

### Supported Server Optimizers

NVFlare supports multiple server-side optimizers:
- **SGD**: Stochastic Gradient Descent with momentum
- **Adam**: Adaptive moment estimation
- **Yogi**: Variant of Adam with better convergence
- **Adagrad**: Adaptive learning rates per parameter

This example uses SGD with momentum for stability.

### Client Training

Client training (`client.py`) remains identical to FedAvg:
- Each client performs standard local training
- Sends model differences back to server
- No changes needed in client code

The magic happens on the server side during aggregation!

### When to Use FedOpt

FedOpt is particularly effective when:
1. **Heterogeneous data**: Clients have non-IID data distributions
2. **Noisy updates**: Client updates have high variance
3. **Slow convergence**: FedAvg converges slowly on your problem

The server-side momentum helps smooth out noisy client updates and accelerates convergence.

## Performance Comparison

### High Heterogeneity (α=0.1)

| Algorithm | Server Optimizer | Val Accuracy | Improvement |
|-----------|-----------------|-------------|-------------|
| FedAvg | None (simple avg) | 80.7% | Baseline |
| FedProx | None | 80.5% | -0.2% |
| **FedOpt** | **SGD + momentum** | **81.2%** | **+0.5%** |
| SCAFFOLD | None | 83.0% | +2.3% |

*Results from 8 clients, 50 rounds, 4 local epochs per round.*

**Key Observation:** FedOpt achieves better performance than FedAvg/FedProx with minimal code changes and no client-side modifications.

## Advantages

✅ **Easy to implement**: Only server configuration changes needed  
✅ **No client changes**: Client code remains identical to FedAvg  
✅ **Better convergence**: Momentum smooths optimization trajectory  
✅ **Flexible**: Can swap in different optimizers (Adam, Yogi, etc.)  

## Viewing Results

After running a job, view the training curves with TensorBoard:

```bash
tensorboard --logdir=/tmp/nvflare/simulation
```

Then open `http://localhost:6006` in your browser.

## Algorithm Comparison

See the [main README](../README.md) for detailed comparisons with other federated learning algorithms.

## Customization

To try different server optimizers, modify `job.py`:

```python
# Try Adam instead of SGD
recipe = FedOptRecipe(
    # ... other parameters ...
    optimizer_args={
        "class_path": "torch.optim.Adam",
        "args": {"lr": 0.01, "betas": (0.9, 0.999)},
    },
)
```

## References

- [FedOpt Paper](https://arxiv.org/abs/2003.00295) - Reddi et al., 2020
- [NVFlare Documentation](https://nvflare.readthedocs.io/)
- [NVFlare FedOptRecipe](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_opt.pt.recipes.fedopt.html)

