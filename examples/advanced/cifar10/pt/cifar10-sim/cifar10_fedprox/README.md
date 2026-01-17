# FedProx Example

This directory implements [FedProx](https://arxiv.org/abs/1812.06127) for CIFAR-10 classification using NVFlare's FL simulator.

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
python job.py --n_clients 8 --num_rounds 50 --alpha 0.1 --fedproxloss_mu 1e-5
```

### Command-line Arguments

#### FedProx-Specific Parameters
- `--fedproxloss_mu` - Proximal term coefficient μ (default: `0.0`)
  - Typical values: `1e-5` to `1e-3`
  - Higher values = stronger regularization (less client drift)
  - Set to `0.0` to disable (equivalent to FedAvg)

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
python job.py --n_clients 8 --num_rounds 50 --alpha 0.1 --fedproxloss_mu 1e-5
```

**Expected Result:** ~80.5% validation accuracy

### Stronger Proximal Regularization

```bash
python job.py --n_clients 8 --num_rounds 50 --alpha 0.1 --fedproxloss_mu 1e-4
```

### Compare with FedAvg (μ=0)

```bash
# FedProx with μ=0 is equivalent to FedAvg
python job.py --n_clients 8 --num_rounds 50 --alpha 0.1 --fedproxloss_mu 0.0
```

### Different Heterogeneity Levels

```bash
# Moderate heterogeneity
python job.py --n_clients 8 --num_rounds 50 --alpha 0.5 --fedproxloss_mu 1e-5

# Very high heterogeneity
python job.py --n_clients 8 --num_rounds 50 --alpha 0.1 --fedproxloss_mu 1e-5
```

## Implementation Details

### Proximal Loss

The implementation in `client.py` adds the proximal term during training:

```python
# Standard cross-entropy loss
loss = criterion(outputs, labels)

# Add FedProx proximal term
if fedproxloss_mu > 0:
    proximal_loss = (μ/2) * ||w_local - w_global||²
    loss += proximal_loss
```

The `PTFedProxLoss` class computes the L2 distance between local and global model parameters.

### When to Use FedProx

FedProx is most beneficial when:
1. **High data heterogeneity**: Clients have very different data distributions
2. **Many local steps**: Each client trains for many epochs before synchronization
3. **System heterogeneity**: Clients have different computational capabilities

### Tuning the μ Parameter

- **Too small (μ < 1e-6)**: Minimal effect, similar to FedAvg
- **Optimal range (μ = 1e-5 to 1e-4)**: Balances local adaptation and global consistency
- **Too large (μ > 1e-3)**: Over-constrains local training, may hurt performance

Start with μ = 1e-5 and adjust based on your data heterogeneity.

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

