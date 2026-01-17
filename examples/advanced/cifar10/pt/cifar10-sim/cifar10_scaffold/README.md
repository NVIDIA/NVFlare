# SCAFFOLD Example

This directory implements [SCAFFOLD](https://arxiv.org/abs/1910.06378) (Stochastic Controlled Averaging for Federated Learning) for CIFAR-10 classification using NVFlare's FL simulator.

## Overview

SCAFFOLD addresses the **client drift problem** in federated learning with heterogeneous data by using **control variates**. It maintains correction terms for both the server and each client to reduce variance caused by non-IID data distributions.

**Key Innovation:**
- Tracks control variates `c_global` (server) and `c_local` (per client)
- Applies correction after each optimizer step: `w = w - lr * (c_global - c_local)`
- Updates control variates after local training to track drift
- Significantly improves convergence on highly heterogeneous data

## Usage

### Basic Usage

Run SCAFFOLD with 8 clients, 50 rounds, and high data heterogeneity (α=0.1):

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

### Standard SCAFFOLD with High Heterogeneity

```bash
python job.py --n_clients 8 --num_rounds 50 --alpha 0.1
```

**Expected Result:** ~83.0% validation accuracy (best among compared algorithms!)

### Different Heterogeneity Levels

```bash
# Uniform data distribution
python job.py --n_clients 8 --num_rounds 50 --alpha 1.0

# Moderate heterogeneity
python job.py --n_clients 8 --num_rounds 50 --alpha 0.5

# Very high heterogeneity
python job.py --n_clients 8 --num_rounds 50 --alpha 0.1
```

### Custom Learning Rate

```bash
python job.py --n_clients 8 --num_rounds 50 --alpha 0.1 --lr 0.05
```

### More Clients

```bash
python job.py --n_clients 16 --num_rounds 100 --alpha 0.1 --aggregation_epochs 2
```

## Implementation Details

### Algorithm Steps

**Server Side (each round):**
1. Send global model `w_global` and global control variates `c_global` to clients
2. Receive model updates and control variate updates from clients
3. Aggregate model updates and control variates
4. Update `c_global`

**Client Side (each round):**
1. Initialize local control variates `c_local` (zero on first round)
2. Receive `w_global` and `c_global` from server
3. For each training step:
   - Compute gradients
   - Apply optimizer step
   - **Apply SCAFFOLD correction**: `w = w - lr * (c_global - c_local)`
4. After local training, update `c_local`:
   ```
   c_local_new = c_local - c_global + (w_global - w_local) / (K * lr)
   ```
   where K is the number of local steps
5. Send model difference and control variate difference to server

### Client Implementation

The implementation in `client.py` uses NVFlare's `PTScaffoldHelper`:

```python
# Initialize SCAFFOLD helper (first round)
scaffold_helper = PTScaffoldHelper()
scaffold_helper.init(model=model)

# Load global control variates from server
scaffold_helper.load_global_controls(weights=global_ctrl_weights)

# During training: apply correction after each step
scaffold_helper.model_update(
    model=model,
    curr_lr=curr_lr,
    c_global_para=c_global_para,
    c_local_para=c_local_para
)

# After training: update control variates
scaffold_helper.terms_update(
    model=model,
    curr_lr=curr_lr,
    c_global_para=c_global_para,
    c_local_para=c_local_para,
    model_global=global_model
)
```

### When to Use SCAFFOLD

SCAFFOLD is most effective when:
1. **High data heterogeneity**: Clients have very different data distributions (low alpha)
2. **Many local steps**: Each client trains for multiple epochs
3. **Client drift is a problem**: FedAvg/FedProx struggle to converge

SCAFFOLD's control variates correct for the drift caused by heterogeneous client updates.

## Performance Comparison

### High Heterogeneity (α=0.1)

| Algorithm | Val Accuracy | Improvement | Convergence |
|-----------|-------------|-------------|-------------|
| FedAvg | 80.7% | Baseline | Slow |
| FedProx | 80.5% | -0.2% | Slow |
| FedOpt | 81.2% | +0.5% | Medium |
| **SCAFFOLD** | **83.0%** | **+2.3%** | **Fast** |

*Results from 8 clients, 50 rounds, 4 local epochs per round.*

**Key Observations:**
- SCAFFOLD achieves **2.3% higher accuracy** than FedAvg on highly heterogeneous data
- **Faster convergence**: Reaches target accuracy in fewer rounds
- More robust to data heterogeneity than other methods

## Advantages

✅ **Best performance**: Highest accuracy on non-IID data  
✅ **Faster convergence**: Reaches target accuracy in fewer rounds  
✅ **Variance reduction**: Control variates reduce update variance  
✅ **Theoretically grounded**: Provable convergence guarantees  

## Trade-offs

⚠️ **More complex**: Requires client-side changes and state tracking  
⚠️ **More memory**: Stores control variates (same size as model)  
⚠️ **More communication**: Sends control variate updates  

## Viewing Results

After running a job, view the training curves with TensorBoard:

```bash
tensorboard --logdir=/tmp/nvflare/simulation
```

Then open `http://localhost:6006` in your browser.

You'll notice SCAFFOLD converges faster and achieves higher final accuracy compared to other algorithms.

## Algorithm Comparison

See the [main README](../README.md) for detailed comparisons with other federated learning algorithms, including convergence curves.

## References

- [SCAFFOLD Paper](https://arxiv.org/abs/1910.06378) - Karimireddy et al., 2020
- [NIID-Bench Implementation](https://github.com/Xtra-Computing/NIID-Bench) - Li et al., 2021
- [NVFlare Documentation](https://nvflare.readthedocs.io/)
- [PTScaffoldHelper API](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_opt.pt.scaffold.html)
