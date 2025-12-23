# FedAvg with Homomorphic Encryption

This example demonstrates **real-world federated learning with privacy-preserving secure aggregation** using homomorphic encryption (HE).

## Overview

Homomorphic encryption enables the server to aggregate encrypted model updates from clients **without decrypting them**. This provides an additional layer of privacy protection:

- **Clients**: Encrypt model updates before sending to server
- **Server**: Performs aggregation on encrypted values
- **Server**: Decrypts only the final aggregated result
- **Privacy**: Server never sees individual client updates in plaintext

**Key Benefits:**
- **Privacy protection**: Server cannot inspect individual client contributions
- **Byzantine robustness**: Prevents malicious server from cherry-picking favorable updates
- **No accuracy loss**: Encrypted aggregation is approximating the aggregation of raw updates
- **Production-ready**: Uses secure provisioning with encrypted communication

## Prerequisites

### 1. Setup Secure FL Workspace

**Important**: Homomorphic encryption requires a securely provisioned workspace with proper certificates.

```bash
# From the parent directory
cd workspaces
nvflare provision -p ./secure_project.yml
cp -r ./workspace/secure_project/prod_00 ./secure_workspace
cd ..

# Start FL system with 8 clients
./start_fl_secure.sh 8
```

### 2. Download CIFAR-10 Dataset

```bash
./prepare_data.sh
```

## Usage

### Basic Usage

Run FedAvg with homomorphic encryption using uniform data distribution (α=1.0):

```bash
python job.py --n_clients 8 --num_rounds 50 --alpha 1.0
```

### Command-line Arguments

#### Federated Learning Parameters
- `--n_clients` - Number of FL clients (default: `8`)
- `--num_rounds` - Number of FL rounds (default: `50`)
- `--alpha` - Data heterogeneity parameter (default: `1.0`)
  - Higher values = more uniform/IID data distribution
  - Lower values = more heterogeneous/non-IID distribution

#### Training Parameters
- `--aggregation_epochs` - Local epochs per round (default: `4`)
- `--lr` - Learning rate (default: `0.01`)
- `--batch_size` - Training batch size (default: `64`)
- `--num_workers` - Data loading workers (default: `2`)

#### Other Options
- `--train_idx_root` - Root directory for data splits (default: `/tmp/cifar10_splits`)
- `--name` - Custom job name (default: auto-generated based on parameters)

## Examples

### Standard FedAvg with HE

```bash
python job.py --n_clients 8 --num_rounds 50 --alpha 1.0
```

**Expected Results:**
- Validation accuracy: ~88.7% (same as without HE!)
- Runtime: ~24 minutes (on NVIDIA H100 GPU)
- Full privacy protection for client updates

### Different Data Distributions

```bash
# Moderate heterogeneity
python job.py --n_clients 8 --num_rounds 50 --alpha 0.5

# High heterogeneity
python job.py --n_clients 8 --num_rounds 50 --alpha 0.1
```

### Custom Job Name

```bash
python job.py --n_clients 8 --num_rounds 50 --alpha 1.0 --name "fedavg_he_experiment1"
```

### Selective Layer Encryption (Advanced)

To encrypt only specific layers (e.g., for better performance with large models), modify `job.py`:

```python
# Get model to inspect layer names
model = ModerateCNN()

# Option 1: Encrypt only final layer
encrypt_layers = ["fc3.weight", "fc3.bias"]

# Option 2: Encrypt only fully-connected layers (using regex)
encrypt_layers = "fc"  # Matches all layers with "fc" in the name

# Option 3: Encrypt multiple specific layers
encrypt_layers = ["fc2.weight", "fc2.bias", "fc3.weight", "fc3.bias"]

recipe = FedAvgRecipeWithHE(
    name="cifar10_fedavg_he_partial",
    min_clients=8,
    num_rounds=50,
    initial_model=model,
    train_script="client.py",
    encrypt_layers=encrypt_layers  # Add this parameter
)
```

This can reduce encryption overhead by 50-80% depending on which layers are encrypted.

## Implementation Details

### Homomorphic Encryption Setup

The `job.py` script uses `FedAvgRecipeWithHE` which automatically configures HE filters:

```python
from nvflare.app_opt.pt.recipes.fedavg_he import FedAvgRecipeWithHE

recipe = FedAvgRecipeWithHE(
    name="cifar10_fedavg_he",
    min_clients=8,
    num_rounds=50,
    initial_model=ModerateCNN(),
    train_script="client.py",
    aggregator_data_kind=DataKind.WEIGHT_DIFF,
    encrypt_layers=None  # None = encrypt all layers (default)
)
```

**Selective Encryption:** By default, all model parameters are encrypted. You can optionally use the `encrypt_layers` parameter to encrypt only specific layers:
- Pass a **list of layer names** for exact matches: `encrypt_layers=["fc3.weight", "fc3.bias"]`
- Pass a **regex string** to match patterns: `encrypt_layers="fc"` (matches all layers with "fc" in the name)
- See the "Selective Layer Encryption" example above for details

For this CIFAR-10 example with a small CNN, we use the default (encrypt all layers) for maximum privacy.

### Privacy Workflow

1. **Client Training**:
   - Client trains model locally on private data
   - Computes model update: Δw = w_local - w_global
   
2. **Encryption**:
   - Client encrypts Δw using homomorphic encryption
   - Sends encrypted update to server
   
3. **Secure Aggregation**:
   - Server aggregates encrypted updates: Enc(Σ Δw_i)
   - Server updates global model: w_global = w_global + Σ Δw_i
   - **Server never sees individual Δw_i in plaintext**
   
4. **Decryption**:
   - Client decrypts only the aggregated result for next training round: Σ Δw_i

### Security Guarantees

✅ **Client privacy**: Server cannot see individual model updates  
✅ **Secure communication**: TLS encryption for data in transit  
✅ **Authentication**: Secure provisioning with certificates  

### Performance Considerations

Homomorphic encryption introduces computational overhead:

| Operation | Time Impact |
|-----------|-------------|
| Encryption (client) | +2-3x per round |
| Encrypted aggregation (server) | +5-10x per round |
| Decryption (server) | +1-2x per round |
| Total training time | ~2.5-3x longer |

**Trade-off**: Slower training for strong privacy guarantees.

**Optimization Strategies:**

1. **Selective encryption**: Encrypt only sensitive layers (e.g., final layer)
   - Can reduce overhead by 50-80%
   - See "Selective Layer Encryption" example above

2. **Fewer FL rounds with more local epochs**: Reduce the number of global aggregations
   - Reduces number of encryption/decryption cycles (the expensive operation with HE)
   - Example: 25 rounds × 8 local epochs instead of 50 rounds × 4 local epochs
   - Same total training epochs (25×8 = 50×4 = 200), but half as many encryption cycles

## Comparison: FedAvg vs. FedAvg with HE

| Configuration | Val Accuracy | Runtime | Privacy |
|--------------|-------------|---------|---------|
| FedAvg (no HE) | 88.7% | ~9 min | Standard |
| FedAvg + HE | 88.7% | ~24 min | **High** |

*Results from 8 clients, 50 rounds, 4 local epochs per round on NVIDIA H100 GPU.*

**Key Observations:**
- ✅ **No accuracy loss**: HE preserves model quality
- ⏱️ **2.7x slower**: Due to encryption/decryption overhead
- 🔒 **Stronger privacy**: Server cannot inspect client updates

![FedAvg vs FedAvg with HE](../figs/fedavg_vs_fedavg_he.png)

## When to Use Homomorphic Encryption

### Use HE When:
- ✅ You don't trust the aggregation server
- ✅ Regulatory compliance requires privacy (GDPR, HIPAA, etc.)
- ✅ Protecting client contributions is critical
- ✅ You can tolerate 2-3x training time overhead

### Consider Alternatives When:
- ❌ Server is fully trusted (internal deployment)
- ❌ Training time is critical bottleneck
- ❌ Differential privacy is sufficient
- ❌ Using small models/frequent updates (overhead compounds)

## Limitations

⚠️ **Server optimizers not supported**: FedOpt (Adam, SGD with momentum on server) cannot work with encrypted values  
⚠️ **Computational overhead**: 2-3x slower training  
⚠️ **Memory requirements**: Encrypted messages are larger  
⚠️ **Requires secure provisioning**: Must use properly configured workspace  

## Monitoring Training

### Using Admin Console

In a separate terminal:

```bash
./workspaces/secure_workspace/admin@nvidia.com/startup/fl_admin.sh
```

Commands:
- `list_jobs` - View running jobs
- `check_status server` - Check server status
- `abort_job <job_id>` - Stop a running job

### Viewing Logs

Client and server logs are in:
```
workspaces/secure_workspace/site-{1..8}/log.txt
workspaces/secure_workspace/localhost/log.txt
```

## Troubleshooting

### "Secure workspace not found"

Ensure you provisioned and copied the workspace:
```bash
cd workspaces
nvflare provision -p ./secure_project.yml
cp -r ./workspace/secure_project/prod_00 ./secure_workspace
```

### "HE encryption failed"

- Verify secure workspace is properly provisioned
- Check that all certificates are present
- Ensure FL system was started with secure workspace

### Training is very slow

This is expected! HE adds 2-3x overhead. For faster experiments without privacy requirements, use the [simulation examples](../../cifar10-sim/README.md).

## Related Examples

- **[FedAvg with MLFlow](../cifar10_fedavg_mlflow/README.md)**: Same algorithm with experiment tracking
- **[Simulation Examples](../../cifar10-sim/README.md)**: Fast experiments without production overhead
- **[Main README](../README.md)**: Overview of all real-world examples

## References

- [NVFlare HE Documentation](https://nvflare.readthedocs.io/en/main/programming_guide/filters.html)
- [Homomorphic Encryption Blog Post](https://developer.nvidia.com/blog/federated-learning-with-homomorphic-encryption/)
- [Secure Provisioning Guide](https://nvflare.readthedocs.io/en/main/programming_guide/provisioning_system.html)
- [FedAvg Paper](https://arxiv.org/abs/1602.05629)
- [TenSEAL Library](https://github.com/OpenMined/TenSEAL) - Used by NVFlare for HE

