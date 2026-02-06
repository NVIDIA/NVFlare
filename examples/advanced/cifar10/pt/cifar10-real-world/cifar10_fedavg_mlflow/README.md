# FedAvg with MLFlow Experiment Tracking

This example demonstrates **real-world federated learning deployment** with FedAvg and centralized experiment tracking using MLFlow.

## Overview

In production federated learning, researchers typically don't have direct access to individual client or server logs. This example shows how to:

1. **Stream metrics to MLFlow**: All training metrics are automatically forwarded to a centralized MLFlow server
2. **Monitor training in real-time**: View training progress from multiple clients in one dashboard
3. **Compare experiments**: Track multiple runs with different hyperparameters
4. **Download results**: Automatically retrieve trained models and artifacts

**Key Features:**
- Uses Production Environment (ProdEnv) with secure provisioning
- Client code uses NVFlare's `SummaryWriter` for transparent metric streaming
- Server forwards metrics to MLFlow tracking server
- Support for experiment comparison and artifact management

## Prerequisites

### 1. Start MLFlow Server

In a separate terminal, start the MLFlow tracking server:

```bash
mlflow server --host 0.0.0.0 --port 5000
```

The MLFlow UI will be available at `http://localhost:5000`

### 2. Setup FL Workspace

Ensure you have created and started the secure FL workspace:

```bash
# From the parent directory
cd workspaces
nvflare provision -p ./secure_project.yml
cp -r ./workspace/secure_project/prod_00 ./secure_workspace
cd ..

# Start FL system with 8 clients
./start_fl_secure.sh 8
```

## Usage

### Basic Usage

Run FedAvg with MLFlow tracking using uniform data distribution (α=1.0):

```bash
python job.py --n_clients 8 --num_rounds 50 --alpha 1.0 --tracking_uri http://localhost:5000
```

### Command-line Arguments

#### Experiment Tracking
- `--tracking_uri` - MLFlow tracking server URI (default: `http://localhost:5000`)
  - **Required**: Must match your running MLFlow server address

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

### Standard MLFlow Tracking

```bash
python job.py --n_clients 8 --num_rounds 50 --alpha 1.0 --tracking_uri http://localhost:5000
```

**Expected Results:**
- Validation accuracy: ~88.7%
- Runtime: ~9 minutes (on NVIDIA H100 GPU)
- All metrics visible in MLFlow UI

### Different Data Heterogeneity

```bash
# Moderate heterogeneity
python job.py --n_clients 8 --num_rounds 50 --alpha 0.5 --tracking_uri http://localhost:5000

# High heterogeneity
python job.py --n_clients 8 --num_rounds 50 --alpha 0.1 --tracking_uri http://localhost:5000
```

### Custom Job Name

```bash
python job.py --n_clients 8 --num_rounds 50 --alpha 1.0 \
  --tracking_uri http://localhost:5000 \
  --name "fedavg_alpha1.0_experiment1"
```

### Remote MLFlow Server

If your MLFlow server is running on a different machine:

```bash
python job.py --n_clients 8 --num_rounds 50 --alpha 1.0 \
  --tracking_uri http://192.168.1.100:5000
```

## Implementation Details

### Experiment Tracking Setup

The `job.py` script adds MLFlow tracking with a single line:

```python
from nvflare.recipe.utils import add_experiment_tracking

add_experiment_tracking(
    recipe, 
    tracking_type="mlflow", 
    tracking_config={"tracking_uri": mlflow_tracking_uri}
)
```

### Client-Side Metrics

Client code (`client.py`) uses NVFlare's `SummaryWriter`:

```python
from nvflare.client.tracking import SummaryWriter

summary_writer = SummaryWriter()

# During training
summary_writer.add_scalar(tag="train_loss", scalar=avg_loss, global_step=epoch)
summary_writer.add_scalar(tag="val_acc_global_model", scalar=val_acc, global_step=round)
```

All metrics are automatically streamed to the server, which forwards them to MLFlow.

### Tracked Metrics

The following metrics are logged per client:
- `train_loss` - Average training loss per epoch
- `val_acc_global_model` - Validation accuracy on received global model
- `learning_rate` - Current learning rate
- `diff_norm` - L2 norm of model differences
- `global_round` - Current FL round
- `global_epoch` - Cumulative epoch count

### Result Download

After training completes, results are automatically downloaded:

```python
run.get_result()
```

Output:
```
Result can be found in: workspaces/secure_workspace/admin@nvidia.com/transfer/<job_id>
```

## Viewing Results

### MLFlow UI

1. Open `http://localhost:5000` in your browser
2. Find your experiment by job ID (displayed in terminal output)
3. View metrics, parameters, and compare runs
4. Download trained models from the Artifacts tab

### Metrics Available

In the MLFlow UI, you can view:
- **Training curves**: Loss, accuracy, learning rate over time
- **Per-client metrics**: Compare performance across different clients
- **Global model metrics**: Validation accuracy on the aggregated model
- **Run comparison**: Side-by-side comparison of different experiments
- **Artifacts**: Trained models, configuration files, logs

![MLFlow UI](../figs/mlflow.png)

## Switching Tracking Systems

You can easily switch to other tracking systems by changing `tracking_type`. If provided, the `tracking_config` keys must match each receiver's constructor arguments.

### TensorBoard

The TensorBoard receiver accepts `tb_folder` (directory name under the job run directory, default `"tb_events"`):

```python
add_experiment_tracking(
    recipe,
    tracking_type="tensorboard",
    tracking_config={"tb_folder": "tb_events"}  # Directory name where TensorBoard event files are stored (relative to job run directory on the server)
)
```

### Weights & Biases

The WandB receiver requires `wandb_args` with `project`, `group`, `job_type`, and `name` (base run name). Optional top-level `mode` (e.g. `"online"` or `"offline"`):

```python
add_experiment_tracking(
    recipe,
    tracking_type="wandb",
    tracking_config={
        "wandb_args": {
            "name": "cifar10-fedavg",  # base run name (receiver appends site and job id)
            "project": "nvflare",
            "group": "nvidia",
            "job_type": "training",
            "entity": "your-wandb-username",  # optional
        },
        "mode": "online",  # optional, default "offline"
    }
)
```

**Authentication:** WandB reads the `WANDB_API_KEY` environment variable. The WandB receiver runs on the FL server, so the variable must be set in the **server's** environment (e.g. in the script or systemd unit that starts the server)—setting it in your local shell before `python job.py` does not apply, since the job is submitted to a remote server. In small POCs where you have shell access to the server, you can `export WANDB_API_KEY=...` there or run `wandb.login()` once on that machine; in production, the site operator must configure the key where the server process runs (e.g. via startup env, secrets manager, or container config). Get your key from [wandb.ai/authorize](https://wandb.ai/authorize).

**No changes needed in client code!** The same `SummaryWriter` API works with all tracking systems.

## Performance

| Configuration | Val Accuracy | Runtime | Job ID |
|--------------|-------------|---------|---------|
| FedAvg + MLFlow (α=1.0) | 88.7% | ~9 min | (auto-generated) |

*Results from 8 clients, 50 rounds, 4 local epochs per round on NVIDIA H100 GPU.*

## Troubleshooting

### MLFlow Server Not Accessible

If clients can't reach the MLFlow server:
- Verify MLFlow is running: `curl http://localhost:5000`
- Check firewall settings if using remote server
- Ensure `--tracking_uri` matches your MLFlow server address

### Job Submission Fails

- Ensure FL system is running: `./start_fl_secure.sh 8`
- Check admin console for errors
- Verify secure workspace was provisioned correctly

## Related Examples

- **[FedAvg with HE](../cifar10_fedavg_he/README.md)**: Same experiment with homomorphic encryption
- **[Simulation Examples](../../cifar10-sim/README.md)**: Compare different FL algorithms without production setup

## References

- [NVFlare Experiment Tracking Guide](https://nvflare.readthedocs.io/en/main/programming_guide/experiment_tracking.html)
- [MLFlow Documentation](https://mlflow.org/docs/latest/index.html)
- [Production Environment (ProdEnv)](https://nvflare.readthedocs.io/en/main/programming_guide/fed_job_api.html)
- [FedAvg Paper](https://arxiv.org/abs/1602.05629)

