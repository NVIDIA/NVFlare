# Hello PyTorch with Weights & Biases

Example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) to train an image classifier
using federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629)) and [PyTorch](https://pytorch.org/)
with **Weights & Biases (WandB)** experiment tracking.

> **_NOTE:_** This example uses the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset and will load its data within the trainer code.

## Overview

This example demonstrates Weights & Biases tracking with flexible options for server-side or client-side metric collection:

```python
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe.utils import add_experiment_tracking

# Create FedAvg recipe
from model import Net  # Your model definition

recipe = FedAvgRecipe(
    name="fedavg_wandb",
    min_clients=2,
    num_rounds=5,
    initial_model=Net(),
    train_script="client.py",
)

# Configure WandB settings
wandb_config = {
    "mode": "online",
    "wandb_args": {
        "project": "wandb-experiment",
        "name": "federated-learning",
        "tags": ["baseline", "cifar10"],
        "config": {"architecture": "CNN", "optimizer": "SGD"},
    }
}

# Server-side tracking (centralized)
add_experiment_tracking(recipe, "wandb", tracking_config=wandb_config)

recipe.run()
```

## Setup and Running

### 1. Install Requirements

```bash
cd examples/advanced/experiment-tracking/wandb
python -m pip install -r requirements.txt
```

### 2. Download Data

```bash
cd examples/advanced/experiment-tracking
./prepare_data.sh
```

### 3. Login to Weights & Biases

**IMPORTANT**: You must login before running:

```bash
python3
>>> import wandb
>>> wandb.login()
```

Provide your API key when prompted. Get your key from https://wandb.ai/authorize

### 4. Run the Experiment

**Server-side tracking** (default - centralized metrics):
```bash
python job.py
```

**Client-side tracking** (decentralized - each site separate):
```bash
python job.py --streamed_to_clients --disable_server_tracking
```

**Both server and client tracking**:
```bash
python job.py --streamed_to_clients
```

---

## Accessing Results

### View on WandB Dashboard

1. Visit https://wandb.ai/
2. Login with your credentials
3. Navigate to your project ("wandb-experiment" by default)
4. View your runs with metrics, charts, and system info

### Local Files

WandB also creates a local `wandb` directory:
```bash
ls /tmp/nvflare/jobs/workdir/server/wandb/
```

With `mode: "online"`, these files sync automatically to the WandB cloud.

---

## How It Works

This example supports both **server-side** and **client-side** tracking modes.

### Logging Metrics (in `client.py`)

Your training script logs metrics using NVFlare's tracking API:

```python
from nvflare.client.tracking import WandBWriter

wandb = WandBWriter()
wandb.log({"train/loss": loss, "train/accuracy": accuracy}, step=epoch)
```

This creates a **local event** (`analytix_log_stats`) on the **NVFlare Client** side.

### Mode 1: Server-Side Tracking (Centralized)

```python
# All clients stream to server, server logs to WandB
add_experiment_tracking(recipe, "wandb", tracking_config=wandb_config)
```

**Event Flow**:
1. User's training script (`client.py`) logs metrics via `WandBWriter`
2. NVFlare Client creates local event: `analytix_log_stats`
3. `ConvertToFedEvent` widget converts event to federated event: `fed.analytix_log_stats`
4. Federated event sent to NVFlare Server
5. `WandBReceiver` on server listens for `fed.analytix_log_stats`
6. Receiver writes to WandB

**Result**: Single centralized WandB run with all clients' metrics combined.

### Mode 2: Client-Side Tracking (Decentralized)

```python
# Each client logs to its own WandB run
for site_name in ["site-1", "site-2"]:
    receiver = WandBReceiver(**client_config)
    recipe.job.to(receiver, site_name, id="wandb_receiver")
```

**Event Flow**:
1. User's training script (`client.py`) logs metrics via `WandBWriter`
2. NVFlare Client creates local event: `analytix_log_stats`
3. `WandBReceiver` on NVFlare Client listens for `analytix_log_stats` (no conversion to federated event)
4. Receiver writes to WandB

**Result**: Each client has its own separate WandB run. Metrics never leave the client.

### Key Terminology

To avoid confusion:
- **`client.py`**: Your training script (user code that logs metrics)
- **NVFlare Client**: The FL client runtime that executes your training script
- **NVFlare Server**: The FL server that coordinates training
- **Server-Side Tracking**: Receiver on NVFlare Server (listens for `fed.analytix_log_stats`)
- **Client-Side Tracking**: Receiver on NVFlare Client (listens for `analytix_log_stats`)

---

## Configuration Options

### Change Project/Run Name

```python
wandb_config = {
    "mode": "online",
    "wandb_args": {
        "project": "my-fl-project",
        "name": "experiment-001",
        "tags": ["production", "v2"],
        "notes": "Testing new architecture",
        "config": {
            "learning_rate": 0.001,
            "batch_size": 32,
        }
    }
}
```

### Offline Mode

For environments without internet:

```python
wandb_config = {
    "mode": "offline",  # Logs locally, sync later
    "wandb_args": {...}
}
```

Then sync later:
```bash
wandb sync /tmp/nvflare/jobs/workdir/server/wandb/
```

### CLI Arguments

The example supports several CLI arguments:

```bash
python job.py \
    --n_clients 3 \
    --num_rounds 10 \
    --script client.py \
    --streamed_to_clients \
    --disable_server_tracking
```

- `--n_clients`: Number of clients (default: 2)
- `--num_rounds`: Training rounds (default: 5)
- `--script`: Training script path (default: client.py)
- `--launch_external_process`: Run training in external process
- `--streamed_to_clients`: Enable client-side tracking (default: disabled)
- `--disable_server_tracking`: Disable server-side tracking (default: enabled)
- `--export_config`: Export config without running

---

## Tracking Modes Comparison

| Mode | Command | Use Case | WandB Runs |
|------|---------|----------|------------|
| **Server-only** | `python job.py` | Centralized monitoring | 1 run (all clients combined) |
| **Client-only** | `python job.py --streamed_to_clients --disable_server_tracking` | Site-specific analysis | N runs (1 per client) |
| **Both** | `python job.py --streamed_to_clients` | Complete visibility | N+1 runs (1 per client + 1 aggregated) |

---

## Client Code Integration

In your training script:

```python
from nvflare.client.tracking import WandBWriter

# Create writer
wandb = WandBWriter()

# Log metrics
for epoch in range(num_epochs):
    loss = train_one_epoch()
    accuracy = evaluate()

    # Log to WandB (through NVFlare)
    wandb.log({
        "train/loss": loss,
        "train/accuracy": accuracy,
        "epoch": epoch,
    }, step=global_step)

# Log artifacts
wandb.log_artifact("/path/to/model.pth", "model", "checkpoint")
```

**Note**: Uses `WandBWriter` instead of native `wandb.log()` for NVFlare integration.

---

## Advanced: Custom WandB Configuration

### Add Custom Metrics

```python
from nvflare.client.tracking import WandBWriter

writer = WandBWriter()

# Log metrics
writer.log({"metrics/accuracy": accuracy}, step=step)

# Log multiple metrics at once
writer.log({
    "metrics/accuracy": accuracy,
    "metrics/f1_score": f1,
    "metrics/precision": precision
}, step=step)
```

**Note**: For advanced WandB visualizations (like `wandb.plot.*`), use the native WandB API directly in your script alongside WandBWriter for basic metrics.

### Log System Metrics

WandB automatically tracks:
- GPU utilization
- CPU usage
- Memory consumption
- Network I/O

View in the "System" tab of your run.

---

## Troubleshooting

### "Not logged in" Error

Make sure to run `wandb.login()` before starting the job:
```bash
python3 -c "import wandb; wandb.login()"
```

### Metrics Not Appearing

Check that WandB mode is "online":
```python
wandb_config = {"mode": "online", ...}
```

### Multiple Clients Same Run

Ensure client-side tracking uses unique run names:
```python
"name": f"nvflare-{site_name}",  # Different per client
```

---

## Additional Resources

- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [WandB Python Library](https://docs.wandb.ai/ref/python/)
- [Experiment Tracking Guide](https://nvflare.readthedocs.io/en/main/programming_guide/experiment_tracking.html)
- [Recipe API Documentation](https://nvflare.readthedocs.io/en/main/user_guide/data_scientist_guide/job_recipe.html)
