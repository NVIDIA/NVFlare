# Experiment Tracking Overview

This section demonstrates how NVIDIA FLARE supports flexible experiment tracking through various backends such as MLflow, TensorBoard, and Weights & Biases.

## Overview

These examples use the Recipe API with the `add_experiment_tracking()` utility for simplified configuration:

```python
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe.utils import add_experiment_tracking

# Create your training recipe
recipe = FedAvgRecipe(
    name="my_job",
    min_clients=2,
    num_rounds=5,
    initial_model=MyModel(),
    train_script="train.py",
)

# Add experiment tracking with ONE line!
add_experiment_tracking(recipe, "mlflow")  # or "tensorboard" or "wandb"

recipe.run()
```

## How Metric Tracking Works

Understanding the flow from logging metrics in your training code to viewing them in tracking tools:

### 1. Logging Metrics (in `client.py`)

Your training script logs metrics using NVFlare's tracking API:

```python
from nvflare.client.tracking import SummaryWriter

summary_writer = SummaryWriter()
summary_writer.add_scalar("train_loss", loss_value, step=epoch)
```

This creates a **local event** called `analytix_log_stats` on the **NVFlare Client** side.

### 2. Event Flow

There are two patterns for metric delivery:

#### Pattern A: Server-Side Tracking (Default, Centralized)

**Flow**:
1. User's training script (`client.py`) logs metrics via `SummaryWriter`
2. NVFlare Client creates local event: `analytix_log_stats`
3. `ConvertToFedEvent` widget converts event to federated event: `fed.analytix_log_stats`
4. Federated event sent to NVFlare Server
5. Receiver on server listens for `fed.analytix_log_stats`
6. Receiver writes to tracking backend (MLflow/TensorBoard/WandB)

**Result**: All client metrics collected in one central location.

#### Pattern B: Client-Side Tracking (Decentralized)

**Flow**:
1. User's training script (`client.py`) logs metrics via `SummaryWriter`
2. NVFlare Client creates local event: `analytix_log_stats`
3. Receiver on NVFlare Client listens for `analytix_log_stats` (no conversion to federated event)
4. Receiver writes to local tracking backend (MLflow/TensorBoard/WandB)

**Result**: Each client has its own separate tracking instance. Metrics never leave the client.

### 3. Key Terminology

To avoid confusion:

- **`client.py`**: Your training script (user code that logs metrics)
- **NVFlare Client**: The FL client runtime that executes your training script
- **NVFlare Server**: The FL server that coordinates training
- **Server-Side Tracking**: Receiver deployed on NVFlare Server (listens for `fed.analytix_log_stats`)
- **Client-Side Tracking**: Receiver deployed on NVFlare Client (listens for `analytix_log_stats`)

### 4. Benefits

**Pluggable Backends**: Your training script (`client.py`) doesn't change when switching between MLflow, TensorBoard, or WandB. Only the receiver configuration changes.

**Flexible Deployment**: Choose centralized (server-side) or decentralized (client-side) tracking based on your privacy and operational needs.

## Configuration Flexibility

FLARE allows seamless switching between centralized and decentralized experiment tracking:

- The training code remains unchanged.
- You can control:
  - Where the metrics are sent (server or site-local).
  - Which experiment tracking framework is used.

This flexible design enables easy integration with different observability platforms, tailored to your deployment needs.

---

## Examples

All examples use the Recipe API for simplified configuration. Choose your preferred tracking framework:

---

### Prerequisites

Please make sure you set up a virtual environment and follow the installation steps on the [example root readme](../../README.md).

This folder contains examples for [experiment tracking](https://nvflare.readthedocs.io/en/main/programming_guide/experiment_tracking.html)
using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) to
train an image classifier using federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629)) and
[PyTorch](https://pytorch.org/) as the deep learning training framework.

---

### TensorBoard

The `tensorboard` folder contains the [TensorBoard Streaming](./tensorboard/README.md) example
showcasing centralized TensorBoard streaming from clients to server using the Recipe API.

**Key Features**:
- Simple one-line integration: `add_experiment_tracking(recipe, "tensorboard")`
- Automatic server-side aggregation of client metrics
- View all client metrics in a single TensorBoard dashboard

### MLflow

The `mlflow` folder contains [three examples](./mlflow/README.md) demonstrating different MLflow configurations:

1. **`hello-pt-mlflow`**: Server-side centralized tracking
2. **`hello-pt-mlflow-client`**: Site-specific decentralized tracking
3. **`hello-lightning-mlflow`**: PyTorch Lightning integration with MLflow

**Key Features**:
- Flexible server-side or client-side tracking
- Experiment and run management
- Automatic metric logging and artifact storage

### Weights and Biases

The `wandb` folder contains the [Hello PyTorch with Weights and Biases](./wandb/README.md) example
showing how to use W&B for experiment tracking with both server-side and client-side options.

**Key Features**:
- Online/offline mode support
- Rich experiment dashboards
- Configurable server or client-side tracking

---

## Quick Start Guide

### 1. Install Requirements

```bash
cd examples/advanced/experiment-tracking/<framework>
pip install -r requirements.txt
```

### 2. Download Data (if needed)

```bash
./prepare_data.sh
```

### 3. Run Example

```bash
cd <framework>/jobs/<job_name>/code
python job.py
```

### 4. View Results

**TensorBoard**:
```bash
tensorboard --logdir=/tmp/nvflare/jobs/workdir/server/simulate_job/tb_events
```

**MLflow**:
```bash
mlflow ui --backend-store-uri /tmp/nvflare/jobs/workdir/server/simulate_job/mlruns
```

**Weights & Biases**:
Visit your W&B dashboard at https://wandb.ai

---

## Adding Tracking to Your Own Recipe

### Server-Side Tracking (Centralized)

```python
from nvflare.recipe.utils import add_experiment_tracking

# After creating your recipe
add_experiment_tracking(recipe, "mlflow", tracking_config={
    "tracking_uri": "file:///tmp/mlruns",
    "kw_args": {
        "experiment_name": "my-experiment",
        "run_name": "my-run",
    }
})
```

### Client-Side Tracking (Decentralized)

```python
from nvflare.app_opt.tracking.mlflow.mlflow_receiver import MLflowReceiver

# Add tracking to specific clients
for site_name in ["site-1", "site-2"]:
    receiver = MLflowReceiver(
        tracking_uri=f"file:///tmp/{site_name}/mlruns",
        kw_args={"experiment_name": f"{site_name}-experiment"}
    )
    recipe.job.to(receiver, site_name, id="mlflow_receiver")
```

---

## Additional Resources

- [Experiment Tracking Programming Guide](https://nvflare.readthedocs.io/en/main/programming_guide/experiment_tracking.html)
- [Recipe API Documentation](https://nvflare.readthedocs.io/en/main/user_guide/data_scientist_guide/job_recipe.html)
- [Client Tracking APIs](https://nvflare.readthedocs.io/en/main/programming_guide/experiment_tracking/experiment_tracking_apis.html)
