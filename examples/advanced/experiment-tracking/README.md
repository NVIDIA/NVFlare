# Experiment Tracking Overview

This section demonstrates how NVIDIA FLARE supports flexible experiment tracking through various backends such as MLflow, TensorBoard, and Weights & Biases.

## Overview

These examples use the Recipe API with the `add_experiment_tracking()` utility for simplified configuration:

```python
from nvflare.app_opt.pt.recipes import FedAvgRecipe
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

## Key Highlights

1. **Centralized Streaming to Server Receiver**
   FLARE can stream all client training metrics to a server-side receiver, allowing a consolidated view of all clients' training progress.

2. **Pluggable Metrics Receivers**
   FLARE allows plugging in different metrics receivers, independent of whether they are used on the server side or client side. This enables streaming metrics to various observability frameworks such as:
   - MLflow
   - TensorBoard
   - Weights & Biases

   **Benefit**: This makes it easy to switch between different experiment tracking frameworks without modifying the training code that logs metrics—only the receiver configuration needs to change.

3. **Site-Specific Metric Streaming**
   FLARE also supports streaming metrics from each client to site-specific receivers. This enables local tracking at each site, with configuration changes only.

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
- [Recipe API Documentation](https://nvflare.readthedocs.io/en/main/programming_guide/job_recipes.html)
- [Client Tracking APIs](https://nvflare.readthedocs.io/en/main/programming_guide/experiment_tracking/experiment_tracking_apis.html)
