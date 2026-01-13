# Hello PyTorch with TensorBoard Streaming

Example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) to train an image classifier
using federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629)) and [PyTorch](https://pytorch.org/)
as the deep learning training framework with **TensorBoard experiment tracking**.

This example demonstrates the **Recipe API** for easily adding TensorBoard streaming to FL training jobs.

> **_NOTE:_** This example uses the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset and will load its data within the trainer code.

## Overview

This example uses the `FedAvgRecipe` with the `add_experiment_tracking()` utility to easily add TensorBoard streaming:

```python
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe.utils import add_experiment_tracking

# Create training recipe
recipe = FedAvgRecipe(
    name="fedavg_tensorboard",
    min_clients=2,
    num_rounds=5,
    initial_model=SimpleNetwork(),
    train_script="client.py",
)

# Add TensorBoard tracking with one line!
add_experiment_tracking(recipe, "tensorboard", tracking_config={"tb_folder": "tb_events"})

# Run
recipe.run()
```

## Setup

### 1. Install Requirements

Install additional requirements:

```bash
cd examples/advanced/experiment-tracking/tensorboard
python -m pip install -r requirements.txt
```

### 2. Download Data

Pre-download the CIFAR-10 dataset to avoid multiple sites downloading simultaneously:

```bash
cd examples/advanced/experiment-tracking
./prepare_data.sh
```

### 3. Run the Experiment

From the tensorboard directory, run:

```bash
cd examples/advanced/experiment-tracking/tensorboard
python3 job.py
```

### 4. Access the Logs and Results

The simulator workspace is defined in `job.py` as `/tmp/nvflare/jobs/workdir`.

After running, you'll find the TensorBoard event files at:

```bash
$ tree /tmp/nvflare/jobs/workdir/server/simulate_job/

/tmp/nvflare/jobs/workdir/server/simulate_job/
├── app_server
│   <... skip ...>
└── tb_events
    ├── site-1
    │   └── events.out.tfevents.1744857479.rtx.30497.0
    └── site-2
        └── events.out.tfevents.1744857479.rtx.30497.1
```

### 5. View TensorBoard Results

To view training metrics that are being streamed to the server, run:

```bash
tensorboard --logdir=/tmp/nvflare/jobs/workdir/server/simulate_job/tb_events
```

Then open your browser to `http://localhost:6006` to view the metrics.

**Note**: If the server is running on a remote machine, use port forwarding:
```bash
ssh -L 6006:127.0.0.1:6006 user@server_ip
```

---

## How It Works

This example demonstrates **server-side tracking** where all client metrics are centralized.

### Step 1: Logging Metrics (in `client.py`)

Your training script logs metrics using NVFlare's tracking API:

```python
from nvflare.client.tracking import SummaryWriter

# Create writer
summary_writer = SummaryWriter()

# Log metrics during training
summary_writer.add_scalar("train_loss", loss, global_step=epoch)
summary_writer.add_scalar("train_accuracy", accuracy, global_step=epoch)
```

This creates a **local event** (`analytix_log_stats`) on the **NVFlare Client** side.

### Step 2: Event Streaming

The `add_experiment_tracking()` utility automatically configures:

1. **`ConvertToFedEvent` widget** (deployed to NVFlare Clients)
   - Listens for local event: `analytix_log_stats`
   - Converts it to federated event: `fed.analytix_log_stats`
   - Sends to server

2. **`TBAnalyticsReceiver`** (deployed to NVFlare Server)
   - Listens for federated event: `fed.analytix_log_stats`
   - Writes metrics to TensorBoard files on the server

### Result

All metrics from all clients are aggregated into a **single centralized TensorBoard view** on the server!

---

## Add TensorBoard to Your Own Recipe

Adding TensorBoard to any Recipe is simple:

```python
from nvflare.recipe.utils import add_experiment_tracking

# After creating your recipe
add_experiment_tracking(recipe, "tensorboard")

# Optional: customize the folder
add_experiment_tracking(
    recipe,
    "tensorboard",
    tracking_config={"tb_folder": "my_custom_folder"}
)
```

You can also switch to other tracking systems by changing the tracking type:
- `"tensorboard"` - TensorBoard streaming
- `"mlflow"` - MLflow tracking
- `"wandb"` - Weights & Biases tracking

---

## Additional Resources

- [Experiment Tracking Guide](https://nvflare.readthedocs.io/en/main/programming_guide/experiment_tracking.html)
- [TensorBoard Streaming Details](https://nvflare.readthedocs.io/en/main/examples/tensorboard_streaming.html)
- [Recipe API Documentation](https://nvflare.readthedocs.io/en/main/user_guide/data_scientist_guide/job_recipe.html)
