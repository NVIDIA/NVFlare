# Hello PyTorch Lightning with MLflow

Example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) to train an image classifier
using federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629)) and **[PyTorch Lightning](https://lightning.ai/)**
as the deep learning training framework with **MLflow experiment tracking**.

> **Minimum Hardware Requirements**: 1 CPU or 1 GPU

## Overview

This example demonstrates PyTorch Lightning integration with MLflow tracking:

```python
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe.utils import add_experiment_tracking

# Create FedAvg recipe for Lightning model
recipe = FedAvgRecipe(
    name="fedavg_lightning_mlflow",
    min_clients=2,
    num_rounds=2,
    initial_model=LitNet(),  # Lightning model
    train_script="client.py",
)

# Add MLflow tracking with one line!
add_experiment_tracking(
    recipe=recipe,
    tracking_type="mlflow",
    tracking_config={
        "tracking_uri": "file:///tmp/nvflare/jobs/workdir/server/simulate_job/mlruns",
        "kw_args": {
            "experiment_name": "lightning-fedavg-experiment",
            "run_name": "lightning-run",
        }
    }
)

recipe.run()
```

**Note:** This example uses the standard PyTorch `FedAvgRecipe` which automatically handles Lightning models. The recipe integrates with Lightning's callback system and trainer, while maintaining the same simple tracking API as regular PyTorch examples.

## Setup and Running

### 1. Install Requirements

```bash
cd examples/advanced/experiment-tracking/mlflow/hello-lightning-mlflow
python -m pip install -r ../requirements.txt
```

### 2. Download Data

```bash
cd examples/advanced/experiment-tracking
./prepare_data.sh
```

### 3. Run the Experiment

```bash
cd examples/advanced/experiment-tracking/mlflow/hello-lightning-mlflow
python job.py
```

Optional arguments:
```bash
python job.py -n 3 -t "file:///my/custom/mlruns" -l verbose
```

- `-n, --n_clients`: Number of clients (default: 2)
- `-t, --tracking_uri`: MLflow tracking URI
- `-w, --work_dir`: Working directory
- `-e, --export_config`: Export config only (don't run)
- `-l, --log_config`: Log level (concise/verbose)

---

## Accessing Results

### View Logs

```bash
tree /tmp/nvflare/jobs/workdir/server/simulate_job/mlruns/
```

### Launch MLflow UI

```bash
mlflow ui --backend-store-uri /tmp/nvflare/jobs/workdir/server/simulate_job/mlruns/
```

Then open your browser to `http://localhost:5000`

---

## How Lightning Integration Works

### 1. Client-Side Logging

In your Lightning model (`model.py`), use standard Lightning logging:

```python
class LitNet(LightningModule):
    def training_step(self, batch, batch_idx):
        loss = ...
        # Use Lightning's built-in logging
        self.log("train_loss", loss)
        return loss
```

In your training script (`client.py`), use the NVFlare Lightning logger:

```python
import nvflare.client.lightning as flare

flare_logger = flare.logger()  # Creates analytics events from Lightning logs
trainer = Trainer(logger=flare_logger)
```

### 2. Event Flow

1. **Lightning trains** → Calls `self.log()` in model
2. **`flare.logger()`** → Captures logs and creates `analytix_log_stats` events
3. **ConvertToFedEvent** (automatic) → Converts to `fed.analytix_log_stats`
4. **Server-side MLflowReceiver** → Writes to MLflow tracking server

### Lightning Callbacks

Lightning callbacks work normally:

```python
from pytorch_lightning.callbacks import ModelCheckpoint

trainer = Trainer(
    callbacks=[
        ModelCheckpoint(monitor='val_loss'),
    ],
)
```

---

## Comparison: PyTorch vs Lightning

| Aspect | PyTorch (vanilla) | PyTorch Lightning (This Example) |
|--------|-------------------|----------------------------------|
| **Recipe Import** | `nvflare.app_opt.pt.recipes` | `nvflare.app_opt.pt.recipes` (same) |
| **Model Type** | `torch.nn.Module` | `LightningModule` |
| **Training Loop** | Custom script | Lightning Trainer |
| **Logging** | `MLflowWriter` | `MLflowWriter` or `ClientLogger` |
| **Callbacks** | Manual | Lightning callbacks |

**Both use the same `add_experiment_tracking()` API!**

---

## Customization

### Change Experiment Details

```python
add_experiment_tracking(
    recipe=recipe,
    tracking_type="mlflow",
    tracking_config={
        "tracking_uri": "http://my-mlflow-server:5000",
        "kw_args": {
            "experiment_name": "my-lightning-experiment",
            "run_name": "run-001",
            "experiment_tags": {
                "framework": "lightning",
                "architecture": "CNN",
            },
            "run_tags": {
                "optimizer": "Adam",
                "lr": "0.001",
            }
        }
    }
)
```

## Additional Resources

- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)
- [Experiment Tracking Guide](https://nvflare.readthedocs.io/en/main/programming_guide/experiment_tracking.html)
- [Recipe API Documentation](https://nvflare.readthedocs.io/en/main/user_guide/data_scientist_guide/job_recipe.html)
