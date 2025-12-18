# Hello PyTorch Lightning with MLflow

Example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) to train an image classifier
using federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629)) and **[PyTorch Lightning](https://lightning.ai/)**
as the deep learning training framework with **MLflow experiment tracking**.

> **Minimum Hardware Requirements**: 1 CPU or 1 GPU

## What's New: Recipe API + Lightning

This example demonstrates the **Lightning-specific Recipe API** for PyTorch Lightning integration:

```python
from nvflare.app_opt.lightning.recipes import FedAvgRecipe  # Lightning Recipe!
from nvflare.recipe.utils import add_experiment_tracking

# Create Lightning FedAvg recipe
recipe = FedAvgRecipe(
    name="fedavg_lightning_mlflow",
    min_clients=2,
    num_rounds=2,
    initial_model=LitNet(),  # Lightning model
    train_script="src/client.py",
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

**Key Features**:
- Uses Lightning-specific `FedAvgRecipe` from `nvflare.app_opt.lightning.recipes`
- Automatically handles Lightning model serialization
- Integrates with Lightning's callback system
- Same simple tracking API as PyTorch examples

---

## Setup and Running

### 1. Install Requirements

```bash
cd examples/advanced/experiment-tracking/mlflow/jobs/hello-lightning-mlflow
python -m pip install -r requirements.txt
```

### 2. Download Data

```bash
cd examples/advanced/experiment-tracking
./prepare_data.sh
```

### 3. Run the Experiment

```bash
cd jobs/hello-lightning-mlflow/code
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

In your Lightning training script (`src/client.py`):

```python
from nvflare.client.tracking import MLflowWriter

class LitNet(LightningModule):
    def __init__(self):
        super().__init__()
        self.mlflow_writer = MLflowWriter()

    def training_step(self, batch, batch_idx):
        loss = ...
        # Log metrics
        self.mlflow_writer.log_metric("train_loss", loss, step=self.global_step)
        return loss
```

### 2. Event Flow

1. **Lightning trains** → Model updates via callbacks
2. **MLflowWriter logs metrics** → Creates `analytix_log_stats` events
3. **ConvertToFedEvent** (automatic) → Converts to `fed.analytix_log_stats`
4. **Server-side MLflowReceiver** → Writes to MLflow tracking server

### 3. Automatic Components

The Recipe API automatically configures:
- ✅ Lightning-compatible model serialization
- ✅ Event conversion widgets
- ✅ MLflow receiver on server
- ✅ Proper callback integration

---

## Lightning-Specific Features

### Using Lightning Logger

You can also use NVFlare's Lightning logger directly:

```python
from nvflare.app_opt.lightning.loggers import ClientLogger

trainer = Trainer(
    logger=ClientLogger(),
    max_epochs=1,
)
```

This integrates seamlessly with the MLflow tracking.

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

| Aspect | PyTorch Recipe | Lightning Recipe (This Example) |
|--------|----------------|--------------------------------|
| **Recipe Import** | `nvflare.app_opt.pt.recipes` | `nvflare.app_opt.lightning.recipes` |
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

### Switch to TensorBoard

Simply change the tracking type:

```python
add_experiment_tracking(recipe, "tensorboard")
```

---

## Additional Resources

- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)
- [NVFlare Lightning Integration](https://nvflare.readthedocs.io/en/main/programming_guide/lightning_integration.html)
- [Experiment Tracking Guide](https://nvflare.readthedocs.io/en/main/programming_guide/experiment_tracking.html)
- [Recipe API Documentation](https://nvflare.readthedocs.io/en/main/programming_guide/job_recipes.html)
