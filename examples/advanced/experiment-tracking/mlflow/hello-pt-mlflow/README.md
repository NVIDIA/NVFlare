# Hello PyTorch with MLflow

Example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) to train an image classifier
using federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629)) and [PyTorch](https://pytorch.org/)
as the deep learning training framework with **MLflow experiment tracking**.

This example demonstrates the **Recipe API** for easily adding MLflow tracking to FL training jobs.

## Setup

### 1. Install requirements

Install additional requirements (if you already have a specific version of nvflare installed in your environment, you may want to remove nvflare in the requirements to avoid reinstalling nvflare):

```
python -m pip install -r requirements.txt
```
### 2. Download Data
Pre-download the CIFAR-10 dataset to avoid multiple sites downloading simultaneously.

```bash
cd examples/advanced/experiment-tracking
./prepare_data.sh
```


### 3. Run the experiment

Navigate to the example directory and run:

```bash
cd examples/advanced/experiment-tracking/mlflow/hello-pt-mlflow
python3 job.py
```

The Recipe API makes it simple:
```python
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe.utils import add_experiment_tracking

# Create training recipe
recipe = FedAvgRecipe(
    name="fedavg_mlflow",
    min_clients=2,
    num_rounds=5,
    initial_model=SimpleNetwork(),
    train_script="client.py",
)

# Add MLflow tracking
add_experiment_tracking(
    recipe,
    "mlflow",
    tracking_config={
        "tracking_uri": "file:///tmp/nvflare/jobs/workdir/server/simulate_job/mlruns",
        "kw_args": {
            "experiment_name": "nvflare-fedavg-experiment",
            "run_name": "nvflare-fedavg-with-mlflow",
        }
    }
)

recipe.run()
```

### 4. Access the logs and results

You can find the running logs and results inside the server's simulator's workspace in a directory named "simulate_job".

```WORKSPACE = "/tmp/nvflare/jobs/workdir"```

By default, MLflow will create an experiment log directory under a directory named "mlruns" in the simulator's workspace.
If you ran the simulator with "/tmp/nvflare/jobs/workdir" as the workspace, then you can launch the MLflow UI with:

```bash
$ tree /tmp/nvflare/jobs/workdir/server/simulate_job/mlruns/
```

```
mlflow ui --backend-store-uri /tmp/nvflare/jobs/workdir/server/simulate_job/mlruns/
```

### 5. MLflow Streaming

For the job `hello-pt-mlflow`, on the client side, the client code in `client.py` uses the generic tracking API:

```python
from nvflare.client.tracking import SummaryWriter

summary_writer = SummaryWriter()
summary_writer.add_scalar("train_accuracy", accuracy, global_step=epoch)
```

**Note**: The `SummaryWriter` works with any tracking backend (MLflow, TensorBoard, WandB). When you use `add_experiment_tracking(recipe, "mlflow")`, the metrics are automatically routed to MLflow.

The `SummaryWriter` sends information in events to the server through NVFlare events
of type `analytix_log_stats` for the server to write the data to the MLflow tracking server.

The `ConvertToFedEvent` widget turns the event `analytix_log_stats` into a fed event `fed.analytix_log_stats`,
which will be delivered to the server side.

On the server side, the `MLflowReceiver` is configured to process `fed.analytix_log_stats` events,
which writes received data from these events to the MLflow tracking server.

This allows for the server to be the only party that needs to deal with authentication for the MLflow tracking server, and the server
can buffer the events from many clients to better manage the load of requests to the tracking server.


### 6. How It Works

This example demonstrates **server-side tracking** where all client metrics are centralized.

#### Step 1: Logging Metrics (in `client.py`)

Your training script logs metrics using NVFlare's tracking API:

```python
from nvflare.client.tracking import SummaryWriter

summary_writer = SummaryWriter()
summary_writer.add_scalar("train_loss", loss, global_step=epoch)
```

This creates a **local event** (`analytix_log_stats`) on the **NVFlare Client** side.

#### Step 2: Event Streaming

The `add_experiment_tracking()` utility automatically configures:

1. **`ConvertToFedEvent` widget** (deployed to NVFlare Clients)
   - Listens for local event: `analytix_log_stats`
   - Converts it to federated event: `fed.analytix_log_stats`
   - Sends to server

2. **`MLflowReceiver`** (deployed to NVFlare Server)
   - Listens for federated event: `fed.analytix_log_stats`
   - Writes metrics to MLflow tracking server

#### Result

All metrics from all clients are aggregated into a **single centralized MLflow experiment** on the server!
