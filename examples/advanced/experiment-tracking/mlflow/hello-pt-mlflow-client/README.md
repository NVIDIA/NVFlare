# Hello PyTorch with MLflow: Client-Side Metric Streaming

This example demonstrates **site-specific (decentralized) MLflow tracking** using the Recipe API. Each client tracks its own metrics locally, enabling privacy-preserving metric collection without centralized server aggregation.

## Key Differences from Server-Side Tracking

**Server-Side** (`hello-pt-mlflow`):
- All client metrics stream to the server
- Centralized view in one MLflow instance
- Server handles MLflow authentication

**Client-Side** (this example):
- Each client has its own local MLflow store
- Site-specific metric tracking
- Each site can optionally point at its own MLflow server

## Overview

This example shows how to configure per-client MLflow tracking:

```python
from model import SimpleNetwork

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking

# Create recipe WITHOUT default server-side tracking
recipe = FedAvgRecipe(
    name="fedavg_mlflow_client",
    min_clients=2,
    num_rounds=5,
    # Model can be class instance or dict config:
    model=SimpleNetwork(),
    # Alternative: model={"class_path": "model.SimpleNetwork", "args": {}},
    # For pre-trained weights: initial_ckpt="/server/path/to/pretrained.pt",
    train_script="client.py",
)

# Add an MLflow receiver to every client, but not to the server.
# With tracking_uri=None, each receiver creates a store in its local job workspace.
add_experiment_tracking(
    recipe,
    "mlflow",
    tracking_config={
        "tracking_uri": None,
        "kw_args": {
            "experiment_name": "nvflare-fedavg-experiment",
            "run_name": "nvflare-fedavg-client",
        },
    },
    client_side=True,
    server_side=False,
)

env = SimEnv(num_clients=2)
run = recipe.execute(env)
```

**Key points**:
- `client_side=True, server_side=False` keeps metric handling on the clients.
- The helper configures receivers to listen to local events rather than federated `fed.` events.
- `tracking_uri=None` resolves to a separate MLflow store in each client's job workspace.
- Sites can use explicit local paths or remote MLflow servers when their deployment requires them.

## Setup

### 1. Install Requirements

```bash
cd examples/advanced/experiment-tracking/mlflow
python -m pip install -r requirements.txt
```

### 2. Download Data

```bash
cd examples/advanced/experiment-tracking
./prepare_data.sh
```

### 3. Run the Experiment

```bash
cd examples/advanced/experiment-tracking/mlflow/hello-pt-mlflow-client
python job.py
```

---

## Accessing Results

Since each site has its own MLflow receiver, metrics are stored separately:

With the example's `tracking_uri=None`, the store is created under each site's
job-result directory as `<site-workspace>/<job-id>/mlflow`. Under the default
simulation workspace, use `find` to locate the generated directory:

### View Site-1 Metrics:
```bash
find /tmp/nvflare/jobs/workdir/fedavg_mlflow_client/site-1 -type d -name mlflow
mlflow ui --backend-store-uri <site-1-mlflow-directory>
```

Open browser to `http://localhost:5000`

### View Site-2 Metrics:
```bash
find /tmp/nvflare/jobs/workdir/fedavg_mlflow_client/site-2 -type d -name mlflow
mlflow ui --backend-store-uri <site-2-mlflow-directory> --port 5001
```

Open browser to `http://localhost:5001`

---

## How It Works

This example demonstrates **client-side tracking** where each client keeps its own metrics locally.

### Step 1: Logging Metrics (in `client.py`)

Your training script logs metrics using NVFlare's tracking API:

```python
from nvflare.client.tracking import MLflowWriter

mlflow = MLflowWriter()
mlflow.log_metric("accuracy", acc, step=epoch)
```

This creates a **local event** (`analytix_log_stats`) on the **NVFlare Client** side.

### Step 2: Local Event Handling

Unlike server-side tracking, there is **no event conversion** to federated events:

1. **`MLflowReceiver`** (deployed to each NVFlare Client)
   - Listens for **local event**: `analytix_log_stats` (NOT `fed.analytix_log_stats`)
   - Writes metrics to **local MLflow directory** on that client
   - Metrics **never leave the client**

2. **No `ConvertToFedEvent` widget**
   - Local events stay local
   - Server never receives metrics

### Result

Each client has its own **separate MLflow instance** with only its own metrics. Complete data privacy!

### Key Terminology

To avoid confusion:
- **`client.py`**: Your training script (user code that logs metrics)
- **NVFlare Client**: The FL client runtime that executes your training script
- **Client-Side Tracking**: Receiver deployed on NVFlare Client (this example)
- **Local Event**: `analytix_log_stats` - stays on the client
- **Federated Event**: `fed.analytix_log_stats` - sent to server (NOT used in this example)

---

## Comparison: Server vs Client Tracking

| Aspect | Server-Side | Client-Side (This Example) |
|--------|-------------|----------------------------|
| **Metrics Location** | Centralized on server | Distributed per site |
| **Event Type** | `fed.analytix_log_stats` | `analytix_log_stats` |
| **Receiver Location** | Server | Each client |
| **MLflow UI** | Single instance | One per site |
| **Privacy** | Server sees all metrics | Sites keep metrics local |
| **Use Case** | Centralized monitoring | Site-specific analysis |

---

## Customization

### Change Tracking Location

```python
add_experiment_tracking(
    recipe,
    "mlflow",
    tracking_config={"tracking_uri": "file:///my/custom/path/mlruns"},
    client_side=True,
    server_side=False,
)
```

In a distributed deployment, the same local path is resolved independently on
each site. In simulation, all clients share the host filesystem, so use the
default `tracking_uri=None` behavior or configure distinct paths if the stores
must remain physically separate.

For different configuration at each site, construct the recipe with per-site
client apps and then target each site explicitly:

```python
sites = ["site-1", "site-2"]
recipe = FedAvgRecipe(
    ...,
    per_site_config={site: {} for site in sites},
)

for site in sites:
    add_experiment_tracking(
        recipe,
        "mlflow",
        tracking_config={
            "tracking_uri": f"file:///my/custom/path/{site}/mlruns",
            "kw_args": {"experiment_name": f"{site}-experiment"},
        },
        client_side=True,
        server_side=False,
        clients=[site],
    )
```

Targeted `clients=[...]` placement requires `per_site_config` when the recipe is
constructed; it cannot split an existing `@ALL` client app after the fact.

### Add Experiment Tags

```python
add_experiment_tracking(
    recipe,
    "mlflow",
    tracking_config={
        "tracking_uri": None,
        "kw_args": {
            "experiment_name": "local-client-experiment",
            "run_name": "client-run-001",
            "experiment_tags": {"privacy": "local"},
        },
    },
    client_side=True,
    server_side=False,
)
```

### Remote MLflow Server

```python
add_experiment_tracking(
    recipe,
    "mlflow",
    tracking_config={"tracking_uri": "http://mlflow-server.local:5000"},
    client_side=True,
    server_side=False,
)
```

If each site uses a different server or credentials, use the per-site pattern
above and pass the appropriate configuration with `clients=[site]`.

---

## Additional Resources

- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [Experiment Tracking Guide](https://nvflare.readthedocs.io/en/main/programming_guide/experiment_tracking.html)
- [Recipe API Documentation](https://nvflare.readthedocs.io/en/main/user_guide/data_scientist_guide/job_recipe.html)
