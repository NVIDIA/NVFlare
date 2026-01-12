# Hello PyTorch with MLflow: Client-Side Metric Streaming

This example demonstrates **site-specific (decentralized) MLflow tracking** using the Recipe API. Each client tracks its own metrics locally, enabling privacy-preserving metric collection without centralized server aggregation.

## Key Differences from Server-Side Tracking

**Server-Side** (`hello-pt-mlflow`):
- All client metrics stream to the server
- Centralized view in one MLflow instance
- Server handles MLflow authentication

**Client-Side** (this example):
- Each client has its own MLflow instance
- Site-specific metric tracking
- Each site manages its own MLflow server

## Overview

This example shows how to configure per-client MLflow tracking:

```python
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.app_opt.tracking.mlflow.mlflow_receiver import MLflowReceiver
from nvflare.apis.analytix import ANALYTIC_EVENT_TYPE

# Create recipe WITHOUT default server-side tracking
recipe = FedAvgRecipe(
    name="fedavg_mlflow_client",
    min_clients=2,
    num_rounds=5,
    initial_model=SimpleNetwork(),
    train_script="client.py",
)

# Add MLflow receiver to each client
for i in range(2):
    site_name = f"site-{i + 1}"
    tracking_uri = f"file:///tmp/nvflare/jobs/workdir/{site_name}/mlruns"

    receiver = MLflowReceiver(
        tracking_uri=tracking_uri,
        events=[ANALYTIC_EVENT_TYPE],  # Listen to LOCAL events (not federated)
        kw_args={"experiment_name": f"site-{site_name}-experiment"}
    )

    recipe.job.to(receiver, site_name, id="mlflow_receiver")

recipe.run()
```

**Key points**:
- By default, no server-side tracking is enabled (analytics_receiver defaults to None)
- `events=[ANALYTIC_EVENT_TYPE]` - Listen to local events (not `fed.` events)
- Each site gets its own `tracking_uri`

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

### View Site-1 Metrics:
```bash
mlflow ui --backend-store-uri /tmp/nvflare/jobs/workdir/site-1/mlruns
```

Open browser to `http://localhost:5000`

### View Site-2 Metrics:
```bash
mlflow ui --backend-store-uri /tmp/nvflare/jobs/workdir/site-2/mlruns --port 5001
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
tracking_uri = f"file:///my/custom/path/{site_name}/mlruns"
```

### Add Experiment Tags

```python
receiver = MLflowReceiver(
    tracking_uri=tracking_uri,
    events=[ANALYTIC_EVENT_TYPE],
    kw_args={
        "experiment_name": f"{site_name}-experiment",
        "run_name": f"{site_name}-run-001",
        "experiment_tags": {"site": site_name, "privacy": "local"},
    }
)
```

### Remote MLflow Server

```python
tracking_uri = "http://mlflow-server.site-1.local:5000"
```

---

## Additional Resources

- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [Experiment Tracking Guide](https://nvflare.readthedocs.io/en/main/programming_guide/experiment_tracking.html)
- [Recipe API Documentation](https://nvflare.readthedocs.io/en/main/user_guide/data_scientist_guide/job_recipe.html)
