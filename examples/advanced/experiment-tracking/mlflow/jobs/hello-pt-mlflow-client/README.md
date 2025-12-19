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
from nvflare.app_opt.pt.recipes import FedAvgRecipe
from nvflare.app_opt.tracking.mlflow.mlflow_receiver import MLflowReceiver
from nvflare.apis.analytix import ANALYTIC_EVENT_TYPE

# Create recipe WITHOUT default server-side tracking
recipe = FedAvgRecipe(
    name="fedavg_mlflow_client",
    min_clients=2,
    num_rounds=5,
    initial_model=SimpleNetwork(),
        train_script="src/client.py",
    analytics_receiver=False,  # Disable server-side tracking
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
- `analytics_receiver=False` - Disables default server-side tracking
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
cd jobs/hello-pt-mlflow-client/code
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

### Client-Side Tracking Flow

1. **Training Script** (`src/client.py`):
   ```python
   from nvflare.client.tracking import MLflowWriter

   mlflow = MLflowWriter()
   mlflow.log_metric("accuracy", acc, step=epoch)
   ```

2. **Local Event Generated**:
   - Creates `analytix_log_stats` event (not federated)

3. **MLflowReceiver on Client**:
   - Configured with `events=[ANALYTIC_EVENT_TYPE]`
   - Receives local events directly
   - Writes to site-specific MLflow directory

4. **No Server Involvement**:
   - Metrics stay local to each site
   - No centralized aggregation
   - Each site controls its own data

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
- [Recipe API Documentation](https://nvflare.readthedocs.io/en/main/programming_guide/job_recipes.html)
