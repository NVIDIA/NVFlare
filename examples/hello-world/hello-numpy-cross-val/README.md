# Hello Numpy Cross-Site Validation

The cross-site model evaluation workflow uses the data from clients to run evaluation with the models of other clients. Data is not shared. Rather the collection of models is distributed to each client site to run local validation. The server collects the results of local validation to construct an all-to-all matrix of model performance vs. client dataset.

This example demonstrates how to use the **Recipe API** for cross-site evaluation with NumPy models.

## Installation

Follow the [Installation](../../getting_started/README.md) instructions.

## What is Cross-Site Evaluation?

Cross-site evaluation creates an all-to-all matrix showing how each model performs on each client's dataset:
- Each client evaluates models from other clients and the server
- No data is shared between sites
- Results show which models generalize best across different data distributions

## Running Cross-Site Evaluation

### Option 1: Standalone CSE with Pre-trained Models (Recommended)

This approach evaluates pre-trained models without running training first.

#### 1. Generate Pre-trained Models

First, create some pre-trained models to evaluate:

```bash
python3 generate_pretrain_models.py
```

This creates models in:
- Server models: `/tmp/nvflare/server_pretrain_models/`
- Client models: `/tmp/nvflare/client_pretrain_models/`

#### 2. Run Cross-Site Evaluation Using Recipe API

```bash
python3 job.py
```

This uses the `NumpyCrossSiteEvalRecipe` to:
- Load pre-trained models from specified directories
- Distribute models to all clients for evaluation
- Collect results and generate an all-to-all evaluation matrix

#### 3. View Results

The cross-site validation results are saved as JSON:

```bash
cat /tmp/nvflare/jobs/workdir/server/simulate_job/cross_site_val/cross_val_results.json
```

The JSON shows how each model performs on each client's data:
```json
{
  "site-1": {
    "server_model_1": {"accuracy": 0.95},
    "server_model_2": {"accuracy": 0.93},
    "site-2": {"accuracy": 0.91}
  },
  "site-2": {
    "server_model_1": {"accuracy": 0.94},
    "server_model_2": {"accuracy": 0.92},
    "site-1": {"accuracy": 0.90}
  }
}
```

### Option 2: Training + Cross-Site Evaluation

Run FedAvg training followed by cross-site evaluation:

```bash
python3 job_train_and_cse.py
```

This performs:
1. **Training Phase**: FedAvg training for 1 round
2. **Evaluation Phase**: Cross-site evaluation of the trained models

> **Note**: This uses the legacy `FedJob` API. For the modern Recipe API approach, see Option 1.

## Understanding the Recipe API

The new `job.py` uses `NumpyCrossSiteEvalRecipe`:

```python
from nvflare.app_common.np.recipes.cross_site_eval import NumpyCrossSiteEvalRecipe

recipe = NumpyCrossSiteEvalRecipe(
    name="hello-numpy-cse",
    min_clients=2,
    model_locator_config={
        "model_dir": "/tmp/nvflare/server_pretrain_models",
        "model_name": {
            "server_model_1": "server_1.npy",
            "server_model_2": "server_2.npy"
        }
    },
    client_model_dir="/tmp/nvflare/client_pretrain_models",
)
```

### Key Parameters

- `name`: Job name
- `min_clients`: Minimum clients required
- `model_locator_config`: Configuration for finding server models
  - `model_dir`: Directory containing server models
  - `model_name`: Dict mapping model names to file names
- `client_model_dir`: Directory where client models are stored
- `server_models`: List of server model names to evaluate (optional)

## Files Overview

- `job.py`: **NEW** - Recipe API approach for standalone CSE
- `job_cse.py`: Legacy FedJob API for standalone CSE
- `job_train_and_cse.py`: Legacy FedJob API for training + CSE
- `generate_pretrain_models.py`: Utility to create pre-trained models
- `README.md`: This file

## Next Steps

- Try modifying the models in `generate_pretrain_models.py`
- Add more clients by changing `--n_clients` parameter
- Explore combining this with training workflows
- See [Cross-Site Model Evaluation docs](https://nvflare.readthedocs.io/en/main/programming_guide/controllers/cross_site_model_evaluation.html) for more details
