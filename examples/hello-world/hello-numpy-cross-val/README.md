# Hello Numpy Cross-Site Evaluation

The cross-site model evaluation workflow uses data from clients to evaluate models from other clients. Data is not shared between sites. Instead, the collection of models is distributed to each client site for local evaluation. The server collects these results to construct an all-to-all matrix of model performance across client datasets.

This example demonstrates cross-site evaluation with NumPy models using both the Recipe API and FedJob API.

## Installation

Follow the [Installation](../../getting_started/README.md) instructions.

## What is Cross-Site Evaluation?

Cross-site evaluation creates an all-to-all matrix showing how each model performs on each client's dataset:
- Each client evaluates models from other clients and the server
- No data is shared between sites
- Results show which models generalize best across different data distributions

## Running Cross-Site Evaluation

The `job.py` script supports two modes:

### Mode 1: Standalone CSE with Pre-trained Models (Recommended)

This approach evaluates pre-trained models without running training first.

#### Step 1: Generate Pre-trained Models

First, create some pre-trained models to evaluate:

```bash
python3 generate_pretrain_models.py
```

This creates models in:
- Server models: `/tmp/nvflare/server_pretrain_models/`
- Client models: `/tmp/nvflare/client_pretrain_models/`

#### Step 2: Run Cross-Site Evaluation

```bash
python3 job.py --mode pretrained
```

This uses the `NumpyCrossSiteEvalRecipe` to:
- Load pre-trained models from specified directories
- Distribute models to all clients for evaluation
- Collect results and generate an all-to-all evaluation matrix

#### Step 3: View Results

The cross-site evaluation results are saved as JSON:

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

### Mode 2: Training + Cross-Site Evaluation

Run FedAvg training followed by cross-site evaluation in a single workflow:

```bash
python3 job.py --mode training --num_rounds 2
```

This performs:
1. **Training Phase**: FedAvg training for specified rounds (default: 1)
2. **Evaluation Phase**: Cross-site evaluation of the trained models

The results will be in `/tmp/nvflare/jobs/workdir/server/simulate_job/`.


## Understanding the Implementation

### Mode 1: Standalone CSE (Recipe API)

Uses `NumpyCrossSiteEvalRecipe` for clean, high-level API:

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

**Key Parameters:**
- `name`: Job name
- `min_clients`: Minimum clients required
- `model_locator_config`: Configuration for finding server models
  - `model_dir`: Directory containing server models
  - `model_name`: Dict mapping model names to file names
- `client_model_dir`: Directory where client models are stored

### Mode 2: Training + CSE (Recipe API)

Uses `FedAvgWithCrossSiteEvalRecipe` to combine training and evaluation:

```python
from nvflare.app_common.np.recipes import FedAvgWithCrossSiteEvalRecipe

recipe = FedAvgWithCrossSiteEvalRecipe(
    name="hello-numpy-train-cse",
    min_clients=2,
    num_rounds=1,
    train_script="client.py",
)
```

This recipe runs two workflows sequentially:
1. `ScatterAndGather` controller for FedAvg training
2. `CrossSiteModelEval` controller for evaluation

**Key Parameters:**
- `name`: Job name
- `min_clients`: Minimum clients required
- `num_rounds`: Number of training rounds
- `train_script`: Path to training script
- `train_args`: Arguments to pass to training script (optional)
- Other CSE parameters: `cross_val_dir`, `submit_model_timeout`, `validation_timeout`

## Files Overview

- `job.py`: Main script supporting both modes
  - Mode 1: Uses `NumpyCrossSiteEvalRecipe`
  - Mode 2: Uses `FedAvgWithCrossSiteEvalRecipe`
- `client.py`: Training script for Mode 2 (training+CSE)
- `generate_pretrain_models.py`: Utility to create pre-trained models for Mode 1
- `README.md`: This file

### Command-Line Options

```bash
python job.py --help
```

Options:
- `--mode`: Choose `pretrained` (default) or `training`
- `--n_clients`: Number of clients (default: 2)
- `--num_rounds`: Training rounds for training mode (default: 1)

## Next Steps

- Try modifying the models in `generate_pretrain_models.py`
- Experiment with different numbers of clients using `--n_clients`
- Test different training configurations with `--num_rounds` in training mode
- See [Cross-Site Model Evaluation documentation](https://nvflare.readthedocs.io/en/main/programming_guide/controllers/cross_site_model_evaluation.html) for more details
