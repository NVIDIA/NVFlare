# Hello NumPy Cross-Site Validation

This example demonstrates cross-site model validation with NumPy models using NVFlare's Recipe API.

## What is Cross-Site Validation?

Cross-site validation creates a matrix showing how each model performs on each client's dataset:
- Server provides models to all clients for evaluation
- Each client evaluates the models on its local data
- No data is shared between sites
- Results show which models generalize best across different data distributions

The workflow uses the [CrossSiteModelEval](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.workflows.cross_site_model_eval.html) controller workflow.

## Installation

Follow the [Installation](../../getting_started/README.md) instructions.

## Two Modes of Operation

This example supports two workflows using a unified `job.py` script:

### Mode 1: Standalone CSE with Pre-trained Models

Evaluate pre-trained models without running training first. This is useful when you have models from a previous training session or external sources.

### Mode 2: Training + CSE

Run FedAvg training followed by cross-site validation in a single workflow.

---

## Mode 1: Standalone CSE with Pre-trained Models

### Step 1: Generate Pre-trained Models

First, create some pre-trained models to evaluate:

```bash
cd examples/hello-world/hello-numpy-cross-val
python3 generate_pretrain_models.py
```

This creates server-side models in `/tmp/nvflare/server_pretrain_models/`

### Step 2: Run Cross-Site Validation

```bash
python3 job.py --mode pretrained --n_clients 2
```

**What happens:**
1. Server loads pre-trained models from specified directories
2. Models are distributed to all clients
3. Each client validates all models on its local data
4. Server collects results and generates validation matrix

### Step 3: View Results

The cross-site validation results are saved as JSON:

```bash
cat /tmp/nvflare/jobs/workdir/server/simulate_job/cross_site_val/cross_val_results.json
```

**Example output:**

```json
{
  "site-1": {
    "server_model_1": {"accuracy": 0.95, "mse": 0.023},
    "server_model_2": {"accuracy": 0.93, "mse": 0.028},
    "site-2": {"accuracy": 0.91, "mse": 0.031}
  },
  "site-2": {
    "server_model_1": {"accuracy": 0.94, "mse": 0.025},
    "server_model_2": {"accuracy": 0.92, "mse": 0.029},
    "site-1": {"accuracy": 0.90, "mse": 0.033}
  }
}
```

This matrix shows how each model performs on each site's data, helping identify which models generalize best.

---

## Mode 2: Training + Cross-Site Validation

Run FedAvg training followed by cross-site validation:

```bash
cd examples/hello-world/hello-numpy-cross-val
python3 job.py --mode training --n_clients 2 --num_rounds 3
```

**What happens:**
1. FedAvg training runs for the specified number of rounds
2. After training completes, trained models are automatically validated across all sites
3. Results include both training metrics and cross-site validation matrix

### View Results

Training results:

```bash
ls /tmp/nvflare/jobs/workdir/server/simulate_job/
```

Cross-site validation results:

```bash
cat /tmp/nvflare/jobs/workdir/server/simulate_job/cross_site_val/cross_val_results.json
```

---

## How It Works: Recipe API Approach

This example demonstrates the recommended pattern for adding cross-site validation to any recipe:

### Training + CSE Mode

```python
from nvflare.app_common.np.recipes import NumpyFedAvgRecipe
from nvflare.recipe.utils import add_cross_site_evaluation
from nvflare.recipe import SimEnv

# 1. Create a standard FedAvg recipe
recipe = NumpyFedAvgRecipe(
    name="hello-numpy-train-cse",
    min_clients=2,
    num_rounds=3,
    train_script="client.py",
)

# 2. Add cross-site validation with one line
add_cross_site_evaluation(recipe)

# 3. Execute
env = SimEnv(num_clients=2)
run = recipe.execute(env)
```

**Key benefits:**
- Works with **any** recipe (FedAvg, Cyclic, custom recipes)
- Same pattern for PyTorch, NumPy, and other frameworks
- Consistent with `add_experiment_tracking()` utility

### Standalone CSE Mode

For evaluating pre-trained models without training, you configure a minimal FedJob with:
- `NPModelLocator` pointing to pre-trained model directories
- `CrossSiteModelEval` controller
- `NPValidator` on clients

See `job.py` for the complete implementation.

---

## Key Files

- `job.py` - Unified script supporting both modes using Recipe API
- `client.py` - NumPy training script (used in training mode)
- `generate_pretrain_models.py` - Creates pre-trained models for standalone CSE

---

## Customization

### Using Different Model Locations

For standalone CSE, modify the server model directory in `job.py`:

```python
SERVER_MODEL_DIR = "/path/to/your/server/models"
```

The server will distribute these models to all clients for evaluation.

### Adding Custom Validation Metrics

Modify `client.py` to compute additional metrics in the validation function:

```python
def validate(model_params, data):
    # Your validation logic here
    return {
        "accuracy": accuracy,
        "mse": mse,
        "custom_metric": custom_value,
    }
```

### Using with PyTorch Models

Replace `NumpyFedAvgRecipe` with `FedAvgRecipe` and change the model locator:

```python
from nvflare.app_opt.pt.recipes import FedAvgRecipe
from nvflare.recipe.utils import add_cross_site_evaluation

recipe = FedAvgRecipe(
    name="hello-pt-cse",
    min_clients=2,
    num_rounds=3,
    train_script="client.py",
    initial_model=YourModel(),
)

add_cross_site_evaluation(recipe)
```

---

## Advanced: Running in POC or Production

### POC Environment

```python
from nvflare.recipe import PocEnv

recipe = NumpyFedAvgRecipe(...)
add_cross_site_evaluation(recipe)

env = PocEnv(num_clients=2)
run = recipe.execute(env)
```

### Production Environment

**Option 1: Execute directly (programmatic submission)**
```python
from nvflare.recipe import ProdEnv

recipe = NumpyFedAvgRecipe(...)
add_cross_site_evaluation(recipe)

env = ProdEnv(startup_kit_location="/path/to/admin/startup/kit")
run = recipe.execute(env)  # Submits and runs the job
```

**Option 2: Export for manual submission**
```python
from nvflare.recipe import ProdEnv

recipe = NumpyFedAvgRecipe(...)
add_cross_site_evaluation(recipe)

env = ProdEnv(startup_kit_location="/path/to/admin/startup/kit")
recipe.export(job_dir="/tmp/nvflare/prod/job_config", env=env)  # Creates job files only
# Then use 'nvflare job submit' command to submit manually
```

---

## Next Steps

- Try the [PyTorch CSE example](../hello-pt) for deep learning models
- Learn about [experiment tracking](../../advanced/experiment-tracking) with TensorBoard, MLflow, or Weights & Biases
- Explore [custom recipes](https://nvflare.readthedocs.io/en/main/user_guide/data_scientist_guide/job_recipe.html) in the documentation
