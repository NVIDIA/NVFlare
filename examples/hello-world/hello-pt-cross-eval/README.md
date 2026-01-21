# Hello PyTorch Cross-Site Evaluation

This example demonstrates **standalone cross-site evaluation** with pre-trained PyTorch models using NVFlare's Recipe API.

**Note**: For Training + Cross-Site Evaluation, see the [hello-pt example](../hello-pt).

## What is Cross-Site Evaluation (CSE)?

Cross-site evaluation creates a matrix showing how each model performs on each client's dataset:
- Server provides models to all clients for evaluation
- Each client evaluates the models on its local data
- No data is shared between sites
- Results show which models generalize best across different data distributions

The workflow uses the [CrossSiteModelEval](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.workflows.cross_site_model_eval.html) controller workflow.

## Use Case

This example is useful when you:
- Have pre-trained models from previous training sessions
- Want to evaluate external models on federated data
- Need to compare multiple model variants without retraining
- Want to validate model generalization across sites

## Installation

Follow the [Installation](../../getting_started/README.md) instructions.

Install PyTorch requirements:

```bash
pip install torch torchvision
```

---

## Running Standalone CSE

### Step 1: Prepare Data

Download CIFAR-10 dataset to avoid race conditions when multiple clients run simultaneously:

```bash
cd examples/hello-world/hello-pt-cross-eval
python3 prepare_data.py
```

This downloads CIFAR-10 to `/tmp/nvflare/data/cifar10/`

### Step 2: Generate Pre-trained Models

Create some pre-trained models to evaluate:

```bash
python3 generate_pretrain_models.py
```

This creates server-side models in `/tmp/nvflare/server_pretrain_models/`

### Step 3: Run Cross-Site Evaluation

```bash
python3 job.py --n_clients 2
```

**What happens:**
1. Server loads pre-trained models from specified directories
2. Models are distributed to all clients
3. Each client evaluates all models on its local data (CIFAR-10)
4. Server collects results and generates evaluation matrix

### Step 4: View Results

The cross-site evaluation results are saved as JSON:

```bash
cat /tmp/nvflare/jobs/workdir/server/simulate_job/cross_site_val/cross_val_results.json
```

**Example output:**

```json
{
  "site-1": {
    "SRV_server_model_1": {"accuracy": 15.2},
    "SRV_server_model_2": {"accuracy": 14.8},
    "site-2": {"accuracy": 16.1}
  },
  "site-2": {
    "SRV_server_model_1": {"accuracy": 15.5},
    "SRV_server_model_2": {"accuracy": 14.9},
    "site-1": {"accuracy": 15.8}
  }
}
```

This matrix shows how each model performs on each site's data, helping identify which models generalize best.

---

## How It Works: Recipe API Approach

This example demonstrates standalone CSE using `PyTorchCrossSiteEvalRecipe`:

```python
from nvflare.app_opt.pt.recipes import PyTorchCrossSiteEvalRecipe
from nvflare.recipe import SimEnv
from model import SimpleNetwork

# Create CSE recipe
recipe = PyTorchCrossSiteEvalRecipe(
    name="hello-pt-cse",
    min_clients=2,
    model=SimpleNetwork(),
    train_script="client.py",
)

# Execute
env = SimEnv(num_clients=2)
run = recipe.execute(env)
```

**For Training + CSE**: See the [hello-pt example](../hello-pt) which demonstrates adding CSE to a training workflow using `add_cross_site_evaluation()`.

---

## Key Files

- `job.py` - Main script for running standalone CSE using Recipe API
- `client.py` - PyTorch client script with CSE support via Client API
- `model.py` - SimpleNetwork model definition
- `prepare_data.py` - Downloads CIFAR-10 dataset (run first to avoid race conditions)
- `generate_pretrain_models.py` - Creates pre-trained models for standalone CSE

---

## Client Script Pattern for CSE

The key to PyTorch CSE is handling the `flare.is_evaluate()` check in your training script:

```python
import nvflare.client as flare

flare.init()

while flare.is_running():
    input_model = flare.receive()
    model.load_state_dict(input_model.params)
    
    # Evaluate model (always required)
    metrics = evaluate(model, test_loader)
    
    # Handle CSE evaluation task
    if flare.is_evaluate():
        output_model = flare.FLModel(metrics=metrics)
        flare.send(output_model)
        continue  # Skip training for validation-only tasks
    
    # Normal training code here...
    train(model, train_loader)
    output_model = flare.FLModel(params=model.state_dict(), metrics=metrics)
    flare.send(output_model)
```

---

## Customization

### Using Different Model Architectures

Modify `model.py` to define your custom PyTorch model:

```python
class CustomNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Your model architecture
        
    def forward(self, x):
        # Your forward pass
        return x
```

Then update `job.py` to use your model:

```python
from model import CustomNetwork

recipe = PyTorchCrossSiteEvalRecipe(
    name="hello-pt-cse",
    model=CustomNetwork(),
    train_script="client.py",
)
```

### Adding Custom Evaluation Metrics

Modify the `evaluate()` function in `client.py`:

```python
def evaluate(net, test_loader, device):
    # Your evaluation logic
    return {
        "accuracy": accuracy,
        "loss": loss,
        "f1_score": f1,
    }
```

### Using Different Datasets

Update `client.py` to load your dataset:

```python
# Replace CIFAR-10 with your dataset
from torchvision.datasets import MNIST

train_set = MNIST(root=DATASET_PATH, train=True, download=True, transform=transform)
test_set = MNIST(root=DATASET_PATH, train=False, download=True, transform=transform)
```

---

## Advanced: Running in POC or Production

### POC Environment

```python
from nvflare.recipe import PocEnv
from nvflare.app_opt.pt.recipes import PyTorchCrossSiteEvalRecipe
from model import SimpleNetwork

recipe = PyTorchCrossSiteEvalRecipe(
    name="hello-pt-cse",
    model=SimpleNetwork(),
    train_script="client.py",
)

env = PocEnv(num_clients=2)
run = recipe.execute(env)
```

### Production Environment

```python
from nvflare.recipe import ProdEnv
from nvflare.app_opt.pt.recipes import PyTorchCrossSiteEvalRecipe
from model import SimpleNetwork

recipe = PyTorchCrossSiteEvalRecipe(
    name="hello-pt-cse",
    model=SimpleNetwork(),
    train_script="client.py",
)

env = ProdEnv(startup_kit_location="/path/to/admin/startup/kit")
run = recipe.execute(env)
```

---

## Related Examples

- **[hello-pt](../hello-pt)**: Training + Cross-Site Evaluation with `add_cross_site_evaluation()`
- **[hello-numpy-cross-val](../hello-numpy-cross-val)**: NumPy version of CSE (simpler, auto-handled validation)

## Comparison with NumPy CSE

| Feature | PyTorch CSE | NumPy CSE |
|---------|-------------|-----------|
| Model Format | PyTorch state_dict | NumPy arrays |
| Validator Component | Not needed (Client API) | NPValidator required |
| Client Script Pattern | Check `flare.is_evaluate()` | Validation handled by NPValidator |
| Model Locator | PTFileModelLocator | NPModelLocator |
| Complexity | Slightly more complex | Simpler (auto-handled) |

---

## Troubleshooting

### Issue: Models not found

**Solution:** Make sure you run `generate_pretrain_models.py` first for standalone CSE mode.

### Issue: CIFAR-10 download fails

**Solution:** Check your internet connection or manually download CIFAR-10 to `/tmp/nvflare/data/cifar10`.

### Issue: CUDA out of memory

**Solution:** Reduce `batch_size` by modifying the `train_args` in `job.py` or passing smaller batches to the client script.

---

## Next Steps

- Try [hello-pt](../hello-pt) to see Training + CSE workflow
- Try the [NumPy CSE example](../hello-numpy-cross-val) for simpler models
- Learn about [experiment tracking](../../advanced/experiment-tracking) with TensorBoard, MLflow, or Weights & Biases
- Explore [custom recipes](https://nvflare.readthedocs.io/en/main/user_guide/data_scientist_guide/job_recipe.html) in the documentation
