# Custom Aggregator Example

This directory demonstrates how to use custom aggregators with NVFlare's `FedAvgRecipe`.

## Overview

The `job.py` file provides a complete example of running federated learning with custom aggregation strategies. Two custom aggregators are implemented in `custom_aggregators.py`:

### 1. **WeightedAggregator**
Weights each client's contribution by their number of training steps (or dataset size). This is more fair when clients have different amounts of data.

**Use case**: When clients have heterogeneous dataset sizes and you want to weight their contributions proportionally.

### 2. **MedianAggregator**
Computes the element-wise median of all client models instead of averaging. This provides robustness against Byzantine (malicious) clients.

**Use case**: When you need protection against adversarial clients who might send malicious model updates.

## Usage

### Basic Usage

Run with weighted aggregator (default):
```bash
python job.py --aggregator weighted --n_clients 8 --num_rounds 50 --alpha 0.1 --seed 0
```

Run with median aggregator:
```bash
python job.py --aggregator median --n_clients 8 --num_rounds 50 --alpha 0.1 --seed 0
```

Run with default FedAvg aggregator (for comparison):
```bash
python job.py --aggregator default --n_clients 8 --num_rounds 50 --alpha 0.1 --seed 0
```

**Important**: Use the same `--seed` value to ensure identical model initialization across experiments!

### Command-line Arguments

#### Aggregator Selection
- `--aggregator {weighted,median,default}` - Choose aggregation strategy (default: `weighted`)

#### Federated Learning Parameters
- `--n_clients` - Number of federated learning clients (default: `8`)
- `--num_rounds` - Number of FL rounds (default: `50`)
- `--alpha` - Data heterogeneity parameter (default: `0.5`)
  - Higher values (e.g., 1.0) = more uniform data distribution
  - Lower values (e.g., 0.1) = more heterogeneous/non-IID distribution
- `--seed` - Random seed for model initialization and reproducibility (default: `0`)
  - Sets random seeds for Python, NumPy, PyTorch (CPU & CUDA), and makes CUDNN deterministic
  - **Important**: Use the same seed across experiments for fair comparison!

#### Training Parameters
- `--aggregation_epochs` - Local epochs per round (default: `4`)
- `--lr` - Learning rate (default: `0.05`)
- `--batch_size` - Training batch size (default: `64`)
- `--num_workers` - Data loading workers (default: `2`)

#### Other Options
- `--name` - Custom job name (default: auto-generated based on aggregator and alpha)

## Examples

### Compare All Three Aggregators

Run these commands to compare the performance of different aggregators on highly heterogeneous data (alpha=0.1):

```bash
# Default FedAvg aggregator (baseline)
python job.py --aggregator default --n_clients 8 --num_rounds 50 --alpha 0.1 --seed 0

# Weighted aggregator
python job.py --aggregator weighted --n_clients 8 --num_rounds 50 --alpha 0.1 --seed 0

# Median aggregator (Byzantine-robust)
python job.py --aggregator median --n_clients 8 --num_rounds 50 --alpha 0.1 --seed 0
```

All three runs will use the same seed, ensuring identical model initialization and data splits for fair comparison.

### Run on Different GPUs (Parallel Execution)

You can run experiments in parallel on different GPUs:

Terminal 1:
```bash
export CUDA_VISIBLE_DEVICES=0
python job.py --aggregator weighted --n_clients 8 --num_rounds 50 --alpha 0.1 --seed 0
```

Terminal 2:
```bash
export CUDA_VISIBLE_DEVICES=1
python job.py --aggregator median --n_clients 8 --num_rounds 50 --alpha 0.1 --seed 0
```

Terminal 3:
```bash
export CUDA_VISIBLE_DEVICES=2
python job.py --aggregator default --n_clients 8 --num_rounds 50 --alpha 0.1 --seed 0
```

## Implementation Details

### Custom Aggregator Structure

Each custom aggregator must inherit from `ModelAggregator` and implement three key methods:

```python
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator


class MyCustomAggregator(ModelAggregator):
    def __init__(self):
        super().__init__()
        # Initialize your state variables here
        self.params_type = None  # Track params_type from accepted models
    
    def accept_model(self, model: FLModel):
        """Called for each client submission - accumulate their contributions."""
        # Track and validate params_type from all models
        if self.params_type is None:
            self.params_type = model.params_type
        elif self.params_type != model.params_type:
            raise ValueError(
                f"ParamsType mismatch: expected {self.params_type}, got {model.params_type}. "
                "All client models must have the same params_type."
            )
        # ... your accumulation logic ...
    
    def aggregate_model(self) -> FLModel:
        """Called after all clients submit - perform aggregation and return result."""
        # ... your aggregation logic ...
        # Return with the same params_type as the accepted models
        return FLModel(params=aggregated_params, params_type=self.params_type)
    
    def reset_stats(self):
        """Reset internal state for the next round."""
        self.params_type = None
        # ... reset other state variables ...
```

**Key Points:**
- **`accept_model(model: FLModel)`**: Accumulate or store client contributions as they arrive
- **`aggregate_model() -> FLModel`**: Perform your aggregation logic and return the aggregated model
- **`reset_stats()`**: Clear all internal state for the next round
- **Important**: Return the aggregated `FLModel` with the same `params_type` as the accepted models
- **Validation**: Check that all client models have the same `params_type` to catch configuration errors early
- Use `self.info()`, `self.error()`, etc. for logging (provided by `ModelAggregator`)

### Key Concepts

1. **FLModel**: NVFlare's standardized data structure for exchanging models
   - `params`: Dictionary of model parameters (e.g., layer weights)
   - `params_type`: Indicates whether params are full weights (`FULL`) or differences (`DIFF`)
   - `meta`: Dictionary for metadata (e.g., `NUM_STEPS_CURRENT_ROUND` for training steps)
   - `metrics`: Dictionary for metrics (e.g., validation accuracy)

2. **ModelAggregator vs Aggregator**: 
   - `ModelAggregator`: Higher-level API that works with `FLModel` objects directly
   - Provides built-in logging methods (`self.info()`, `self.error()`)
   - Simpler to use for most custom aggregation scenarios
   - The `Aggregator` base class works with lower-level `Shareable` objects

3. **ParamsType**: Enum indicating the type of parameters
   - `ParamsType.FULL`: Full model weights
   - `ParamsType.DIFF`: Model weight differences (used in this example)

## Viewing Results

After running a job, view the training curves with TensorBoard:

```bash
tensorboard --logdir=/tmp/nvflare/simulation
```

Then open `http://localhost:6006` in your browser.

## Extending This Example

To create your own custom aggregator:

1. Create a new class inheriting from `ModelAggregator` in `custom_aggregators.py`
2. Implement the three required methods: `accept_model()`, `aggregate_model()`, `reset_stats()`
3. Add your aggregator to the `get_aggregator()` function in `job.py`
4. Add it to the `--aggregator` choices in `define_parser()`

Example skeleton:

```python
import numpy as np
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator


class MyCustomAggregator(ModelAggregator):
    def __init__(self, custom_param=1.0):
        super().__init__()
        self.custom_param = custom_param
        # Initialize your state variables
        self.accumulated_params = {}
        self.params_type = None
    
    def accept_model(self, model: FLModel):
        """Process incoming client model."""
        # Track and validate params_type
        if self.params_type is None:
            self.params_type = model.params_type
        elif self.params_type != model.params_type:
            raise ValueError(
                f"ParamsType mismatch: expected {self.params_type}, got {model.params_type}"
            )
        
        # Your custom accumulation logic here
        for key, value in model.params.items():
            if key not in self.accumulated_params:
                self.accumulated_params[key] = [value]
            else:
                self.accumulated_params[key].append(value)
    
    def aggregate_model(self) -> FLModel:
        """Perform custom aggregation."""
        if not self.accumulated_params:
            self.error("No models to aggregate!")
            return FLModel(params={})
        
        # Your custom aggregation logic
        aggregated_params = {}
        for key, value_list in self.accumulated_params.items():
            # Example: compute mean
            aggregated_params[key] = np.mean(value_list, axis=0)
        
        # Return with the same params_type as accepted models
        return FLModel(params=aggregated_params, params_type=self.params_type)
    
    def reset_stats(self):
        """Reset state for next round."""
        self.accumulated_params = {}
        self.params_type = None
```

For complete working examples, see `custom_aggregators.py` in this directory.

## References

- [NVFlare Documentation](https://nvflare.readthedocs.io/)
- [FedAvg Paper](https://arxiv.org/abs/1602.05629)
- [Byzantine-Robust Aggregation](https://arxiv.org/abs/1803.01498)

