# Feature Election Examples

Examples demonstrating federated feature selection using NVIDIA FLARE.

## Quick Start

Run the synthetic data example with auto-tuning:

```bash
python job.py --num-clients 3 --auto-tune --fs-method mutual_info
```

## Files

| File | Description |
|------|-------------|
| `job.py` | Main entry point - creates and runs FL job |
| `client.py` | Client-side executor with synthetic data loading |
| `server.py` | Server configuration helpers |

## Usage

### Basic Run

```bash
python job.py --num-clients 3 --num-rounds 5
```

### With Auto-tuning

```bash
python job.py --num-clients 3 --auto-tune --tuning-rounds 4
```

### Different Feature Selection Methods

```bash
# Lasso (default)
python job.py --fs-method lasso

# Mutual Information
python job.py --fs-method mutual_info

# Random Forest
python job.py --fs-method random_forest

# Elastic Net
python job.py --fs-method elastic_net
```

### Custom synthetic dataset configuration

```bash
python job.py \
    --n-samples 2000 \
    --n-features 200 \
    --n-informative 40 \
    --n-redundant 60 \
    --split-strategy dirichlet
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-clients` | 3 | Number of federated clients |
| `--num-rounds` | 5 | FL training rounds |
| `--freedom-degree` | 0.5 | Feature inclusion threshold (0-1) |
| `--auto-tune` | False | Enable freedom degree optimization |
| `--tuning-rounds` | 4 | Rounds for auto-tuning |
| `--fs-method` | lasso | Feature selection method |
| `--split-strategy` | stratified | Data splitting strategy |
| `--n-samples` | 1000 | Total synthetic samples |
| `--n-features` | 100 | Number of features |
| `--workspace` | /tmp/nvflare/feature_election | Simulator workspace |

## Customization

### Using Your Own Data

Modify `client.py` to load your data instead of synthetic data:

```python
class MyDataExecutor(FeatureElectionExecutor):
    def _load_data_if_needed(self, fl_ctx):
        if self._data_loaded:
            return
        
        # Load your data
        X_train, y_train = load_my_data(self.client_id)
        self.set_data(X_train, y_train)
        self._data_loaded = True
```

### Exporting Job Configuration

```bash
python job.py --export-dir ./exported_jobs
```

This creates FLARE job configs that can be deployed to production.