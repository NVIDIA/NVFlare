# Feature Election for NVIDIA FLARE

A plug-and-play horizontal federated feature selection framework for tabular datasets in NVIDIA FLARE.

## Overview

This work originates from FLASH: A framework for Federated Learning with Attribute Selection and Hyperparameter optimization, presented at [FLTA IEEE 2025](https://flta-conference.org/flta-2025/) achieving the Best Student Paper Award.

Feature Election enables multiple clients with tabular datasets to collaboratively identify the most relevant features without sharing raw data. It works by using conventional feature selection algorithms on the client side and performing a weighted aggregation of their results.

FLASH is available on [GitHub](https://github.com/parasecurity/FLASH)

## Citation

If you use Feature Election in your research, please cite the FLASH framework paper:

**IEEE Style:**
> I. Christofilogiannis, G. Valavanis, A. Shevtsov, I. Lamprou and S. Ioannidis, "FLASH: A Framework for Federated Learning with Attribute Selection and Hyperparameter Optimization," 2025 3rd International Conference on Federated Learning Technologies and Applications (FLTA), Dubrovnik, Croatia, 2025, pp. 93-100, doi: 10.1109/FLTA67013.2025.11336571.

**BibTeX:**
```bibtex
@INPROCEEDINGS{11336571,
  author={Christofilogiannis, Ioannis and Valavanis, Georgios and Shevtsov, Alexander and Lamprou, Ioannis and Ioannidis, Sotiris},
  booktitle={2025 3rd International Conference on Federated Learning Technologies and Applications (FLTA)}, 
  title={FLASH: A Framework for Federated Learning with Attribute Selection and Hyperparameter Optimization}, 
  year={2025},
  pages={93-100},
  doi={10.1109/FLTA67013.2025.11336571}
}
```

### Key Features

- **Easy Integration**: Simple API for tabular datasets (pandas, numpy)
- **Multiple Feature Selection Methods**: Lasso, Elastic Net, Mutual Information, Random Forest, PyImpetus, and more
- **Flexible Aggregation**: Configurable freedom degree (0=intersection, 1=union, 0-1=weighted voting)
- **Auto-tuning**: Automatic optimization of freedom degree using hill-climbing
- **Multi-phase Workflow**: Local FS → Feature Election with tuning → FL Aggregation
- **Privacy-Preserving**: Only feature selections and scores are shared, not raw data
- **Production-Ready**: Fully compatible with NVIDIA FLARE workflows

### Optional Dependencies

- `scikit-learn` ≥ 1.0 is required for most feature selection methods  
  → automatically installed with `pip install nvflare`

- `PyImpetus` ≥ 0.0.6 is optional (enables advanced permutation importance methods)  
  → install manually if needed:
```bash
pip install PyImpetus
```

## Quick Start

### Basic Usage

```python
from nvflare.app_opt.feature_election import quick_election
import pandas as pd

# Load your tabular dataset
df = pd.read_csv("your_data.csv")

# Run feature election (simulation mode)
selected_mask, stats = quick_election(
    df=df,
    target_col='target',
    num_clients=4,
    fs_method='lasso',
)

# Get selected features
selected_features = df.columns[:-1][selected_mask]
print(f"Selected {len(selected_features)} features: {list(selected_features)}")
print(f"Freedom degree: {stats['freedom_degree']}")
```

### Custom Configuration

```python
from nvflare.app_opt.feature_election import FeatureElection

# Initialize with custom parameters
fe = FeatureElection(
    freedom_degree=0.6,
    fs_method='elastic_net',
    aggregation_mode='weighted',
    auto_tune=True,
    tuning_rounds=5
)

# Prepare data splits for clients
client_data = fe.prepare_data_splits(
    df=df,
    target_col='target',
    num_clients=5,
    split_strategy='stratified'  # or 'random', 'sequential', 'dirichlet'
)

# Run simulation
stats = fe.simulate_election(client_data)

# Access selected features
selected_features = fe.selected_feature_names
print(f"Selected {stats['num_features_selected']} features")
```

## Workflow Architecture

The Feature Election workflow consists of three phases:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: Local Feature Selection             │
│  Clients perform local FS using configured method (lasso, etc.) │
│  → Each client sends: selected_features, feature_scores         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 2: Tuning & Global Mask Generation           │
│  If auto_tune=True: Hill-climbing to find optimal freedom_degree│
│  → Aggregates selections using weighted voting                  │
│  → Distributes global feature mask to all clients               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 PHASE 3: FL Aggregation (Training)              │
│  Standard FedAvg training on reduced feature set                │
│  → num_rounds of federated training                             │
└─────────────────────────────────────────────────────────────────┘
```

## NVIDIA FLARE Deployment

### 1. Generate Configuration Files

```python
from nvflare.app_opt.feature_election import FeatureElection

fe = FeatureElection(
    freedom_degree=0.5,
    fs_method='lasso',
    aggregation_mode='weighted',
    auto_tune=True,
    tuning_rounds=4
)

# Generate FLARE job configuration
config_paths = fe.create_flare_job(
    job_name="feature_selection_job",
    output_dir="./jobs/feature_selection",
    min_clients=2,
    num_rounds=5,
    client_sites=['hospital_1', 'hospital_2', 'hospital_3']
)
```

### 2. Prepare Client Data

Each client should prepare their data:

```python
from nvflare.app_opt.feature_election import FeatureElectionExecutor
import numpy as np

# In your client script
executor = FeatureElectionExecutor(
    fs_method='lasso',
    eval_metric='f1'
)

# Load and set client data
X_train, y_train = load_client_data()  # Your data loading logic
executor.set_data(X_train, y_train, feature_names=feature_names)
```

### 3. Submit FLARE Job

```bash
nvflare job submit -j ./jobs/feature_selection
```

## Feature Selection Methods

| Method | Description | Best For | Parameters |
|--------|-------------|----------|------------|
| `lasso` | L1 regularization | High-dimensional sparse data | `alpha`, `max_iter` |
| `elastic_net` | L1+L2 regularization | Correlated features | `alpha`, `l1_ratio`, `max_iter` |
| `random_forest` | Tree-based importance | Non-linear relationships | `n_estimators`, `max_depth` |
| `mutual_info` | Information gain | Any data type | `n_neighbors` |
| `pyimpetus` | Permutation importance | Robust feature selection | `p_val_thresh`, `num_sim` |

## Parameters

### FeatureElection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `freedom_degree` | float | 0.5 | Controls feature inclusion (0=intersection, 1=union) |
| `fs_method` | str | "lasso" | Feature selection method |
| `aggregation_mode` | str | "weighted" | How to weight client votes ('weighted' or 'uniform') |
| `auto_tune` | bool | False | Enable automatic tuning of freedom_degree |
| `tuning_rounds` | int | 5 | Number of rounds for auto-tuning |

### FeatureElectionController

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `freedom_degree` | float | 0.5 | Initial freedom degree |
| `aggregation_mode` | str | "weighted" | Client vote weighting |
| `min_clients` | int | 2 | Minimum clients required |
| `num_rounds` | int | 5 | FL training rounds after feature selection |
| `auto_tune` | bool | False | Enable auto-tuning |
| `tuning_rounds` | int | 0 | Number of tuning rounds |
| `train_timeout` | int | 300 | Training phase timeout (seconds) |

### Data Splitting Strategies

- **stratified**: Maintains class distribution (recommended for classification)
- **random**: Random split
- **sequential**: Sequential split for ordered data
- **dirichlet**: Non-IID split with Dirichlet distribution (alpha=0.5)

## API Reference

### Core Classes

#### FeatureElection

Main interface for feature election.

```python
class FeatureElection:
    def __init__(
        self,
        freedom_degree: float = 0.5,
        fs_method: str = "lasso",
        aggregation_mode: str = "weighted",
        auto_tune: bool = False,
        tuning_rounds: int = 5,
    )
    
    def prepare_data_splits(...) -> List[Tuple[pd.DataFrame, pd.Series]]
    def simulate_election(...) -> Dict
    def create_flare_job(...) -> Dict[str, str]
    def apply_mask(...) -> Union[pd.DataFrame, np.ndarray]
    def save_results(filepath: str)
    def load_results(filepath: str)
```

#### FeatureElectionController

Server-side controller for NVIDIA FLARE.

```python
class FeatureElectionController(Controller):
    def __init__(
        self,
        freedom_degree: float = 0.5,
        aggregation_mode: str = "weighted",
        min_clients: int = 2,
        num_rounds: int = 5,
        task_name: str = "feature_election",
        train_timeout: int = 300,
        auto_tune: bool = False,
        tuning_rounds: int = 0,
    )
```

#### FeatureElectionExecutor

Client-side executor for NVIDIA FLARE.

class FeatureElectionExecutor(Executor):
    def __init__(
        self,
        fs_method: str = "lasso",
        fs_params: Optional[Dict] = None,
        eval_metric: str = "f1",
        task_name: str = "feature_election"
    )
    
    def set_data(X_train, y_train, X_val=None, y_val=None, feature_names=None)
    def evaluate_model(X_train, y_train, X_val, y_val) -> float
```

### Convenience Functions

```python
def quick_election(
    df: pd.DataFrame,
    target_col: str,
    num_clients: int = 3,
    freedom_degree: float = 0.5,
    fs_method: str = "lasso",
    split_strategy: str = "stratified",
    **kwargs
) -> Tuple[np.ndarray, Dict]

def load_election_results(filepath: str) -> Dict
```

## Troubleshooting

### Common Issues

1. **"No features selected"**
   - Increase freedom_degree
   - Try different fs_method
   - Check feature scaling

2. **"No feature votes received"**
   - Ensure client data is loaded before execution
   - Check that task_name matches between controller and executor

3. **"Poor performance after selection"**
   - Enable auto_tune to find optimal freedom_degree
   - Try weighted aggregation mode

4. **"PyImpetus not available"**
   - Install with: `pip install PyImpetus`
   - Falls back to mutual information if unavailable

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Running Tests

```bash
pytest tests/unit_test/app_opt/feature_election/test_feature_election.py -v
```
pytest tests/unit_test/app_opt/feature_election/test.py -v
## Acknowledgments

- NVIDIA FLARE team for the federated learning framework
- FLASH paper authors (Ioannis Christofilogiannis, Georgios Valavanis, Alexander Shevtsov, Ioannis Lamprou and Sotiris Ioannidis) for the feature election algorithm

## Support

- **FLASH Repository**: [GitHub](https://github.com/parasecurity/FLASH)