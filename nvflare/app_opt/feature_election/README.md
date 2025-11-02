# Feature Election for NVIDIA FLARE

A plug-and-play horizontal federated feature selection framework for tabular datasets in NVIDIA FLARE.

## Overview

This work originates from FLASH: A framework for Federated Learning with Attribute Selection and Hyperparameter optimization framework a work presented in [FLTA IEEE 2025](https://flta-conference.org/flta-2025/) achieving the best student paper award.
Feature election enables multiple clients with tabular datasets to collaboratively identify the most relevant features without sharing raw data. It works by using conventional Feature selection algorithms in the client side and performing a weighted aggregation of their results.
FLASH is available on [Github](https://github.com/parasecurity/FLASH)

### Key Features

- **Easy Integration**: Simple API for tabular datasets (pandas, numpy)
- **Multiple Feature Selection Methods**: Lasso, Elastic Net, Mutual Information, PyImpetus, and more
- **Flexible Aggregation**: Configurable freedom degree (0=intersection, 1=union, 0-1=weighted voting)
- **Auto-tuning**: Automatic optimization of freedom degree parameter
- **Privacy-Preserving**: Only feature selections and scores are shared, not raw data
- **Production-Ready**: Fully compatible with NVIDIA FLARE workflows

## Installation

```bash
pip install nvflare
# Optional: for advanced feature selection
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
    auto_tune=True
)

# Get selected features
selected_features = df.columns[:-1][selected_mask]
print(f"Selected {len(selected_features)} features: {list(selected_features)}")
```

### Custom Configuration

```python
from nvflare.app_opt.feature_election import FeatureElection

# Initialize with custom parameters
fe = FeatureElection(
    freedom_degree=0.6,
    fs_method='elastic_net',
    aggregation_mode='weighted'
)

# Prepare data splits for clients
client_data = fe.prepare_data_splits(
    df=df,
    target_col='target',
    num_clients=5,
    split_strategy='stratified'  # or 'random', 'dirichlet'
)

# Run simulation
stats = fe.simulate_election(client_data)

# Access results
global_mask = fe.global_mask
selected_feature_names = fe.selected_feature_names
print(f"Reduction: {stats['reduction_ratio']:.1%}")
```

## NVIDIA FLARE Deployment

### 1. Generate Configuration Files

```python
from nvflare.app_opt.feature_election import FeatureElection

fe = FeatureElection(
    freedom_degree=0.5,
    fs_method='lasso',
    aggregation_mode='weighted'
)

# Generate FLARE job configuration
config_paths = fe.create_flare_job(
    job_name="feature_selection_job",
    output_dir="./jobs/feature_selection",
    min_clients=2,
    num_rounds=1
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

### 4. Retrieve Results

```python
# After job completion
from nvflare.fuel.flare_api.flare_api import new_secure_session

session = new_secure_session()
job_result = session.get_job_result(job_id)

# Extract global feature mask
global_mask = job_result['global_feature_mask']
selected_features = [feature_names[i] for i, selected in enumerate(global_mask) if selected]
```

## Feature Selection Methods

| Method | Description | Best For | Parameters |
|--------|-------------|----------|------------|
| `lasso` | L1 regularization | High-dimensional sparse data | `alpha` |
| `elastic_net` | L1+L2 regularization | Correlated features | `alpha`, `l1_ratio` |
| `random_forest` | Tree-based importance | Non-linear relationships | `n_estimators`, `max_depth` |
| `mutual_info` | Information gain | Any data type | `n_neighbors` |
| `f_classif` | ANOVA F-test | Gaussian features | `k` |
| `chi2` | Chi-squared test | Non-negative features | `k` |
| `pyimpetus` | Permutation importance | Robust feature selection | `p_val_thresh`, `num_sim` |

## Parameters

### FeatureElection

- **freedom_degree** (float, 0-1): Controls feature selection strategy
  - 0.0: Intersection only (most conservative)
  - 0.5: Balanced weighted voting (recommended)
  - 1.0: Union (most permissive)
  
- **fs_method** (str): Feature selection method (see table above)

- **aggregation_mode** (str): 'weighted' or 'uniform'
  - `weighted`: Weight by number of samples per client
  - `uniform`: Equal weight for all clients

- **auto_tune** (bool): Automatically optimize freedom_degree

### Data Splitting Strategies

- **stratified**: Maintains class distribution (recommended for classification)
- **random**: Random split
- **dirichlet**: Non-IID split with Dirichlet distribution
- **feature_split**: Each client gets different feature subsets

## Advanced Features

### Auto-tuning

Automatically finds the optimal freedom_degree:

```python
selected_mask, stats = quick_election(
    df=df,
    target_col='target',
    num_clients=4,
    auto_tune=True,
    candidate_freedoms=[0.0, 0.3, 0.5, 0.7, 1.0]
)
print(f"Optimal freedom_degree: {stats['freedom_degree']}")
```

### Cross-validation

Evaluate feature selection quality:

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Apply selected features
X_selected = X[:, selected_mask]

# Evaluate
clf = RandomForestClassifier()
scores = cross_val_score(clf, X_selected, y, cv=5)
print(f"CV Score: {scores.mean():.3f} ± {scores.std():.3f}")
```

### Saving and Loading Results

```python
# Save results
fe.save_results("feature_election_results.json")

# Load results
from nvflare.app_opt.feature_election import load_election_results
results = load_election_results("feature_election_results.json")
```

## Architecture

```
Server (Aggregator)
    │
    ├── FeatureElectionController
    │   ├── Collect feature selections from clients
    │   ├── Aggregate using freedom_degree
    │   └── Distribute global feature mask
    │
Clients (Executors)
    │
    ├── FeatureElectionExecutor
    │   ├── Perform local feature selection
    │   ├── Evaluate feature quality
    │   └── Send results to server
```

## Examples

See the `/examples` directory for comprehensive examples:

- `basic_usage.py`: Simple feature election
- `production_deployment.py`: Full FLARE deployment
- `high_dimensional.py`: Genomics/high-dimensional data
- `comparison.py`: Compare different methods
- `custom_methods.py`: Integrate custom feature selection

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
        auto_tune: bool = False
    )
    
    def prepare_data_splits(
        self,
        df: pd.DataFrame,
        target_col: str,
        num_clients: int,
        split_strategy: str = 'stratified'
    ) -> Dict
    
    def simulate_election(self, client_data: Dict) -> Dict
    
    def create_flare_job(
        self,
        job_name: str,
        output_dir: str,
        min_clients: int = 2,
        num_rounds: int = 1
    ) -> Dict[str, str]
```

#### FeatureElectionController

Server-side controller for NVIDIA FLARE.

```python
class FeatureElectionController(ScatterAndGather):
    def __init__(
        self,
        freedom_degree: float = 0.1,
        aggregation_mode: str = 'weighted',
        min_clients: int = 2,
        num_rounds: int = 1
    )
```

#### FeatureElectionExecutor

Client-side executor for NVIDIA FLARE.

```python
class FeatureElectionExecutor(Executor):
    def __init__(
        self,
        fs_method: str = "lasso",
        fs_params: Optional[Dict] = None,
        eval_metric: str = "f1"
    )
    
    def set_data(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    )
```

### Helper Functions

```python
def quick_election(
    df: pd.DataFrame,
    target_col: str,
    num_clients: int = 4,
    fs_method: str = 'lasso',
    auto_tune: bool = True,
    **kwargs
) -> Tuple[np.ndarray, Dict]

def load_election_results(filepath: str) -> Dict
```

## Performance Considerations

### Memory Usage

For high-dimensional datasets (>10,000 features):
- Use sparse methods: `lasso`, `elastic_net`
- Consider feature batching
- Set appropriate `max_iter` parameters

### Computational Cost

| Method | Time Complexity | Best For |
|--------|----------------|----------|
| Lasso | O(n*p) | p > n |
| Mutual Info | O(n*p*log(n)) | n > p |
| Random Forest | O(n*p*log(n)*trees) | Medium datasets |
| PyImpetus | O(n*p*sim) | When accuracy critical |

### Scalability

- Clients: Tested with 2-100 clients
- Features: Tested with 10-50,000 features
- Samples: Tested with 100-1M samples per client

## Troubleshooting

### Common Issues

1. **"No features selected"**
   - Increase freedom_degree
   - Try different fs_method
   - Check feature scaling

2. **"Memory Error"**
   - Reduce num_sim for PyImpetus
   - Use Lasso instead of Random Forest
   - Enable feature batching

3. **"Poor performance after selection"**
   - Enable auto_tune
   - Increase min_clients
   - Try weighted aggregation

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Development Setup

```bash
git clone https://github.com/NVIDIA/NVFlare.git
cd NVFlare
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/unit_test/app_opt/test_feature_election.py
pytest tests/integration_test/app_opt/test_feature_election_integration.py
```

## Citation

If you use this library in your research, please cite (PENDING)

<!--```bibtex
@inproceedings{flash2024,
  title={FLASH: A framework for Federated Learning with Attribute Selection and Hyperparameter optimization},
  author={[Ioannis Christofilogiannis, Georgios Valavanis, Alexander Shevtsov, Ioannis Lamprou and Sotiris Ioannidis
]},
  booktitle={[FLTA]},
  year={2025}
}
```-->

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NVIDIA FLARE team for the federated learning framework
- FLASH paper authors (Ioannis Christofilogiannis, Georgios Valavanis, Alexander Shevtsov, Ioannis Lamprou and Sotiris Ioannidis) for the feature election algorithm
- Future contributors and users of this library

## Support

-**FLASH Repository**: [Github](https://github.com/parasecurity/FLASH)
- **Flare Documentation**: [Full documentation](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_opt.feature_election.html)

