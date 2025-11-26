# Feature Election for NVIDIA FLARE

A plug-and-play horizontal federated feature selection framework for tabular datasets in NVIDIA FLARE.

## Overview

This work originates from FLASH: A framework for Federated Learning with Attribute Selection and Hyperparameter optimization, a work presented at [FLTA IEEE 2025](https://flta-conference.org/flta-2025/) achieving the Best Student Paper Award.

Feature Election enables multiple clients with tabular datasets to collaboratively identify the most relevant features without sharing raw data. It works by using conventional feature selection algorithms on the client side and performing a weighted aggregation of their results.

FLASH is available on [GitHub](https://github.com/parasecurity/FLASH)

### Key Features

- **Easy Integration**: Simple API for tabular datasets (pandas, numpy)
- **Multiple Feature Selection Methods**: Lasso, Elastic Net, Mutual Information, RFE, Random Forest, PyImpetus, and more
- **Flexible Aggregation**: Configurable freedom degree (0=intersection, 1=union, 0-1=weighted voting)
- **Auto-tuning**: Automatic optimization of freedom degree parameter
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
    auto_tune=True
)

# Get selected features
selected_features = df.columns[:-1][selected_mask]
print(f"Selected {len(selected_features)} features: {list(selected_features)}")
print(f"Optimal freedom_degree: {stats['freedom_degree']}")
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
    split_strategy='stratified'  # or 'random', 'sequential', 'dirichlet'
)

# Run simulation
stats = fe.simulate_election(client_data)

# Access selected features
selected_features = fe.selected_feature_names
print(f"Selected {stats['num_features_selected']} features")
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
    num_rounds=1,
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
| `lasso` | L1 regularization | High-dimensional sparse data | `alpha`, `max_iter` |
| `elastic_net` | L1+L2 regularization | Correlated features | `alpha`, `l1_ratio`, `max_iter` |
| `random_forest` | Tree-based importance | Non-linear relationships | `n_estimators`, `max_depth`, `k` |
| `mutual_info` | Information gain | Any data type | `n_neighbors`, `k` |
| `f_classif` | ANOVA F-test | Gaussian features | `k` |
| `chi2` | Chi-squared test | Non-negative features | `k` |
| `rfe` | Recursive Feature Elimination | Iterative selection | `n_features_to_select`, `step` |
| `selectkbest` | SelectKBest wrapper | General use | `k`, `score_func` |
| `pyimpetus` | Permutation importance | Robust feature selection | `p_val_thresh`, `num_sim`, `model` |

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
- **sequential**: Sequential split for ordered data
- **dirichlet**: Non-IID split with Dirichlet distribution (alpha=0.5)

## Advanced Features

### Auto-tuning

Automatically finds the optimal freedom_degree:

```python
selected_mask, stats = quick_election(
    df=df,
    target_col='target',
    num_clients=4,
    auto_tune=True
)
print(f"Optimal freedom_degree: {stats['freedom_degree']}")
```

### Applying Feature Mask to New Data

```python
# After running election
X_test_selected = fe.apply_mask(X_test)
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

# Or load into an existing FeatureElection instance
fe.load_results("feature_election_results.json")
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
- `flare_deployment.py`: Deployment example

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
        num_clients: int = 3,
        split_strategy: str = "stratified",
        split_ratios: Optional[List[float]] = None,
        random_state: int = 42
    ) -> List[Tuple[pd.DataFrame, pd.Series]]
    
    def simulate_election(
        self,
        client_data: List[Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]],
        feature_names: Optional[List[str]] = None
    ) -> Dict
    
    def create_flare_job(
        self,
        job_name: str = "feature_election",
        output_dir: str = "jobs/feature_election",
        min_clients: int = 2,
        num_rounds: int = 1,
        client_sites: Optional[List[str]] = None
    ) -> Dict[str, str]
    
    def apply_mask(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None
    ) -> Union[pd.DataFrame, np.ndarray]
    
    def save_results(self, filepath: str)
    
    def load_results(self, filepath: str)
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
        num_rounds: int = 1,
        task_name: str = "feature_election",
        train_timeout: int = 0
    )
    
    def get_results(self) -> Dict
```

#### FeatureElectionExecutor

Client-side executor for NVIDIA FLARE.

```python
class FeatureElectionExecutor(Executor):
    def __init__(
        self,
        fs_method: str = "lasso",
        fs_params: Optional[Dict] = None,
        eval_metric: str = "f1",
        quick_eval: bool = True,
        task_name: str = "feature_election"
    )
    
    def set_data(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    )
    
    def get_selected_features(self) -> Optional[np.ndarray]
    
    def get_feature_names(self) -> Optional[List[str]]
    
    def get_pyimpetus_info(self) -> Dict[str, Any]
```

### Convenience Functions

#### quick_election

```python
def quick_election(
    df: pd.DataFrame,
    target_col: str,
    num_clients: int = 3,
    freedom_degree: float = 0.5,
    fs_method: str = "lasso",
    auto_tune: bool = False,
    split_strategy: str = "stratified",
    **kwargs
) -> Tuple[np.ndarray, Dict]
```

#### load_election_results

```python
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

4. **"PyImpetus not available"**
   - Install with: `pip install PyImpetus`
   - The executor will fall back to mutual information if unavailable

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
pytest tests/unit_test/app_opt/feature_election/test_feature_election.py
```

## Citation

If you use Feature Election in your research, please cite (PENDING)

<!--```bibtex
@inproceedings{flash2025,
  title={FLASH: A framework for Federated Learning with Attribute Selection and Hyperparameter optimization},
  author={Ioannis Christofilogiannis, Georgios Valavanis, Alexander Shevtsov, Ioannis Lamprou and Sotiris Ioannidis},
  booktitle={FLTA IEEE 2025},
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

- **FLASH Repository**: [GitHub](https://github.com/parasecurity/FLASH)
