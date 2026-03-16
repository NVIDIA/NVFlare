# Feature Election for NVIDIA FLARE

A plug-and-play horizontal federated feature selection framework for tabular datasets in NVIDIA FLARE.

This work originates from FLASH: A Framework for Federated Learning with Attribute Selection and Hyperparameter Optimization, presented at [FLTA IEEE 2025](https://flta-conference.org/flta-2025/) achieving the **Best Student Paper Award**.

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

## NVIDIA FLARE Installation

For the complete installation instructions, see [Installation](https://nvflare.readthedocs.io/en/main/installation.html)

```bash
pip install nvflare
```

Install optional dependencies:

```bash
pip install PyImpetus  # Optional: enables permutation importance methods
```

> **Note:** `scikit-learn ≥ 1.0` is required for most feature selection methods and is automatically installed with `nvflare`.

## Code Structure

```
feature_election/
|
|-- job.py             # Main entry point - creates and runs FL job
|-- client.py          # Client-side executor with data loading and local feature selection
|-- prepare_data.py    # Synthetic dataset generation and client data splitting utilities
```

## Data

Feature Election works with any tabular dataset represented as a pandas DataFrame. In a real FL experiment, each client would have their own local dataset — only feature selections and scores are shared, never raw data.

For the quick-start example, synthetic data is generated automatically. To use your own data, modify `client.py` to load it:

```python
class MyDataExecutor(FeatureElectionExecutor):
    def _load_data_if_needed(self, fl_ctx):
        if self._data_loaded:
            return

        # Retrieve the site name assigned by the FL platform (e.g. "site-1").
        # FeatureElectionExecutor has no client_id attribute; use fl_ctx instead.
        site_name = fl_ctx.get_identity_name()
        X_train, y_train = load_my_data(site_name)
        self.set_data(X_train, y_train)
        self._data_loaded = True
```

You can control the synthetic dataset configuration directly from the command line:

```bash
python job.py \
    --n-samples 2000 \
    --n-features 200 \
    --n-informative 40 \
    --n-redundant 60 \
    --split-strategy non_iid
```

### Data Splitting Strategies

| Strategy | Description |
|----------|-------------|
| `stratified` | Maintains class distribution (recommended for classification) |
| `random` | Random split |
| `non_iid ` | Non-IID split with Dirichlet distribution (alpha=0.5) |

## Model

Feature Election follows a three-phase federated workflow:

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

The `freedom_degree` parameter controls how features are selected across clients:

- `0` = intersection (only features selected by all clients)
- `1` = union (any feature selected by at least one client)
- `0–1` = weighted voting threshold

### Feature Selection Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `lasso` | L1 regularization | High-dimensional sparse data |
| `elastic_net` | L1+L2 regularization | Correlated features |
| `random_forest` | Tree-based importance | Non-linear relationships |
| `mutual_info` | Information gain | Any data type |
| `pyimpetus` | Permutation importance | Robust feature selection |

## Client

The client code (`client.py`) is responsible for local feature selection. It loads local data, runs the configured feature selection method, and sends the resulting feature mask and scores to the server — **no raw data is ever shared**.

```python
from nvflare.app_opt.feature_election import FeatureElectionExecutor

executor = FeatureElectionExecutor(
    fs_method='lasso',
    eval_metric='f1'
)

# Load and set client data
X_train, y_train = load_client_data()  # Your data loading logic
executor.set_data(X_train, y_train, feature_names=feature_names)
```

The client workflow:
1. Receive the global task from the FL server.
2. Perform local feature selection using the configured method.
3. Send feature votes and scores back to the server.
4. Receive the global feature mask and train on the reduced feature set.

### FeatureElectionExecutor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fs_method` | str | `"lasso"` | Feature selection method |
| `fs_params` | dict | `None` | Additional method-specific parameters |
| `eval_metric` | str | `"f1"` | Metric used to evaluate the reduced feature set |
| `task_name` | str | `"feature_election"` | Must match the server controller |

## Server

The server-side controller (`FeatureElectionController`) aggregates feature votes from all clients, optionally tunes the `freedom_degree` via hill-climbing, and broadcasts the final global feature mask.

With the Recipe API, **there is no need to write custom server code** for the aggregation logic. The controller handles everything automatically:

1. Collect feature selections and scores from all clients.
2. Run auto-tuning (if enabled) to find the optimal `freedom_degree`.
3. Compute the global feature mask using weighted voting.
4. Distribute the mask and coordinate FedAvg training on the reduced feature set.

```python
from nvflare.app_opt.feature_election import FeatureElectionController

controller = FeatureElectionController(
    freedom_degree=0.5,
    aggregation_mode='weighted',
    min_clients=2,
    num_rounds=5,
    auto_tune=True,
    tuning_rounds=4,          # must be >= 2 when auto_tune=True
    wait_time_after_min_received=10,  # seconds; use 0 only for local simulation
)
```

### FeatureElectionController Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `freedom_degree` | float | `0.5` | Initial freedom degree |
| `aggregation_mode` | str | `"weighted"` | Client vote weighting (`'weighted'` or `'uniform'`) |
| `min_clients` | int | `2` | Minimum clients required per phase |
| `num_rounds` | int | `5` | FL training rounds after feature selection |
| `auto_tune` | bool | `False` | Enable hill-climbing optimisation of `freedom_degree` |
| `tuning_rounds` | int | `0` | Rounds of hill-climbing; **must be ≥ 2** when `auto_tune=True` (0 = no tuning, 1 = disabled with warning) |
| `train_timeout` | int | `300` | Per-phase timeout in seconds |
| `wait_time_after_min_received` | int | `10` | Seconds to wait for stragglers after `min_clients` have responded; set to `0` only for local simulation |

> **Auto-tune note:** `auto_tune=True` has no effect when `tuning_rounds=0` (the default). The controller emits a warning and skips tuning in that case. Use `tuning_rounds >= 2` to activate hill-climbing.
>
> **Production note:** `wait_time_after_min_received=10` gives slower clients a window to participate in every phase. Setting it to `0` causes the controller to close each phase the instant `min_clients` responses arrive, silently excluding any later responders.

## Job

The job recipe (`job.py`) combines the client and server into a runnable FLARE job. It generates all necessary configuration files and submits them to the simulator or a production FLARE deployment.

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

To export job configs for production deployment:

```bash
python job.py --export-dir ./exported_jobs
```

Then submit to a running FLARE deployment:

```bash
nvflare job submit -j ./jobs/feature_selection
```

### Job Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-clients` | `3` | Number of federated clients |
| `--num-rounds` | `5` | FL training rounds |
| `--freedom-degree` | `0.5` | Feature inclusion threshold (0–1) |
| `--auto-tune` | `False` | Enable freedom degree optimization |
| `--tuning-rounds` | `4` | Rounds for auto-tuning |
| `--fs-method` | `lasso` | Feature selection method |
| `--split-strategy` | `stratified` | Data splitting strategy |
| `--n-samples` | `1000` | Total synthetic samples |
| `--n-features` | `100` | Number of features |
| `--workspace` | `/tmp/nvflare/feature_election` | Simulator workspace |

## Run Job

From the terminal, run with default settings:

```bash
python job.py --num-clients 3 --num-rounds 5
```

With auto-tuning enabled:

```bash
python job.py --num-clients 3 --auto-tune --tuning-rounds 4
```

With a specific feature selection method:

```bash
# Mutual Information
python job.py --fs-method mutual_info

# Random Forest
python job.py --fs-method random_forest

# Elastic Net
python job.py --fs-method elastic_net
```

For quick simulation using the Python API:

```python
from nvflare.app_opt.feature_election import quick_election
import pandas as pd

df = pd.read_csv("your_data.csv")

selected_mask, stats = quick_election(
    df=df,
    target_col='target',
    num_clients=4,
    fs_method='lasso',
)

selected_features = df.columns[:-1][selected_mask]
print(f"Selected {len(selected_features)} features: {list(selected_features)}")
print(f"Freedom degree: {stats['freedom_degree']}")
```

## Troubleshooting

**"No features selected"** — Increase `freedom_degree`, try a different `fs_method`, or check feature scaling.

**"No feature votes received"** — Ensure client data is loaded before execution and that `task_name` matches between controller and executor.

**"Poor performance after selection"** — Enable `auto_tune` with `tuning_rounds >= 2` to find the optimal `freedom_degree`, or switch to `weighted` aggregation mode.

**"Auto-tune has no effect"** — `auto_tune=True` requires `tuning_rounds >= 2`. The default `tuning_rounds=0` is intentional for users who set `freedom_degree` manually; the controller logs a warning if `auto_tune=True` is combined with `tuning_rounds=0` or `tuning_rounds=1`.

**"Slower clients are not participating"** — The default `wait_time_after_min_received=10` gives stragglers a 10-second window after the minimum quorum is reached. If clients are still being excluded, increase this value. Set to `0` only for local simulation where all clients run in the same process.

**"Client excluded after mask distribution failure"** — If fewer than `min_clients` clients acknowledge the global mask, the entire workflow is aborted (not just Phase 3). Check network connectivity and client logs for the root cause.

**"PyImpetus not available"** — Install with `pip install PyImpetus`. The framework falls back to mutual information if unavailable.

Enable detailed logging for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Running Tests

```bash
pytest tests/unit_test/app_opt/feature_election/test.py -v
```