# Federated Linear Model with Scikit-learn

Please make sure you set up virtual environment and Jupyterlab follows [example root readme](../../README.md)

## Introduction to Scikit-learn, tabular data, and federated linear model
### Scikit-learn
This example shows how to use [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) on tabular data.
It uses [Scikit-learn](https://scikit-learn.org/), a widely used
open-source machine learning library that supports supervised and unsupervised learning.
Follow along in this [notebook](./sklearn_linear_higgs.ipynb) for an interactive experience.
### Tabular data
The data used in this example is tabular in a format that can be handled by [pandas](https://pandas.pydata.org/), such that:
- rows correspond to data samples.
- the first column represents the label.
- the other columns cover the features.

Each client is expected to have one local data file containing both training
and validation samples. To load the data for each client, the following
parameters are expected by the local learner:
- data_file_path: (`string`) the full path to the client's data file.
- train_start: (`int`) start row index for the training set.
- train_end: (`int`) end row index for the training set.
- valid_start: (`int`) start row index for the validation set.
- valid_end: (`int`) end row index for the validation set.

### Federated Linear Model
This example shows the use of [linear classifiers with SGD training](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) in a federated scenario.
Under this setting, federated learning can be formulated as a [FedAvg](https://arxiv.org/abs/1602.05629) process with local training that each client optimizes the local model starting from global parameters with SGD.
This can be achieved by setting the `warm_start` flag of SGDClassifier to
`True` in order to allow repeated fitting of the classifiers to the local data.

## Data preparation
The examples illustrate a binary classification task based on [HIGGS dataset](https://mlphysics.ics.uci.edu/data/higgs/).
This dataset contains 11 million instances, each with 28 attributes. Download the dataset from the HIGGS link above, containing a single `.csv` file.
By default, we assume the dataset is downloaded, uncompressed, and stored
in `DATASET_ROOT/HIGGS.csv`.

Please note that the UCI's website may experience occasional downtime.

To prepare the data:
```bash
bash prepare_data.sh
```

## Run with Job Recipe (Recommended)

The simplest way to run this example is using the Job Recipe API:

### Basic Usage

```bash
python job.py --n_clients 5 --num_rounds 50 --data_path /tmp/nvflare/dataset/HIGGS.csv
```

This will:
- Create a FedAvg recipe with 5 clients and 50 rounds
- Run in simulation environment (all clients on one machine as threads)
- Store results in `/tmp/nvflare/simulation/sklearn_linear/`

### Options

```bash
python job.py --help
```

Available arguments:
- `--n_clients`: Number of clients (default: 5)
- `--num_rounds`: Number of training rounds (default: 50)
- `--data_path`: Path to HIGGS.csv file (default: /tmp/nvflare/dataset/HIGGS.csv)

### Per-Client Data Splits

The job automatically divides data into **non-overlapping ranges** for each client:
- First 1.1M rows used for validation (shared across all clients)
- Remaining 9.9M rows split evenly among clients for training
- Each client receives different `--train_start`, `--train_end`, `--valid_start`, `--valid_end` arguments

**Example splits for 5 clients:**
- site-1: train [1100000:3080000], valid [0:1100000]
- site-2: train [3080000:5060000], valid [0:1100000]
- site-3: train [5060000:7040000], valid [0:1100000]
- site-4: train [7040000:9020000], valid [0:1100000]
- site-5: train [9020000:11000000], valid [0:1100000]

**Customizing Split Logic**

Modify `calculate_data_splits()` in `job.py` to implement different split strategies:
- **Non-IID splits**: Assign different label distributions to clients
- **Unbalanced splits**: Give clients different amounts of data (e.g., linear, exponential ratios)
- **Overlapping validation**: Use different validation sets per client

The key is passing a dict to `train_args`:
```python
train_args = {
    "site-1": "--data_path /data/HIGGS.csv --train_start 1100000 --train_end 3080000 ...",
    "site-2": "--data_path /data/HIGGS.csv --train_start 3080000 --train_end 5060000 ...",
    # ... more sites
}
```

**Alternative: Using Separate Data Files**

Instead of using data ranges, you can split your data into separate files for each client:

```python
# Split data into files (e.g., using prepare_data.py or pandas)
# - /data/site1_HIGGS.csv
# - /data/site2_HIGGS.csv
# - /data/site3_HIGGS.csv

train_args = {
    "site-1": "--data_path /data/site1_HIGGS.csv",
    "site-2": "--data_path /data/site2_HIGGS.csv",
    "site-3": "--data_path /data/site3_HIGGS.csv",
}
```

This replaces the old `prepare_job_config.sh` approach with a more flexible Python-based solution.

### View Results

You can use TensorBoard to view the training metrics:
```bash
tensorboard --logdir /tmp/nvflare/simulation/sklearn_linear
```

### Different Execution Environments

The same recipe can run in different environments by changing just one line:

**Simulation (default)**: All clients run as threads in a single process
```python
from nvflare.recipe import SimEnv
env = SimEnv(num_clients=5)
run = recipe.execute(env)
```

**Proof-of-Concept**: Clients run as separate processes on one machine
```python
from nvflare.recipe import PocEnv
env = PocEnv(num_clients=5)
run = recipe.execute(env)
```

**Production**: Clients run on separate machines in a real deployment
```python
from nvflare.recipe import ProdEnv
env = ProdEnv(startup_kit_location="/path/to/admin/startup/kit")
run = recipe.execute(env)
```

### How it Works

The recipe approach uses:
- `job.py`: Defines the federated learning job using the `SklearnFedAvgRecipe`
- `client.py`: Client training script using the NVFlare Client API

The recipe automatically handles:
- Server-side component configuration (controller, aggregator, persistor)
- Client-side executor setup
- Job packaging and deployment

---

## Results

Running with deterministic setting, the resulting curve showing the classification
performance using area-under the curve (AUC) is:

![linear curve](./figs/linear.png)

---

## Legacy Approach

> **Note**: This example has been updated to use the simplified Job Recipe API. If you need the previous JSON-based configuration approach, please refer to the [NVFlare 2.6 documentation](https://github.com/NVIDIA/NVFlare/tree/2.6/examples/advanced/sklearn-linear) or earlier versions.
