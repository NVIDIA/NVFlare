# Federated SVM with Scikit-learn and cuML

Please make sure you set up virtual environment and Jupyterlab follows [example root readme](../../README.md)

## Introduction to Scikit-learn, tabular data, and federated SVM
### Scikit-learn
This example shows how to use [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) on tabular data.
It uses [Scikit-learn](https://scikit-learn.org/), a widely used open-source machine learning library that supports supervised and unsupervised learning.
Follow along in this [notebook](./sklearn_svm_cancer.ipynb) for an interactive experience.
### cuML
You can also use [cuML](https://docs.rapids.ai/api/cuml/stable/) as backend instead of Scikit-learn.
Please install cuml following instructions: https://rapids.ai/start.html

### Tabular data
The data used in this example is tabular in a format that can be handled by [pandas](https://pandas.pydata.org/), such that:
- rows correspond to data samples.
- the first column represents the label.
- the other columns cover the features.

Each client is expected to have one local data file containing both training and validation samples.
To load the data for each client, the following parameters are expected by local learner:
- data_file_path: (`string`) the full path to the client's data file.
- train_start: (`int`) start row index for the training set.
- train_end: (`int`) end row index for the training set.
- valid_start: (`int`) start row index for the validation set.
- valid_end: (`int`) end row index for the validation set.

### Federated SVM
The machine learning algorithm shown in this example is [SVM for Classification (SVC)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).
Under this setting, federated learning can be formulated in two steps:
- local training: each client trains a local SVM model with their own data
- global training: server collects the support vectors from all clients and
  trains a global SVM model based on them

Unlike other iterative federated algorithms, federated SVM only involves
these two training steps. Hence, in the server config, we have
```
"num_rounds": 2
```
The first round is the training round, performing local training and global aggregation.
Next, the global model will be sent back to clients for the second round,
performing model validation and local model update.
If this number is set to a number greater than 2, the system will report an error and exit.

## Data preparation
This example uses the breast cancer dataset available from Scikit-learn's dataset API.

First, we will load the data, format it properly by removing the header, order
the label and feature columns, and save it to a CSV file with comma separation.
The default path is `/tmp/nvflare/dataset/cancer.csv`.
```commandline
bash prepare_data.sh
```

## Run with Job Recipe (Recommended)

The simplest way to run this example is using the Job Recipe API:

### Basic Usage

```bash
python job.py --n_clients 3 --kernel rbf --backend sklearn --data_path /tmp/nvflare/dataset/cancer.csv
```

This will:
- Create an SVM recipe with 3 clients using rbf kernel
- Run in simulation environment (all clients on one machine as threads)
- Train for 2 rounds: round 0 (training), round 1 (validation)
- Store results in `/tmp/nvflare/simulation/sklearn_svm/`

### Options

```bash
python job.py --help
```

Available arguments:
- `--n_clients`: Number of clients (default: 3)
- `--kernel`: Kernel type - linear, poly, rbf, or sigmoid (default: rbf)
- `--backend`: Backend library - sklearn or cuml (default: sklearn)
- `--data_path`: Path to cancer CSV file (default: /tmp/nvflare/dataset/cancer.csv)

### Per-Client Data Splits

The job automatically divides data into **non-overlapping ranges** for each client:
- First 80% of data (455 samples) split among clients for training
- Last 20% (114 samples) used as shared validation set
- Each client receives different `--train_start`, `--train_end`, `--valid_start`, `--valid_end` arguments

**Example splits for 3 clients:**
- site-1: train [0:151], valid [455:569]
- site-2: train [151:303], valid [455:569]
- site-3: train [303:455], valid [455:569]

> **Alternative**: Instead of using data ranges, you can also save the split data to separate individual files and pass different `--data_path` to each client. See the "Alternative: Using Separate Data Files" section below.

**Customizing Split Logic**

Modify `calculate_data_splits()` in `job.py` to implement different strategies:
- **Non-IID splits**: Assign different class distributions to clients
- **Unbalanced splits**: Give clients different amounts of data
- **Separate validation**: Use different validation sets per client

Pass a dict to `train_args` for per-client configuration:
```python
train_args = {
    "site-1": "--data_path /data/cancer.csv --backend sklearn --train_start 0 --train_end 151 ...",
    "site-2": "--data_path /data/cancer.csv --backend sklearn --train_start 151 --train_end 303 ...",
    # ... more sites
}
```

**Alternative: Using Separate Data Files**

As an alternative to passing different data range arguments, you can also save the split data to separate individual files and pass different `--data_path` to each client:

```python
# Split data into separate files (e.g., using prepare_data.py or pandas)
# - /data/site1_cancer.csv  (contains rows 0-151)
# - /data/site2_cancer.csv  (contains rows 151-303)
# - /data/site3_cancer.csv  (contains rows 303-455)

# Then configure per-client data paths in job.py:
train_args = {
    "site-1": "--data_path /data/site1_cancer.csv --backend sklearn",
    "site-2": "--data_path /data/site2_cancer.csv --backend sklearn",
    "site-3": "--data_path /data/site3_cancer.csv --backend sklearn",
}

# No need to pass --train_start, --train_end, etc. when using separate files
```

### Using cuML Backend

For GPU-accelerated SVM training:
```bash
python job.py --n_clients 3 --kernel rbf --backend cuml
```

### View Results

You can use TensorBoard to view the training metrics:
```bash
tensorboard --logdir /tmp/nvflare/simulation/sklearn_svm
```

### Different Execution Environments

The same recipe can run in different environments by changing just one line:

**Simulation (default)**: All clients run as threads in a single process
```python
from nvflare.recipe import SimEnv
env = SimEnv(num_clients=3)
run = recipe.execute(env)
```

**Proof-of-Concept**: Clients run as separate processes on one machine
```python
from nvflare.recipe import PocEnv
env = PocEnv(num_clients=3)
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
- `job.py`: Defines the federated learning job using the `SVMFedAvgRecipe`
- `client.py`: Client training script using the NVFlare Client API

The recipe automatically handles:
- Server-side component configuration (controller, aggregator, persistor, SVMAssembler)
- Client-side executor setup
- Job packaging and deployment
- Two-round training cycle (train + validate)

### Understanding SVM Training

Unlike iterative algorithms, federated SVM only requires one training round:
- **Round 0 (Training)**: Each client trains a local SVM and sends support vectors to the server.
  The server aggregates all support vectors and trains a global SVM.
- **Round 1 (Validation)**: Clients validate the global model using the global support vectors.

This is automatically configured by the recipe!

---

## Results

Running with default [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) classifier with RBF kernel, the
resulting global model's AUC is approximately 0.8088, which can be seen in the clients' logs or TensorBoard.

You can visualize the training metrics:
```bash
tensorboard --logdir /tmp/nvflare/simulation/sklearn_svm
```

---

## Legacy Approach

> **Note**: This example has been updated to use the simplified Job Recipe API. If you need the previous JSON-based configuration approach, please refer to the [NVFlare 2.6 documentation](https://github.com/NVIDIA/NVFlare/tree/2.6/examples/advanced/sklearn-svm) or earlier versions.
