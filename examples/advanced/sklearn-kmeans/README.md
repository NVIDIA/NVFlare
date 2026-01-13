# Federated K-Means Clustering with Scikit-learn

Please make sure you set up virtual environment and Jupyterlab follows [example root readme](../../README.md)

## Introduction to Scikit-learn, tabular data, and federated k-Means
### Scikit-learn
This example shows how to use [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) on tabular data.
It uses [Scikit-learn](https://scikit-learn.org/),
a widely used open-source machine learning library that supports supervised
and unsupervised learning.
Follow along in this [notebook](./sklearn_kmeans_iris.ipynb) for an interactive experience.
### Tabular data
The data used in this example is tabular in a format that can be handled by [pandas](https://pandas.pydata.org/), such that:
- rows correspond to data samples
- the first column represents the label
- the other columns cover the features.

Each client is expected to have one local data file containing both training
and validation samples. To load the data for each client, the following
parameters are expected by the local learner:
- data_file_path: string, the full path to the client's data file
- train_start: int, start row index for the training set
- train_end: int, end row index for the training set
- valid_start: int, start row index for the validation set
- valid_end: int, end row index for the validation set

### Federated k-Means clustering
The machine learning algorithm in this example is [k-Means clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html).
The aggregation follows the scheme defined in [Mini-batch k-Means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html).
Under this setting, each round of federated learning can be formulated as follows:
- local training: starting from global centers, each client trains a local MiniBatchKMeans model with their own data
- global aggregation: server collects the cluster center,
  counts information from all clients, aggregates them by considering
  each client's results as a mini-batch, and updates the global center and per-center counts.

For center initialization, at the first round, each client generates its
initial centers with the k-means++ method. Then, the server collects all
initial centers and performs one round of k-means to generate the initial
global center.

## Data preparation
This example uses the Iris dataset available from Scikit-learn's dataset API.
```commandline
bash prepare_data.sh
```
This will load the data, format it properly by removing the header, order
the label and feature columns, randomize the dataset, and save it to a CSV file with comma separation.
The default path is `/tmp/nvflare/dataset/sklearn_iris.csv`.

Note that the dataset contains a label for each sample, which will not be
used for training since k-Means clustering is an unsupervised method.
The entire dataset with labels will be used for performance evaluation
based on [homogeneity_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html).

## Run with Job Recipe (Recommended)

The simplest way to run this example is using the Job Recipe API:

### Basic Usage

```bash
python job.py --n_clients 3 --num_rounds 5 --n_clusters 3 --data_path /tmp/nvflare/dataset/sklearn_iris.csv
```

This will:
- Create a K-Means recipe with 3 clients, 5 rounds, and 3 clusters
- Run in simulation environment (all clients on one machine as threads)
- Store results in `/tmp/nvflare/simulation/sklearn_kmeans/`

### Options

```bash
python job.py --help
```

Available arguments:
- `--n_clients`: Number of clients (default: 3)
- `--num_rounds`: Number of training rounds (default: 5)
- `--n_clusters`: Number of clusters (default: 3)
- `--data_path`: Path to iris CSV file (default: /tmp/nvflare/dataset/sklearn_iris.csv)

### Per-Client Data Splits

The job automatically divides data into **non-overlapping ranges** for each client:
- First 80% of data (120 samples) split among clients for training
- Last 20% (30 samples) used as shared validation set
- Each client receives different `--train_start`, `--train_end`, `--valid_start`, `--valid_end` arguments

**Example splits for 3 clients:**
- site-1: train [0:40], valid [120:150]
- site-2: train [40:80], valid [120:150]
- site-3: train [80:120], valid [120:150]

**Customizing Split Logic**

Modify `calculate_data_splits()` in `job.py` to implement different strategies:
- **Non-IID splits**: Assign different class distributions to clients
- **Unbalanced splits**: Give clients different amounts of data
- **Separate validation**: Use different validation sets per client

Use `per_site_config` to pass `train_args` for per-client configuration:
```python
per_site_config={
    "site-1": {
        "train_args": "--data_path /data/iris.csv --train_start 0 --train_end 40 ..."
    },
    "site-2": {
        "train_args": "--data_path /data/iris.csv --train_start 40 --train_end 80 ..."
    },
    # ... more sites
}
```

**Alternative: Using Separate Data Files**

Instead of using data ranges, you can split your data into separate files for each client:

```python
# Split data into files (e.g., using prepare_data.py or pandas)
# - /data/site1_iris.csv
# - /data/site2_iris.csv
# - /data/site3_iris.csv

per_site_config={
    "site-1": {
        "train_args": "--data_path /data/site1_iris.csv ..."
    },
    "site-2": {
        "train_args": "--data_path /data/site2_iris.csv ..."
    },
    # ... more sites
}
```

### View Results

You can use TensorBoard to view the training metrics:
```bash
tensorboard --logdir /tmp/nvflare/simulation/sklearn_kmeans
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
- `job.py`: Defines the federated learning job using the `KMeansFedAvgRecipe`
- `client.py`: Client training script using the NVFlare Client API

The recipe automatically handles:
- Server-side component configuration (controller, aggregator, persistor, KMeansAssembler)
- Client-side executor setup
- Job packaging and deployment

### Advanced: Custom Data Splits

For heterogeneous data splits across clients, you can use the `utils/split_data.py` utility
to generate per-client data ranges and pass them as arguments to the client script.

---

## Results

The resulting curve for `homogeneity_score` shows the clustering quality improving over rounds:

![minibatch curve](./figs/minibatch.png)

You can visualize the metrics using TensorBoard:
```commandline
tensorboard --logdir /tmp/nvflare/simulation/sklearn_kmeans
```

---

## Legacy Approach

> **Note**: This example has been updated to use the simplified Job Recipe API. If you need the previous Job API or JSON-based configuration approach, please refer to the [NVFlare 2.6 documentation](https://github.com/NVIDIA/NVFlare/tree/2.6/examples/advanced/sklearn-kmeans) or earlier versions.
