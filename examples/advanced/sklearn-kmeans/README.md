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
- `src/kmeans_fl.py`: Client training script using the NVFlare Client API

The recipe automatically handles:
- Server-side component configuration (controller, aggregator, persistor, KMeansAssembler)
- Client-side executor setup
- Job packaging and deployment

### Advanced: Custom Data Splits

For heterogeneous data splits across clients, you can use the `utils/split_data.py` utility
to generate per-client data ranges and pass them as arguments to the client script.

---

## Alternative: Run FL with Job API (Legacy)

For reference, this example also supports the traditional approach using the Job API
with manual data splitting. 
For real-world FL applications, the config JSON files are expected to be 
specified by each client individually, according to their own local data path and splits for training and validation.

In this simulated study, we generate automatic data split and run experiments with different data heterogeneity levels.

For an experiment with `K` clients, we split one dataset into `K+1` parts in a non-overlapping fashion: 
`K` clients' training data and `1` common validation data. 

To simulate data imbalance among clients, we provided several options for client data splits by specifying how a client's data amount correlates with its ID number (from `1` to `K`):
- Uniform
- Linear
- Square
- Exponential

These options can be used to simulate no data imbalance (uniform), moderate 
data imbalance (linear), and high data imbalance (square for larger client 
number, e.g. `K=20`, exponential for smaller client number, e.g. `K=5` as 
it will be too aggressive for a larger number of clients)

### Legacy Job API Example

In this example, we experiment with 3 clients under a uniform data split. 
We run the federated training using NVFlare Simulator via [JobAPI](https://nvflare.readthedocs.io/en/main/programming_guide/fed_job_api.html):
```commandline
python kmeans_job.py --num_clients 3 --split_mode uniform
```

Note: The above approach is preserved for reference. The recommended approach is to use the recipe method described earlier.

Below is a sample config for site-1, saved to `/tmp/nvflare/workspace/jobs/kmeans/sklearn_kmeans_uniform_3_clients/app_site-1/config/config_fed_client.json`:
```json
{
  "format_version": 2,
  "executors": [
    {
      "tasks": [
        "train"
      ],
      "executor": {
        "id": "Executor",
        "path": "nvflare.app_opt.sklearn.sklearn_executor.SKLearnExecutor",
        "args": {
          "learner_id": "kmeans_learner"
        }
      }
    }
  ],
  "task_result_filters": [],
  "task_data_filters": [],
  "components": [
    {
      "id": "kmeans_learner",
      "path": "kmeans_learner.KMeansLearner",
      "args": {
        "data_path": "/tmp/nvflare/dataset/sklearn_iris.csv",
        "train_start": 0,
        "train_end": 50,
        "valid_start": 0,
        "valid_end": 150,
        "random_state": 0
      }
    }
  ]
}
```

Alternative to using Learner+Executor as above, we can also use [ClientAPI](https://nvflare.readthedocs.io/en/2.6/programming_guide/execution_api_type/client_api.html) 
to run the federated training:
```commandline
python kmeans_job_clientapi.py --num_clients 3 --split_mode uniform --workspace_dir "/tmp/nvflare/workspace/works/kmeans_clientapi" --job_dir "/tmp/nvflare/workspace/jobs/kmeans_clientapi"
```

## Results

The resulting curve for `homogeneity_score` is:

![minibatch curve](./figs/minibatch.png)

Both the recipe-based approach and the legacy Job API approach produce the same results.

You can visualize the metrics using TensorBoard:
```commandline
# For recipe approach
tensorboard --logdir /tmp/nvflare/simulation/sklearn_kmeans

# For legacy approach
tensorboard --logdir /tmp/nvflare/workspace/works/kmeans/sklearn_kmeans_uniform_3_clients
```