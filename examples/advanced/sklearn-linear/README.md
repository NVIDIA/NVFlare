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

## Alternative: Run with JSON Configs (Legacy)

For reference, this example also supports the traditional approach using JSON configuration files.
This method requires more setup but provides finer control over individual client configurations.

### Prepare clients' configs with proper data information 
For real-world FL applications, the config JSON files are expected to be 
specified by each client individually, according to their own local data path and splits for training and validation.

In this simulated study, to efficiently generate the config files for a 
study under a particular setting, we provide a script to automate the process. 
Note that manual copying and content modification can achieve the same.

For an experiment with `K` clients, we split one dataset into `K+1` parts in a non-overlapping fashion: `K` clients' training data and `1` common validation data. 
To simulate data imbalance among clients, we provided several options for client data splits by specifying how a client's data amount correlates with its ID number (from `1` to `K`):
- Uniform
- Linear
- Square
- Exponential

These options can be used to simulate no data imbalance (`uniform`), 
moderate data imbalance (`linear`), and high data imbalance (`square` for 
larger client number e.g., `K=20`, exponential for smaller client number e.g., 
`K=5` as it will be too aggressive for larger client numbers)

This step is performed by 
```commandline
bash prepare_job_config.sh
```
In this example, we perform an experiment with five clients under a uniform data split. 

Below is a sample config for site-1, saved to `./jobs/sklearn_linear_5_uniform/app_site-1/config/config_fed_client.json`:
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
                    "learner_id": "linear_learner"
                }
            }
        }
    ],
    "task_result_filters": [],
    "task_data_filters": [],
    "components": [
        {
            "id": "linear_learner",
            "path": "linear_learner.LinearLearner",
            "args": {
                "data_path": "~/dataset/HIGGS.csv",
                "train_start": 1100000,
                "train_end": 3080000,
                "valid_start": 0,
                "valid_end": 1100000,
                "random_state": 0
            }
        }
    ]
}
```

### Run experiment with FL simulator (Legacy)
[FL simulator](https://nvflare.readthedocs.io/en/latest/user_guide/nvflare_cli/fl_simulator.html) is used to simulate FL experiments or debug codes, not for real FL deployment.
We can run the FL simulator with five clients under the uniform data split with
```commandline
bash run_experiment_simulator.sh
```
Note that there will be a warning during training: `ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.`, which is the expected behavior since every round we perform 1-step training on each client. 

## Results

Running with deterministic setting `random_state=0`, the resulting curve 
showing the classification performance using area-under the curve (AUC) is:

![linear curve](./figs/linear.png)

Both the recipe-based approach and the legacy JSON config approach produce the same results.
