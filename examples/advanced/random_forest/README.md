# Federated Learning for Random Forest based on XGBoost 

Please make sure you set up virtual environment and Jupyterlab follows [example root readme](../../README.md)

## Quick Start with Recipe API (Recommended)

The simplest way to run this example is using NVFlare's **Recipe API**, which provides a high-level, simplified interface:

### 1. Data Preparation
First, prepare the HIGGS dataset splits:

```bash
bash data_split_gen.sh DATASET_ROOT
```

Replace `DATASET_ROOT` with the path to your HIGGS dataset directory. This will generate data splits in `/tmp/nvflare/random_forest/HIGGS/data_splits`.

### 2. Run with Recipe

```bash
python job.py --n_clients 5 --local_subsample 0.5 --data_split_path /tmp/nvflare/random_forest/HIGGS/data_splits/5_uniform
```

**Key arguments:**
- `--n_clients`: Number of federated clients (default: 5)
- `--num_local_parallel_tree`: Number of parallel trees per client (default: 5)
- `--local_subsample`: Subsample ratio for local training (default: 0.5)
- `--data_split_path`: Path to data split directory

**What the Recipe does:**
- Uses `XGBBaggingRecipe` for tree-based federated Random Forest
- Each client trains a local sub-forest on their data
- Server aggregates all sub-forests to form the global model
- Runs in simulation environment by default

### 3. Enable GPU Support (Optional)

To use GPUs for training:

```bash
python job.py --use_gpus --tree_method hist
```

---

## Introduction to Libraries and HIGGS Data

### Libraries
This example show how to use [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) on tabular data applications.
It illustrates the [Random Forest](https://xgboost.readthedocs.io/en/stable/tutorials/rf.html) functionality using [XGBoost] (https://github.com/dmlc/xgboost) library,
which is an optimized distributed gradient boosting library covering random forest.
Follow along in this [notebook](./random_forest.ipynb) for an interactive experience.

### Dataset
This example illustrate a binary classification task based on [HIGGS dataset](https://mlphysics.ics.uci.edu/data/higgs/).
This dataset contains 11 million instances, each with 28 attributes.

Please note that the UCI's website may experience occasional downtime.

## Federated Training of Random Forest using XGBoost
In this example, we illustrate the use of NVFlare to carry out *horizontal* federated learning with tree-based collaboration - forming a random forest.

### Horizontal Federated Learning
Under horizontal setting, each participant / client joining the federated learning will have part of the whole data / instances / examples/ records, while each instance has all the features.
This is in contrast to vertical federated learning, where each client has part of the feature values for each instance.

### Tree-based Collaboration
Under tree-based collaboration, individual trees are independently trained on each client's local data without aggregating the global sample gradient histogram information.
Trained trees are collected and passed to the server / other clients for aggregation. Note that under Random Forest setting, only one round of training will be performed.

### Local Training and Aggregation
Random forest training with multiple clients can be achieved in two steps:

- Local training: each site train a local sub-forest consisting of a number of trees based on their local data by utilizing the `subsample` and `num_parallel_tree` functionalities from XGBoost. 
- Global aggregation: server collects all sub-forests from clients, and a bagging aggregation scheme is applied to generate the global forest model.

No further training will be performed, `num_boost_round` should be 1 to align with the basic setting of random forest.

---

## Results

Federated Random Forest using XGBoost bagging produces competitive AUC scores. The model quality 
depends on hyperparameters like `local_subsample` and `num_local_parallel_tree`.

Example AUC results on HIGGS dataset (first 1M instances for validation):
- With local_subsample=0.5: AUC ≈ 0.77-0.78
- With local_subsample=0.05: AUC ≈ 0.78-0.82
- With local_subsample=0.005: AUC ≈ 0.78-0.83

Lower subsample rates generally produce better models but require more local data per client.

---

## Legacy Approach

> **Note**: This example has been updated to use the simplified Job Recipe API. If you need the previous JSON-based configuration approach with automated job generation scripts, please refer to the [NVFlare 2.6 documentation](https://github.com/NVIDIA/NVFlare/tree/2.6/examples/advanced/random_forest) or earlier versions.
