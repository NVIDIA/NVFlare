# Federated Learning for Random Forest based on XGBoost 

## Introduction to XGBoost and HIGGS Data

### XGBoost
These examples show how to use [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) on tabular data applications.
It illustrates the [Random Forest](https://xgboost.readthedocs.io/en/stable/tutorials/rf.html) algorithm using [XGBoost](https://github.com/dmlc/xgboost),
which is an optimized distributed gradient boosting library covering random forest.

### HIGGS
The examples illustrate a binary classification task based on [HIGGS dataset](https://archive.ics.uci.edu/ml/datasets/HIGGS).
This dataset contains 11 million instances, each with 28 attributes.

## Federated Training of Random Forest using XGBoost
In this example, we illustrate the use of NVFlare to carry out *horizontal* federated learning with tree-based collaboration.

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


## HIGGS Data Preparation
### Download and Store Data
To run the examples, we first download the dataset from the HIGGS link above, which is a single `.csv` file.
By default, we assume the dataset is downloaded, uncompressed, and stored in `~/dataset/HIGGS.csv`.

> **_NOTE:_** If the dataset is downloaded in another place,
> make sure to modify the corresponding `DATASET_PATH` inside `data_split_gen.sh`.

### Data Split
Since HIGGS dataset is already randomly recorded,
data split will be specified by the continuous index ranges for each client,
rather than a vector of random instance indices.
We provide four options to split the dataset to simulate the non-uniformity in data quantity: 

1. uniform: all clients has the same amount of data 
2. linear: the amount of data is linearly correlated with the client ID (1 to M)
3. square: the amount of data is correlated with the client ID in a squared fashion (1^2 to M^2)
4. exponential: the amount of data is correlated with the client ID in an exponential fashion (exp(1) to exp(M))

The choice of data split depends on dataset and the number of participants.

For a large dataset like HIGGS, if the number of clients is small (e.g. 5),
each client will still have sufficient data to train on with uniform split,
and hence exponential would be used to observe the performance drop caused by non-uniform data split.
If the number of clients is large (e.g. 20), exponential split will be too aggressive, and linear/square should be used.

Data splits used in this example can be generated with
```
bash data_split_gen.sh
```

This will generate data splits for three client sizes: 2, 5 and 20, and 3 split conditions: uniform, square, and exponential.
If you want to customize for your experiments, please check `utils/prepare_data_split.py`.

> **_NOTE:_** The generated train config files will be stored in the folder `/tmp/nvflare/xgboost_higgs_dataset/`,
> and will be used by job_configs by specifying the path within `config_fed_client.json` 


## HIGGS job configs preparation under various training settings

Please follow the [Installation](https://nvflare.readthedocs.io/en/main/quickstart.html) instructions to install NVFlare.

We then prepare the NVFlare job configs for different settings by running
```
bash job_config_gen.sh
```

This script modifies settings from base job configuration
(`./job_configs/random_forest_base`),
and copies the correct data split file generated in the data preparation step.

> **_NOTE:_** To customize your own job configs, you can just edit from the generated ones.
> Or check the code in `./utils/prepare_job_config.py`.

The script will generate a total of 18 different configs in `./job_configs` for random forest algorithm with different data split, client number, local tree number, and local subsample rate.

## GPU support
By default, CPU based training is used.

If the CUDA is installed on the site, tree construction and prediction can be
accelerated using GPUs.

GPUs are enabled by using :code:`gpu_hist` as :code:`tree_method` parameter.
For example,
::
              "xgboost_params": {
                "max_depth": 8,
                "eta": 0.1,
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "tree_method": "gpu_hist",
                "gpu_id": 0,
                "nthread": 16
              }

For GPU based training, edit `job_config_gen.sh` to change `TREE_METHOD="hist"` to `TREE_METHOD="gpu_hist"`.
Then run the `job_config_gen.sh` again to generates new job configs for GPU-based training.

## Run experiments 
After you run the two scripts `data_split_gen.sh` and `job_config_gen.sh`, the experiments can be run with the NVFlare simulator.
```
bash run_experiment_simulator.sh
```

