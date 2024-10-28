# Federated Learning for Random Forest based on XGBoost 

Please make sure you set up virtual environment and Jupyterlab follows [example root readme](../../README.md)

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


## Data Preparation
### Download and Store Data
To run the examples, we first download the dataset from the HIGGS link above, which is a single `HIGGS.csv` file.
By default, we assume the dataset is downloaded, uncompressed, and stored in `DATASET_ROOT/HIGGS.csv`.


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
bash data_split_gen.sh DATASET_ROOT
```
> **_NOTE:_** make sure to put the correct path for `DATASET_ROOT`.

This will generate data splits for two client sizes: 5 and 20, and 3 split conditions: uniform, square, and exponential.
If you want to customize for your experiments, please check `utils/prepare_data_split.py`.

> **_NOTE:_** The generated train config files will be stored in the folder `/tmp/nvflare/random_forest/HIGGS/data_splits`,
> and will be used by jobs by specifying the path within `config_fed_client.json` 


## HIGGS jobs preparation under various training settings

Please follow the [Installation](../../getting_started/README.md) instructions to install NVFlare.

We then prepare the NVFlare jobs for different settings by running
```
bash jobs_gen.sh
```

This script modifies settings from base job configuration
(`./jobs/random_forest_base`),
and copies the correct data split file generated in the data preparation step.

> **_NOTE:_** To customize your own job configs, you can just edit from the generated ones.
> Or check the code in `./utils/prepare_job_config.py`.

The script will generate a total of 18 different configs in `./jobs` for random forest algorithm with different data split, client number, local tree number, and local subsample rate.

## GPU support
By default, CPU based training is used.

If the CUDA is installed on the site, tree construction and prediction can be
accelerated using GPUs.

In order to enable GPU accelerated training, first ensure that your machine has CUDA installed and has at least one GPU.
In `config_fed_client.json` set `"use_gpus": true` and  `"tree_method": "hist"`.
Then, in `FedXGBTreeExecutor` we use the `device` parameter to map each rank to a GPU device ordinal.
If using multiple GPUs, we can map each rank to a different GPU device, however you can also map each rank to the same GPU device if using a single GPU.

## Run experiments 
After you run the two scripts `data_split_gen.sh` and `jobs_gen.sh`, the experiments can be run with the NVFlare simulator.
```
bash run_experiment_simulator.sh
```

## Validate the trained model
The trained global random forest model can further be validated using
```
bash model_validation.sh 
```
The output is 
```
5_clients_uniform_split_uniform_lr_split_0.5_subsample
AUC over first 1000000 instances is: 0.7739160662860319
5_clients_exponential_split_uniform_lr_split_0.5_subsample
AUC over first 1000000 instances is: 0.7770108010299946
5_clients_exponential_split_scaled_lr_split_0.5_subsample
AUC over first 1000000 instances is: 0.7727404041835751
5_clients_uniform_split_uniform_lr_split_0.05_subsample
AUC over first 1000000 instances is: 0.7810306437097397
5_clients_exponential_split_uniform_lr_split_0.05_subsample
AUC over first 1000000 instances is: 0.7821852372076727
5_clients_exponential_split_scaled_lr_split_0.05_subsample
AUC over first 1000000 instances is: 0.7787293667285318
5_clients_uniform_split_uniform_lr_split_0.005_subsample
AUC over first 1000000 instances is: 0.7825158041290863
5_clients_exponential_split_uniform_lr_split_0.005_subsample
AUC over first 1000000 instances is: 0.7797689977305647
5_clients_exponential_split_scaled_lr_split_0.005_subsample
AUC over first 1000000 instances is: 0.7817849808015369
20_clients_uniform_split_uniform_lr_split_0.8_subsample
AUC over first 1000000 instances is: 0.7768044721524759
20_clients_square_split_uniform_lr_split_0.8_subsample
AUC over first 1000000 instances is: 0.7788396083758498
20_clients_square_split_scaled_lr_split_0.8_subsample
AUC over first 1000000 instances is: 0.7757795325989565
20_clients_uniform_split_uniform_lr_split_0.2_subsample
AUC over first 1000000 instances is: 0.7808745447533768
20_clients_square_split_uniform_lr_split_0.2_subsample
AUC over first 1000000 instances is: 0.7817719906970683
20_clients_square_split_scaled_lr_split_0.2_subsample
AUC over first 1000000 instances is: 0.7797564888670013
20_clients_uniform_split_uniform_lr_split_0.02_subsample
AUC over first 1000000 instances is: 0.7828698775310959
20_clients_square_split_uniform_lr_split_0.02_subsample
AUC over first 1000000 instances is: 0.779952094937354
20_clients_square_split_scaled_lr_split_0.02_subsample
AUC over first 1000000 instances is: 0.7825360505137948
```
