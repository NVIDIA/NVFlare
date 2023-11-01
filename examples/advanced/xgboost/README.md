# Federated Learning for XGBoost 

Please make sure you set up virtual environment and Jupyterlab follows [example root readme](../../README.md)

## Introduction to XGBoost and HIGGS Data

You can also follow along in this [notebook](./data_job_setup.ipynb) for an interactive experience.

### XGBoost
These examples show how to use [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) on tabular data applications.
They use [XGBoost](https://github.com/dmlc/xgboost),
which is an optimized distributed gradient boosting library.

### HIGGS
The examples illustrate a binary classification task based on [HIGGS dataset](https://archive.ics.uci.edu/dataset/280/higgs).
This dataset contains 11 million instances, each with 28 attributes.

Please note that the UCI's website may experience occasional downtime.

## Federated Training of XGBoost
Several mechanisms have been proposed for training an XGBoost model in a federated learning setting.
In these examples, we illustrate the use of NVFlare to carry out *horizontal* federated learning using two approaches: histogram-based collaboration and tree-based collaboration.

### Horizontal Federated Learning
Under horizontal setting, each participant / client joining the federated learning will have part of the whole data / instances / examples/ records, while each instance has all the features.
This is in contrast to vertical federated learning, where each client has part of the feature values for each instance.

#### Histogram-based Collaboration
The histogram-based collaboration federated XGBoost approach leverages NVFlare integration of recently added [federated learning support](https://github.com/dmlc/xgboost/issues/7778) in the XGBoost open-source library,
which allows the existing *distributed* XGBoost training algorithm to operate in a federated manner,
with the federated clients acting as the distinct workers in the distributed XGBoost algorithm.

In distributed XGBoost, the individual workers share and aggregate coarse information about their respective portions of the training data,
as required to optimize tree node splitting when building the successive boosted trees.

The shared information is in the form of quantile sketches of feature values as well as corresponding sample gradient and sample Hessian histograms.

Under federated histogram-based collaboration, precisely the same information is exchanged among the clients.

The main differences are that the data is partitioned across the workers according to client data ownership, rather than being arbitrarily partionable, and all communication is via an aggregating federated [gRPC](https://grpc.io) server instead of direct client-to-client communication.

Histograms from different clients, in particular, are aggregated in the server and then communicated back to the clients.

See [histogram-based/README](histogram-based/README.md) for more information on the histogram-based collaboration.

#### Tree-based Collaboration
Under tree-based collaboration, individual trees are independently trained on each client's local data without aggregating the global sample gradient histogram information.
Trained trees are collected and passed to the server / other clients for aggregation and further boosting rounds.

The XGBoost Booster api is leveraged to create in-memory Booster objects that persist across rounds to cache predictions from trees added in previous rounds and retain other data structures needed for training.

See [tree-based/README](tree-based/README.md) for more information on two different types of tree-based collaboration algorithms.


## HIGGS Data Preparation
For data preparation, you can follow this [notebook](./data_job_setup.ipynb):
### Download and Store Data
To run the examples, we first download the dataset from the HIGGS link above, which is a single `.csv` file.
By default, we assume the dataset is downloaded, uncompressed, and stored in `~/dataset/HIGGS.csv`.

> **_NOTE:_** If the dataset is downloaded in another place,
> make sure to modify the corresponding `DATASET_PATH` inside `prepare_data.sh`.

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
bash prepare_data.sh
```

This will generate data splits for three client sizes: 2, 5 and 20, and 3 split conditions: uniform, square, and exponential.
If you want to customize for your experiments, please check `utils/prepare_data_split.py`.

> **_NOTE:_** The generated train config files will be stored in the folder `/tmp/nvflare/xgboost_higgs_dataset/`,
> and will be used by jobs by specifying the path within `config_fed_client.json` 


## HIGGS job configs preparation under various training schemes

Please follow the [Installation](https://nvflare.readthedocs.io/en/main/quickstart.html) instructions to install NVFlare.

We then prepare the NVFlare job configs for different settings by running
```
bash prepare_job_config.sh
```

This script modifies settings from base job configuration
(`./tree-based/jobs/bagging_base` or `./tree-based/jobs/cyclic_base`
or `./histogram-based/jobs/base`),
and copies the correct data split file generated in the data preparation step.

> **_NOTE:_** To customize your own job configs, you can just edit from the generated ones.
> Or check the code in `./utils/prepare_job_config.py`.

The script will generate a total of 10 different configs in `tree-based/jobs` for tree-based algorithm:

- tree-based cyclic training with uniform data split for 5 clients
- tree-based cyclic training with non-uniform data split for 5 clients
- tree-based bagging training with uniform data split and uniform shrinkage for 5 clients
- tree-based bagging training with non-uniform data split and uniform shrinkage for 5 clients
- tree-based bagging training with non-uniform data split and scaled shrinkage for 5 clients
- tree-based cyclic training with uniform data split for 20 clients
- tree-based cyclic training with non-uniform data split for 20 clients
- tree-based bagging training with uniform data split and uniform shrinkage for 20 clients
- tree-based bagging training with non-uniform data split and uniform shrinkage for 20 clients
- tree-based bagging training with non-uniform data split and scaled shrinkage for 20 clients


The script will also generate 2 configs in `histogram-based/jobs` for histogram-base algorithm:

- histogram-based training with uniform data split for 2 clients
- histogram-based training with uniform data split for 5 clients

## Run experiments for tree-based and histogram-based settings
After you run the two scripts `prepare_data.sh` and `prepare_job_config.sh`,
please go to sub-folder [tree-based](tree-based) for running tree-based algorithms,
and sub-folder [histogram-based](histogram-based) for running histogram-based algorithms.


## GPU support
By default, CPU based training is used.

If the CUDA is installed on the site, tree construction and prediction can be
accelerated using GPUs.

To enable GPU accelerated training, in `config_fed_client.json` set `"use_gpus": true` and  `"tree_method": "hist"`. Then, in `FedXGBHistogramExecutor` we use the `device` parameter to map each rank to a GPU device ordinal in `xgb_params`. For a single GPU, assuming it has enough memory, we can map each rank to the same device with `params["device"] = f"cuda:0"`.

### Multi GPU support

Multiple GPUs can be supported by running one NVFlare client for each GPU.

In the `xgb_params`, we can set the `device` parameter to map each rank to a corresponding GPU device ordinal in with `params["device"] = f"cuda:{self.rank}"`

Assuming there are 2 physical client sites, each with 2 GPUs (id 0 and 1).
We can start 4 NVFlare client processes (site-1a, site-1b, site-2a, site-2b), one for each GPU.
The job layout looks like this,
::

    xgb_multi_gpu_job
    ├── app_server
    │   └── config
    │       └── config_fed_server.json
    ├── app_site1_gpu0
    │   └── config
    │       └── config_fed_client.json
    ├── app_site1_gpu1
    │   └── config
    │       └── config_fed_client.json
    ├── app_site2_gpu0
    │   └── config
    │       └── config_fed_client.json
    ├── app_site2_gpu1
    │   └── config
    │       └── config_fed_client.json
    └── meta.json

Each app is deployed to its own client site. Here is the `meta.json`,
::

    {
      "name": "xgb_multi_gpu_job",
      "resource_spec": {
        "site-1a": {
          "num_of_gpus": 1,
          "mem_per_gpu_in_GiB": 1
        },
        "site-1b": {
          "num_of_gpus": 1,
          "mem_per_gpu_in_GiB": 1
        },
        "site-2a": {
          "num_of_gpus": 1,
          "mem_per_gpu_in_GiB": 1
        },
        "site-2b": {
          "num_of_gpus": 1,
          "mem_per_gpu_in_GiB": 1
        }
      },
      "deploy_map": {
        "app_server": [
          "server"
        ],
        "app_site1_gpu0": [
          "site-1a"
        ],
        "app_site1_gpu1": [
          "site-1b"
        ],
        "app_site2_gpu0": [
          "site-2a"
        ],
        "app_site2_gpu1": [
          "site-2b"
        ]
      },
      "min_clients": 4
    }

For federated XGBoost, all clients must participate in the training. Therefore,
`min_clients` must equal to the number of clients.

