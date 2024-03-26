# Histogram-based Federated Learning for XGBoost   

## Run automated experiments
Please make sure to finish the [preparation steps](../README.md) before running the following steps.
To run this example with NVFlare, follow the steps below or this [notebook](./xgboost_histogram_higgs.ipynb) for an interactive experience.

### Environment Preparation

Switch to this directory and install additional requirements (suggest to do this inside virtual environment):
```
python3 -m pip install -r requirements.txt
```

### Run centralized experiments
```
bash run_experiment_centralized.sh
```

### Run federated experiments with simulator locally
Next, we will use the NVFlare simulator to run FL training automatically.
```
nvflare simulator jobs/higgs_2_histogram_v2_uniform_split_uniform_lr \
   -w /tmp/nvflare/xgboost_v2_workspace -n 2 -t 2
```

Model accuracy can be visualized in tensorboard:
```
tensorboard --logdir /tmp/nvflare/xgboost_v2_workspace/simulate_job/tb_events
```

### Run federated experiments in real world

To run in a federated setting, follow [Real-World FL](https://nvflare.readthedocs.io/en/main/real_world_fl.html) to
start the overseer, FL servers and FL clients.

You need to download the HIGGS data on each client site.
You will also need to install the xgboost on each client site and server site.

You can still generate the data splits and job configs using the scripts provided.

You will need to copy the generated data split file into each client site.
You might also need to modify the `data_path` in the `data_site-XXX.json`
inside the `/tmp/nvflare/xgboost_higgs_dataset` folder,
since each site might save the HIGGS dataset in different places.

Then you can use admin client to submit the job via `submit_job` command.

## Customization

The provided XGBoost executor can be customized using Boost parameters
provided in `xgb_params` argument.

If the parameter change alone is not sufficient and code changes are required,
a custom executor can be implemented to make calls to xgboost library directly.

The custom executor can inherit the base class `FedXGBHistogramExecutor` and
overwrite the `xgb_train()` method.

To use other dataset, can inherit the base class `XGBDataLoader` and
implement the `load_data()` method.

## GPU support
By default, CPU based training is used.

If the CUDA is installed on the site, tree construction and prediction can be
accelerated using GPUs.

To enable GPU accelerated training, in `config_fed_client` set the args of 
`FedXGBHistogramExecutor` to `"use_gpus": true` and set `"tree_method": "hist"`
in `xgb_params`.

We can also use the `device` parameter to map each rank to a GPU device ordinal in `xgb_params`.
For a single GPU, assuming it has enough memory, we can map each rank to the same device with `xgb_params["device"] = f"cuda:0"`.

### Multi GPU support

Multiple GPUs can be supported by running one NVFlare client for each GPU.

In the `xgb_params`, we can set the `device` parameter to map each rank to a corresponding GPU
device ordinal in with `xgb_params["device"] = f"cuda:{self.rank}"`

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

## Loose integration

We can use the NVFlare controller/executor just to launch the external xgboost
federated server and client.

### Run federated experiments with simulator locally
Next, we will use the NVFlare simulator to run FL training automatically.
```
nvflare simulator jobs/higgs_2_histogram_uniform_split_uniform_lr \
   -w /tmp/nvflare/xgboost_workspace -n 2 -t 2
```

Model accuracy can be visualized in tensorboard:
```
tensorboard --logdir /tmp/nvflare/xgboost_workspace/simulate_job/tb_events
```
