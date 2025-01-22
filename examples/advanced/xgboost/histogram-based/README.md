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
You will also need to install XGBoost on each client site and server site.

You can still generate the data splits and job configs using the scripts provided.

You will need to copy the generated data split file into each client site.
You might also need to modify the `data_path` in the `data_site-XXX.json`
inside the `/tmp/nvflare/xgboost_higgs_dataset` folder,
since each site might save the HIGGS dataset in different places.

Then, you can use the admin client to submit the job via the `submit_job` command.

## Customization

The provided XGBoost executor can be customized using boost parameters
provided in the `xgb_params` argument.

If the parameter change alone is not sufficient and code changes are required,
a custom executor can be implemented to make calls to xgboost library directly.

The custom executor can inherit the base class `FedXGBHistogramExecutor` and
overwrite the `xgb_train()` method.

To use a different dataset, you can inherit the base class `XGBDataLoader` and
implement the `load_data()` method.

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
