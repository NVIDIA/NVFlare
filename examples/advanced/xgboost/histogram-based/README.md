# Histogram-based Federated Learning for XGBoost   

## Run automated experiments
Please make sure to finish the [preparation steps](../README.md) before running the following steps.
To run this example with NVFlare, follow the steps below or this [notebook](./xgboost_histogram_higgs.ipynb) for an interactive experience.

### Environment Preparation

Switch to this directory and install additional requirements (suggest to do this inside virtual environment):
```
python3 -m pip install -r requirements.txt
```

### Run federated experiments with simulator locally
Next, we will use the NVFlare simulator to run FL training automatically.
```
bash run_experiment_simulator.sh
```

### Run centralized experiments
```
bash run_experiment_centralized.sh
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
implement that `load_data()` method.
