# Histogram-based Federated Learning for XGBoost   

## Run automated experiments
To run this example with NVFlare, follow the below steps.

### Environment Preparation

1. Install NVFlare: Follow the [Installation](https://nvflare.readthedocs.io/en/main/quickstart.html) instructions.
2. Clone xgboost: `git clone https://github.com/dmlc/xgboost xgboost`
3. Init xgboost submodule: `cd xgboost; git submodule update --init`
4. Build and install xgboost-federated following [README](https://github.com/dmlc/xgboost/blob/master/plugin/federated/README.md)

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
start the overseer, fl server and fl clients.

You can still generate the data splits and job configs using the scripts provided.

Note that after you generate the job config, you might need to modify the `data_path` in the `data_split_XXX.json`
inside the `app/config` folder of each job, since each site might save the HIGGS dataset in different places.

One way is that you can copy the app and modify it for each site, for example the job will be:

```commandline
higgs_2_histogram_uniform_split_uniform_lr/
    app_server/
        config/
            config_fed_server.json
    app_site-1/
        config/
            config_fed_client.json
            data_split_2_uniform.json
        custom/
            higgs_executor.py
    app_site-2/
        config/
            config_fed_client.json
            data_split_2_uniform.json
        custom/
            higgs_executor.py
```

The meta.json can be modified to be:

```json
{
    "name": "higgs_2_histogram_uniform_split_uniform_lr",
    "resource_spec": {},
    "deploy_map": {
        "app_server": [
            "server"
        ],
        "app_site-1": [
            "site-1"
        ],
        "app_site-2": [
            "site-2"
        ]
    },
    "min_clients": 2
}
```

Then you can use admin client to submit the job via `submit_job` command.
