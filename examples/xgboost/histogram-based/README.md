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

Note that after you generate the job config,
you might need to modify the `data_path` in the `data_site-XXX.json`
inside the `/tmp/nvflare/xgboost_higgs_dataset` folder,
since each site might save the HIGGS dataset in different places.

Then you can use admin client to submit the job via `submit_job` command.
