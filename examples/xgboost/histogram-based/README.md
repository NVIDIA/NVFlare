# Histogram-based Federated Learning for XGBoost   

## Run automated experiments
To run this example with NVFlare, follow the below steps.

### Environment Preparation

1. Install NVFlare: Follow the [Installation](https://nvflare.readthedocs.io/en/main/quickstart.html) instructions.
2. Clone xgboost: `git clone https://github.com/dmlc/xgboost xgboost`
3. Init xgboost submodule: `cd xgboost; git submodule update --init`
4. Build and install xgboost-federated following [README](https://github.com/dmlc/xgboost/blob/master/plugin/federated/README.md)

### Run local experiments with simulator
Next, we will use the NVFlare simulator to run FL training automatically.
```
bash run_experiment_simulator.sh
```

## Centralized training
```
bash run_experiment_centralized.sh
```
