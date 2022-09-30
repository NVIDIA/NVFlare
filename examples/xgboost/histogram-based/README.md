# Histogram-based Federated Learning for XGBoost   

## Run automated experiments
To run this example with NVFlare, follow the below steps.

### Environment Preparation

1. Clone xgboost: `git clone https://github.com/dmlc/xgboost xgboost`
2. Init submodule: `cd xgboost; git submodule update --init`
3. Build and install xgboost-federated following [README](https://github.com/dmlc/xgboost/blob/master/plugin/federated/README.md)

### Run local experiments with simulator
Next, we will use the NVFlare simulator to run FL training automatically.
```
bash run_experiment_simulator.sh
```

## Results on 5- and 20-client under various training settings
For comparison, we train baseline models in a centralized manner with same round of training
```
bash run_experiment_centralized.sh
```
