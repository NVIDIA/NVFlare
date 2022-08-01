# Federated Learning for XGBoost 

## Introduction to XGBoost and HIGGS Data

### XGBoost
This example shows how to use [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) on tabular data applications.
It uses [XGBoost](https://github.com/dmlc/xgboost),
which is an optimized distributed gradient boosting library.

### HIGGS
This example illustrates a binary classification task based on [HIGGS dataset](https://archive.ics.uci.edu/ml/datasets/HIGGS). This dataset contains 11 million instances, each with 28 attributes.

## Data Preparation
To run the examples, we first download the dataset, which is a single `.csv` file. The dataset will be downloaded, uncompressed, and stored under `./dataset` as `./dataset/HIGGS.csv`.

### Data Split
Since HIGGS dataset is already randomly recorded, data split will be specified by the continuous index ranges for each client, rather than a vector of random instance indices.

## Environment Preparation
### Install NVIDIA FLARE
Follow the [Installation](https://nvflare.readthedocs.io/en/main/quickstart.html) instructions.
Install additional requirements:
```
pip3 install tensorflow
pip3 install xgboost
```

### Set up FL workspace

Follow the [Quickstart](https://nvflare.readthedocs.io/en/main/quickstart.html) instructions to set up your POC ("proof of concept") workspace.

## Federated Training of XGBoost under Various Schemes
Please go to subfolders [./lossless](./lossless), [./cyclic](./cyclic), and [./bagging](./bagging) for further instructions on each federated training scheme.
