# Financial Application with Federated XGBoost Methods
This example illustrates the use of [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) on a financial application. 
These examples show how to use [XGBoost](https://github.com/dmlc/xgboost) in various ways to train a model in a federated manner to perform fraud detection with a 
[finance dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Federated Training of XGBoost
Several mechanisms have been proposed for training an XGBoost model in a federated learning setting.
In these examples, we illustrate the use of NVFlare to carry out the following four approaches:
- *vertical* federated learning using histogram-based collaboration
- *horizontal* federated learning using three approaches: 
  - histogram-based collaboration 
  - tree-based collaboration with cyclic federation
  - tree-based collaboration with bagging federation

For more details, please refer to the README files for 
[vertical](https://github.com/NVIDIA/NVFlare/blob/main/examples/advanced/vertical_xgboost/README.md), 
[histogram-based](https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/xgboost/histogram-based/README.md),
and [tree-based](https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/xgboost/tree-based/README.md) 
methods.

## Data Preparation
### Download and Store Data
To run the examples, we first download the dataset from the link above, which is a single `.csv` file.
By default, we assume the dataset is downloaded, uncompressed, and stored in `${PWD}/dataset/creditcard.csv`.

> **_NOTE:_** If the dataset is downloaded in another place,
> make sure to modify the corresponding `DATASET_PATH` inside `prepare_data.sh`.

### Data Split
We first split the dataset into two parts: training and testing. Then perform data split for each client under both horizontal and vertical settings.

Data splits used in this example can be generated with
```
bash prepare_data.sh
```

This will generate data splits for 2 clients under all experimental settings. Note that the overlapping ratio between clients for vertical setting is 1.0 by default, so that the training data amount is the same as horizontal experiments.
If you want to customize for your experiments to simulate more realistic scenarios, please check their corresponding scripts under `utils/`.

> **_NOTE:_** The generated data files will be stored in the folder `/tmp/dataset/`,
> and will be used by jobs by specifying the path within `config_fed_client.json` 

## Run experiments for all settings
To run all experiments, we provide a script for all settings.
```
bash run_training.sh
```
This will cover baseline centralized training, horizontal FL with histogram-based, tree-based cyclic, and tree-based bagging
collaborations, as well as vertical FL.

Then, we test the resulting models on the test dataset using 
```
bash run_testing.sh
``` 
The results are as follows:
```
Testing baseline_xgboost
AUC score:  0.965017768854869
Testing xgboost_vertical
AUC score:  0.9650650531737737
Testing xgboost_horizontal_histogram
AUC score:  0.9579533839422094
Testing xgboost_horizontal_cyclic
AUC score:  0.9688269828190139
Testing xgboost_horizontal_bagging
AUC score:  0.9713936151275366
```