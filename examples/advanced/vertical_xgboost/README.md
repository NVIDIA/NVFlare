# Vertical Federated XGBoost
This example shows how to use vertical federated learning with [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) on tabular data.
Here we use the optimized gradient boosting library [XGBoost](https://github.com/dmlc/xgboost) and leverage its federated learning plugin.

Before starting please make sure you set up a [virtual environment](../../../README.md#set-up-a-virtual-environment) and install the additional requirements:
```
python3 -m pip install -r requirements.txt
```

> **_NOTE:_** If the vertical federated learning plugin is not available in the XGBoost PyPI release yet, reinstall XGBoost from a [wheel](https://xgboost.readthedocs.io/en/stable/install.html#nightly-build) with a recent commit.

## Preparing HIGGS Data
In this example we showcase a binary classification task based on the [HIGGS dataset](https://archive.ics.uci.edu/dataset/280/higgs), which contains 11 million instances, each with 28 features and 1 class label.

### Download and Store Dataset
We first download the dataset from the HIGGS link above, which is a single zipped `.csv` file.
By default, we assume the dataset is downloaded, uncompressed, and stored in `~/dataset/HIGGS.csv`.

### Vertical Data Splits
In vertical federated learning, sites share overlapping data samples (rows), but contain different features (columns).
In order to achieve this, we split the HIGGS dataset both horizontally and vertically. As a result, each site has an overlapping subset of the rows and a  subset of the 29 columns.

<img src="./figs/vertical_fl.png" alt="vertical fl diagram" width="500"/>

Run the following commands to prepare the data splits and job configurations:
```
./prepare_data.sh
./prepare_job_config.sh
```

### Private Set Intersection (PSI)
Since not every site will have the same set of data samples, we can use PSI to compare encrypted versions of the sites' datasets in order to jointly compute the intersection based on common IDs. In this example, the HIGGS dataset does not contain unique identifiers so we add a temporary `uid_{idx}` to each instance and give each site a portion of the HIGGS dataset (rows) that includes a common overlap. Afterwards the identifiers are dropped since they are only used for matching, and training is then done on the intersected data. To learn more about our PSI protocol implementation, see our [psi example](../psi/README.md).

> **_NOTE:_** The uid can be a composition of multiple variabes with a transformation, however in this example we use indices for simplicity. PSI can also be used for computing the intersection of overlapping features, but here we give each site unique features.

## Vertical XGBoost Federated Learning Plugin with FLARE

This Vertical XGBoost example leverages the recently added [vertical federated learning support](https://github.com/dmlc/xgboost/issues/8424) in the XGBoost open-source library. The plugin allows for the distributed XGBoost algorithm to operate in a federated manner on vertically split data.

For integrating with FLARE, we can use the predefined `XGBFedController` to run the federated server and control the workflow.


Next we can subclass `FedXGBHistogramExecutor` to write our XGBoost training code in the `xgb_train()` method, and subclass `XGBDataLoader` to implement the `load_data()` method. For vertical federated learning, it is important when creating the `xgb.Dmatrix` to set `data_split_mode=1` for column mode, and to specify the presence of a label column `?format=csv&label_column=0` for the csv file.

> **_NOTE:_** For secure mode, make sure to provide the required certificates for the federated communicator. As of now, GPUs are not yet supported by vertical federated XGBoost.

## Run the Example
Run a job using the simulator (feel free to modify the number of clients and other arguments in `prepare_data.sh` and `prepare_job_config.sh` and run beforehand as desired):
```
nvflare simulator ./jobs/vertical_xgboost_2 -w /tmp/nvflare/workspaces/vertical_xgboost -n 2 -t 2
```

The model will be saved to `test.model.json`.

## Results
Model accuracy can be visualized in tensorboard:
```
tensorboard --logdir /tmp/nvflare/vertical_xgboost
```

An example training (pink) and validation (orange) AUC graph from running vertical XGBoost on HIGGS.
Used an intersection of 50000 samples across 5 clients each with different features, and ran for ~50 rounds due to early stopping.

![Vertical XGBoost graph](./figs/vertical_xgboost_graph.png)
