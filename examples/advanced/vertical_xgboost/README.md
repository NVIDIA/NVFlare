# Vertical Federated XGBoost
This example shows how to use vertical federated learning with [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) on tabular data.
Here we use the optimized gradient boosting library [XGBoost](https://github.com/dmlc/xgboost) and leverage its federated learning plugin.

Before starting please make sure you set up a [virtual environment](../../../README.md#set-up-a-virtual-environment) and install the additional requirements:
```
python3 -m pip install -r requirements.txt
```

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
Since not every site will have the same set of data samples, we can use PSI to compare encrypted versions of clients datasets in order to jointly compute the intersection based on common IDs. In this example, the HIGGS dataset does not contain unique identifiers so we add a temporary `uid_{idx}` to each instance and give each site a portion of the HIGGS dataset (rows) that includes a common overlap. Afterwards the identifiers are dropped since they are only used for matching, and training is then done on the intersected data. To learn more about our PSI protocol implementation, see our [psi example](../psi/README.md).

> **_NOTE:_** The uid can be a composition of multiple variabes with a transformation, however in this example we use indices for simplicity. PSI can also be used for computing the intersection of overlapping features, but here we give each site unique features.

## XGBoost Federated Learning Plugin with FLARE

### Installation

Since the XGBoost federated learning plugin is not available in the official XGBoost releases yet, we must build XGBoost with the plugin from the source code.
First clone the repo:
```
git clone --recursive https://github.com/dmlc/xgboost
```
Next, follow the steps [here](https://github.com/dmlc/xgboost/blob/master/plugin/federated/README.md) to install gRPC, build XGBoost with the federated plugin enabled, and install the python package into your virtual environemnt.

### Integration with FLARE
The plugin allows the XGBoost internal training parameter `DataSplitMode` to be set to column split mode rather than row split mode for distributed training. Additionally, all communication is done through the federated communicator, such as broadcasting split results and gradients.

Information about the federated communicator is defined in the XGBoost communicator environment:

          communicator_env = {
              'xgboost_communicator': 'federated',
              'federated_server_address': self._server_address,
              'federated_world_size': self._world_size,
              'federated_rank': rank,
              "federated_server_cert": self._server_cert_path,
              "federated_client_key": self._client_key_path,
              "federated_client_cert": self._client_cert_path
          }

For integrating with FLARE, a controller can start the gRPC federated server with `xgboost.federated.run_federated_server` and broadcast the train task to the clients. The DMatrix `data_split_mode` is set to `1` for column split mode, and `?format=csv&label_column=0` is used when creating the DMatrix from a csv file to specifiy which column contains the labels. Since only one site contains the labels, other site will not specifiy a label column `?format=csv`. Lastly, booster params for the training are set in the `xgb_params` dict, however not all `tree_methods` are currently supported with the plugin.

> **_NOTE:_** For secure mode, make sure to pass the server certificate, client key, and client certificate into the communicator environment and into the federated server. As of now, GPUs are not yet supported by vertical federated XGBoost.

## Run the Example
Run a job using the simulator (modify number of clients and other arguments in `prepare_data.sh` and run beforehand as desired):
```
nvflare simulator ./jobs/vertical_xgboost_2 -w /tmp/nvflare/workspaces/vertical_xgboost -n 2 -t 2
```

The model will be saved to `higgs.model.federated.vertical.json` in the run directory.

## Results
Model accuracy can be visualized in tensorboard:
```
tensorboard --logdir /tmp/nvflare/vertical_xgboost
```

An example training (pink) and validation (orange) AUC graph from running vertical XGBoost on HIGGS.
Used an intersection of 50000 samples across 5 clients each with different features, and ran for ~50 rounds due to early stopping.

![Vertical XGBoost graph](./figs/vertical_xgboost_graph.png)
