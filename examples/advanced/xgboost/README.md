# Federated Learning for XGBoost 
This example demonstrates how to use NVFlare to train an XGBoost model in a federated learning setting. 
Several potential variations of federated XGBoost are illustrated, including:
- non-secure horizontal collaboration with histogram-based and tree-based mechanisms.
- non-secure vertical collaboration with histogram-based mechanism.
- secure horizontal and vertical collaboration with histogram-based mechanism and homomorphic encryption.

To run the examples and notebooks, please make sure you set up a virtual environment and Jupyterlab, following [the example root readme](../../README.md)
and install the additional requirements:
```
python3 -m pip install -r requirements.txt
```
Also, follow [encryption plugins instruction](https://github.com/NVIDIA/NVFlare/tree/main/integration/xgboost/encryption_plugins) to setup encryption plugins.

## XGBoost 
XGBosot is a machine learning algorithm that uses decision/regression trees to perform classification and regression tasks, 
mapping a vector of feature values to its label prediction. It is especially powerful for tabular data, so even in the age of LLM, 
it still widely used for many tabular data use cases. It is also preferred for its explainability and efficiency.

In these examples, we use [DMLC XGBoost](https://github.com/dmlc/xgboost), which is an optimized distributed gradient boosting library. 
It offers advanced features like GPU accelerated capabilities, and distributed/federated learning support.

## Data 
We use two datasets: [HIGGS](https://mlphysics.ics.uci.edu/data/higgs/) and [creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
to perform the experiments, both of them are binary classification task, but of significantly different scales:
HIGGS dataset contains 11 million instances, each with 28 attributes; while creditcardfraud dataset contains 284,807 instances, each with 30 attributes.

We use the HIGGS dataset to compare the performance of different federated learning settings for its large scale; 
and the creditcardfraud dataset to demonstrate the secure federated learning with homomorphic encryption for computation efficiency.
Please note that the websites may experience occasional downtime.

First download the dataset from the links above, which is a single zipped `HIGGS.csv.gz` file and a single `creditcard.csv` file.
By default, we assume the dataset is downloaded, uncompressed, and stored in `DATASET_ROOT/HIGGS.csv` and `DATASET_ROOT/creditcard.csv`.
Each row corresponds to a data sample, and each column corresponds to a feature. 

## Collaboration Modes and Data Split
Essentially there are two collaboration modes: horizontal and vertical:
- In horizontal case, each participant has access to the same features (columns) of different data samples (rows). 
In this case, everyone holds equal status as "label owner"
- In vertical case, each client has access to different features (columns) of the same data samples (rows).
We assume that only one is the "label owner" (or we call it as the "active party")

To simulate the above two collaboration modes, we split the two datasets both horizontally and vertically, and 
we give site-1 the label column for simplicity.

## Federated Training of XGBoost
Continue with this example for two scenarios:
### [Federated XGBoost without Encryption](./fedxgb/README.md)
This example includes instructions on running federated XGBoost without encryption under histogram-based and tree-based horizontal 
collaboration, and histogram-based vertical collaboration.

### [Secure Federated XGBoost with Homomorphic Encryption](./fedxgb_secure/README.md)
This example includes instructions on running secure federated XGBoost with homomorphic encryption under 
histogram-based horizontal and vertical collaboration. Note that tree-based collaboration does not have security concerns 
that can be handled by encryption.
