# Federated Learning for XGBoost 

## Introduction to XGBoost and HIGGS Data

### XGBoost
These examples show how to use [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) on tabular data applications.
They use [XGBoost](https://github.com/dmlc/xgboost),
which is an optimized distributed gradient boosting library.

### HIGGS
The examples illustrate a binary classification task based on [HIGGS dataset](https://archive.ics.uci.edu/ml/datasets/HIGGS). This dataset contains 11 million instances, each with 28 attributes.

## Federated Training of XGBoost
Several mechanisms have been proposed for training an XGBoost model in a federated learning setting. In these examples, we illustrate the use of NVFlare to carry out  *horizontal* federated learning using two approaches: histogram-based collaboration and tree-based collaboration.
### Horizontal Federated Learning
Under horizontal setting, each participant / client joining the federated learning will have part of the whole data / instances / examples/ records, while each instance has all the features. This is in contrast to vertical federated learning, where each client has part of the feature values for each instance.
#### Histogram-based Collaboration
The histogram-based collaboration federated XGBoost approach leverages NVFlare integration of recently added [federated learning support](https://github.com/dmlc/xgboost/issues/7778) in the XGBoost open-source library, which allows the existing *distributed* XGBoost training algorithm to operate in a federated manner, with the federated clients acting as the distinct workers in the distributed XGBoost algorithm.  In distributed XGBoost, the individual workers share and aggregate coarse information about their respective portions of the training data, as required to optimize tree node splitting when building the successive boosted trees.   The shared information is in the form of quantile sketches of feature values as well as corresponding sample gradient and sample Hessian histograms.   Under federated histogram-based collaboration, precisely the same information is exchanged among the clients.    The main differences are that the data is partitioned across the workers according to client data ownership, rather than being arbitrarily partionable, and all communication is via an aggregating federated [gRPC](https://grpc.io) server instead of direct client-to-client communication.   Histograms from different clients, in particular, are aggregated in the server and then communicated back to the clients.  See [histogram-based](histogram-based) for more information on the histogram-based collaboration example.

#### Tree-based Collaboration
Under tree-based collaboration, individual trees are independently trained on each client's local data without aggregating the global sample gradient histogram information. Trained trees are collected and passed to the server / other clients for aggregation and further boosting rounds.   The XGBoost Booster api is leveraged to create in-memory Booster objects that persist across rounds to cache predictions from trees added in previous rounds and retain other data structures needed for training.  See [tree-based/README](tree-based/README.md) for more information on two different types of tree-based collaboration algorithms.

