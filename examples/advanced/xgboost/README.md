# Federated Learning for XGBoost 
This example demonstrates how to use NVFlare to train an XGBoost model in a federated learning setting. 
Several potential variations of federated XGBoost are illustrated, including:
- non-secure horizontal collaboration with histogram-based and tree-based mechanisms.
- non-secure vertical collaboration with histogram-based mechanism.
- secure horizontal and vertical collaboration with histogram-based mechanism and Homomorphic Encryption (HE).

To run the examples and notebooks, please make sure you set up a virtual environment and Jupyterlab, following [the example root readme](../../README.md)
and install the additional requirements:
```
python3 -m pip install -r requirements.txt
```
Also, follow [encryption plugins instruction](../../../integration/xgboost/encryption_plugins) to setup encryption plugins.

## XGBoost 
XGBosot is a machine learning algorithm that uses decision/regression trees to perform classification and regression tasks, 
mapping a vector of feature values to its label prediction. It is especially powerful for tabular data, so even in the age of LLM, 
it still widely used for many tabular data use cases. It is also preferred for its explainability and efficiency.

In these examples, we use [DMLC XGBoost](https://github.com/dmlc/xgboost), which is an optimized distributed gradient boosting library. 
It offers advanced features like GPU accelerated capabilities, and distributed/federated learning support.

## Collaboration Modes and Data Split
Essentially there are two collaboration modes: horizontal and vertical:
- In horizontal case, each participant has access to the same features (columns) of different data samples (rows). 
In this case, everyone holds equal status as "label owner".
- In vertical case, each client has access to different features (columns) of the same data samples (rows).
We assume that only one is the "label owner" (or we call it as the "active party"), all other clients are "passive parties".

## Security Measures
The following table outlines the different collaboration modes, algorithms, and security measures available in federated XGBoost:

| Collaboration Mode | Algorithm | Data Exchange | Security Measures                                                              | Notes                                                                                                                         |
|-------------------|-----------|---------------|--------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| **Horizontal** | Tree-based | Clients submit locally boosted trees to server; server combines and routes trees back to clients | None                                                                           | All trees become part of the final model.                                                                                     |
| **Horizontal** | Histogram-based | Clients submit local histograms to server; server aggregates them to global histogram | Encryption of histograms                                                       | Local histograms encrypted before sending to server.                                                                          |
| **Vertical** | Histogram-based | Active party computes gradients for all data samples; passive parties receive gradients and compute local histograms; histograms sent back to active party | **Primary:** encryption of gradients; **Secondary:** feature ownership masking | Gradients encrypted before sending to passive parties. Split values in final model are masked according to feature ownership. |

### Notes:
- In horizontal mode, tree-based collaboration does not have security concerns that can be handled by encryption.
- In vertical mode, histogram-based collaboration has two security goals:
  - **Primary** goal is to protect the sample gradients sent to passive parties, as they can be used to recover the labels of every single data samples.
  - **Secondary** goal is to let clients only see split values for their own features. This is a feature good to have, while it does not pose a secure risk as significant as the primary goal.

## GPU Accelerations
There are two levels of GPU accelerations in federated XGBoost:
1. XGBoost itself has built-in GPU acceleration for training. To enable it, set the `tree_method` parameter to `gpu_hist` when initializing the XGBoost model. [GPU XGBoost Blog](https://developer.nvidia.com/blog/gradient-boosting-decision-trees-xgboost-cuda/) shows that this method can achieve a **4.15x** speed improvement compared to CPU-based training for the dataset and testing environment.
2. NVFlare provides GPU acceleration for HE operations. To enable it, use different encryption plugins. This can significantly speed up the encryption and decryption processes, as shown in [NVFlare Secure XGBoost Blog](https://developer.nvidia.com/blog/security-for-data-privacy-in-federated-learning-with-cuda-accelerated-homomorphic-encryption-in-xgboost/), GPU acceleration can achieve **up to 36.5x** speed improvement compared to CPU-based encryption for the dataset and testing environment.

We will refer to them as "CPU / GPU XGBoost" and "CPU / GPU Encryption"

## Security Implementation Matrix
As shown above, histogram-based XGBoost in horizontal and vertical collaboration modes can utilize HE to enhance data privacy. The following table shows which security measures are implemented (as shown by ✅) across different combinations of XGBoost and encryption modes:

| Collaboration Mode | Security Goal | CPU XGBoost + CPU Encryption | CPU XGBoost + GPU Encryption | GPU XGBoost + CPU Encryption | GPU XGBoost + GPU Encryption |
|-------------------|---------------|------------------------------|------------------------------|------------------------------|------------------------------|
| **Horizontal** | Protection of histograms against server | ✅ | Not needed | ✅ | Not needed |
| **Vertical** | **Primary:** Protection of sample gradients against passive parties | ✅ | ✅ | ✅ | ✅ |
| **Vertical** | **Secondary:** Protection of split values against non-feature owners | ✅ | ✅ | ❌ | ❌ |

### Note on Client-side Horizontal Vulnerabilities
In this example, we utilize HE to protect histograms against server without considering client-side vulnerabilities. 

For client-side, a recent research [TimberStrike](https://arxiv.org/pdf/2506.07605) highlights privacy vulnerabilities in federated tree-based systems. The attack exploits split values and decision paths to reconstruct training data, achieving reconstruction accuracies around 80% on certain benchmark datasets.

The vulnerability affects both collaboration modes:
- **Tree-based collaboration**: Since local trees are shared directly, they allow for **local reconstruction** of specific client's private data.
- **Histogram-based collaboration**: As data is aggregated, the resulting global histogram still leaks enough information for **global reconstruction** of the overall underlying data distribution.

One potential solution as proposed in this work is that we can move the split finding phase to the server, such that clients will not have access to the histograms. This indeed will handle the client-side leakage. Unfortunately, we note that this solution is not compatible with existing server-end protection schemes of HE because the computations needed (e.g. division / argmax) are beyond the capability of standard HE. Therefore, implementing this would only "move" the vulnerability to the server-side rather than "address" it. Even worse, in this solution where the server performs split finding, the server would have access to individual histograms, enabling **local reconstruction** for each client's data at a higher accuracy than tree-based collaboration as shown in the paper.

Future work combining HE with Confidential Computing (CC) could potentially address the issue effectively.

### Note on Implementation:
- **Horizontal mode**: 
  - Histogram-based horizontal model does not need GPU encryption, as it is not as computationally intensive (encrypt histogram vectors) as in vertical mode (encrypt gradients).
- **Vertical mode**: 
  - Primary goal (gradient protection) is fully supported across all combinations
  - Secondary goal (split value masking) is only supported with CPU XGBoost, regardless of encryption type

## Data 
We use two datasets: [HIGGS](https://mlphysics.ics.uci.edu/data/higgs/) and [creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
to perform the experiments, both of them are binary classification task, but of significantly different scales:
HIGGS dataset contains 11 million instances, each with 28 attributes; while creditcardfraud dataset contains 284,807 instances, each with 30 attributes.

We use the HIGGS dataset to compare the performance of different federated learning settings for its large scale; 
and the creditcardfraud dataset to demonstrate the secure federated learning with HE for computation efficiency.
Please note that the websites may experience occasional downtime.

First download the dataset from the links above, which is a single zipped `HIGGS.csv.gz` file and a single `creditcard.csv` file.
By default, we assume the dataset is downloaded, uncompressed, and stored in `DATASET_ROOT/HIGGS.csv` and `DATASET_ROOT/creditcard.csv`.
Each row corresponds to a data sample, and each column corresponds to a feature. 

To simulate the above collaboration modes, we split the two datasets both horizontally and vertically, and we give site-1 the label column for simplicity.

## Federated Training of XGBoost
Continue with this example for two scenarios:
### [Federated XGBoost without Encryption](./fedxgb/README.md)
This example includes instructions on running federated XGBoost without encryption under histogram-based and tree-based horizontal 
collaboration, and histogram-based vertical collaboration.

### [Secure Federated XGBoost with Homomorphic Encryption](./fedxgb_secure/README.md)
This example includes instructions on running secure federated XGBoost with HE under 
histogram-based horizontal and vertical collaboration. Note that tree-based collaboration does not have security concerns 
that can be handled by encryption.
