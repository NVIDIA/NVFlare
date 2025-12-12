# Federated Learning for XGBoost 
This example demonstrates how to use NVFlare to train an XGBoost model in a federated learning setting. 
Several potential variations of federated XGBoost are illustrated, including:
- Non-secure horizontal collaboration with histogram-based and tree-based mechanisms
- Non-secure vertical collaboration with histogram-based mechanism
- Secure horizontal and vertical collaboration with histogram-based mechanism and Homomorphic Encryption (HE)

To run the examples and notebooks, please make sure you set up a virtual environment and Jupyterlab, following [the example root readme](../../README.md)
and install the additional requirements:
```
python3 -m pip install -r requirements.txt
```
Also, follow [encryption plugins instruction](../../../integration/xgboost/encryption_plugins) to setup encryption plugins.

## XGBoost 
XGBoost is a machine learning algorithm that uses decision/regression trees to perform classification and regression tasks, 
mapping a vector of feature values to its label prediction. It is especially powerful for tabular data, so even in the age of LLMs, 
it is still widely used for many tabular data use cases. It is also preferred for its explainability and efficiency.

In these examples, we use [DMLC XGBoost](https://github.com/dmlc/xgboost), which is an optimized distributed gradient boosting library. 
It offers advanced features like GPU accelerated capabilities, and distributed/federated learning support.

## Collaboration Modes and Data Split
Essentially, there are two collaboration modes: horizontal and vertical:
- In the horizontal case, each participant has access to the same features (columns) of different data samples (rows). 
In this case, everyone holds equal status as a "label owner".
- In the vertical case, each client has access to different features (columns) of the same data samples (rows).
We assume that only one client is the "label owner" (also called the "active party"), while all other clients are "passive parties".

## Security Measures
Security risks exist based on prior research [e.g. [SecureBoost](https://arxiv.org/abs/1901.08755), [TimberStrike](https://arxiv.org/abs/2506.07605)] that exploits several types of information: sample-wise gradients for label recovery, gradient histograms for distribution recovery, and final model statistics for model inversion. We have three basic security categories:
- **Model statistics leakage:** The default XGBoost model JSON with "sum_hessian" statistics can enable model inversion to recover data distribution.
- **Histogram leakage:** Information can be recovered from gradient histograms and used to reconstruct data distributions.
- **Gradient leakage:** Sample-wise gradients may leak label information.

Based on the above vulnerabilities, the following table outlines the different collaboration modes, algorithms, assumptions, and security measures available in secure federated XGBoost:

| Collaboration Mode | Algorithm | Data Exchange | Security Category | Security Assumptions | Security Measures | Notes |
|-------------------|-----------|---------------|-------------------|----------------------|-------------------|-------|
| **Horizontal** | Tree-based | Clients submit locally boosted trees to server; server combines and routes trees back to clients | Model statistics leakage | No trust in either server or other clients | Remove the actual "sum_hessian" numbers from JSON model before sending to server | All trees become part of the final model. |
| **Horizontal** | Histogram-based | Clients submit local histograms to server; server aggregates them to global histogram | Histogram leakage | No trust in server, trust other clients | Encryption of histograms | Local histograms encrypted before sending to server. |
| **Vertical** | Histogram-based | Active party computes gradients for all data samples; passive parties receive gradients and compute local histograms; histograms sent back to active party | Gradient leakage | No trust in passive parties, trust in active party | **Primary:** encryption of gradients; **Secondary:** feature ownership masking | Gradients encrypted before sending to passive parties. Split values in final model are masked according to feature ownership. |

### Notes:
- In horizontal mode, tree-based collaboration is secured by removing "sum_hessian" values so that it cannot be exploited for inversion attacks.
- In vertical mode, histogram-based collaboration has two security goals:
  - **Primary** goal is to protect the sample gradients sent to passive parties.
  - **Secondary** goal is to allow clients to only see split values for their own features. This is a desirable feature, but it does not pose as significant a security risk as the primary goal.

## GPU Accelerations
There are two levels of GPU accelerations in federated XGBoost:
1. XGBoost itself has built-in GPU acceleration for training. To enable it, set the `tree_method` parameter to `gpu_hist` when initializing the XGBoost model. [GPU XGBoost Blog](https://developer.nvidia.com/blog/gradient-boosting-decision-trees-xgboost-cuda/) shows that this method can achieve a **4.15x** speed improvement compared to CPU-based training for the dataset and testing environment.
2. NVFlare provides GPU acceleration for HE operations. To enable it, use different encryption plugins. This can significantly speed up the encryption and decryption processes, as shown in [NVFlare Secure XGBoost Blog](https://developer.nvidia.com/blog/security-for-data-privacy-in-federated-learning-with-cuda-accelerated-homomorphic-encryption-in-xgboost/), GPU acceleration can achieve **up to 36.5x** speed improvement compared to CPU-based encryption for the dataset and testing environment.

We will refer to them as "CPU/GPU XGBoost" and "CPU/GPU Encryption".

## Security Implementation Matrix
As shown above, histogram-based XGBoost in horizontal and vertical collaboration modes can utilize HE to enhance data privacy. The following table shows which security measures are implemented (as shown by ✅) across different combinations of XGBoost and encryption modes:

| Collaboration Mode | Security Goal | CPU XGBoost + CPU Encryption | CPU XGBoost + GPU Encryption | GPU XGBoost + CPU Encryption | GPU XGBoost + GPU Encryption |
|-------------------|---------------|------------------------------|------------------------------|------------------------------|------------------------------|
| **Horizontal** | Protection of histograms against server | ✅ | Not needed | ✅ | Not needed |
| **Vertical** | **Primary:** Protection of sample gradients against passive parties | ✅ | ✅ | ✅ | ✅ |
| **Vertical** | **Secondary:** Protection of split values against non-feature owners | ✅ | ✅ | ❌ | ❌ |

### Note on Implementation:
- **Horizontal mode**: 
  - The histogram-based horizontal model does not need GPU encryption, as it is not as computationally intensive (encrypting histogram vectors) as vertical mode (encrypting gradients).
- **Vertical mode**: 
  - Primary goal (gradient protection) is fully supported across all combinations.
  - Secondary goal (split value masking) is only supported with CPU XGBoost, regardless of encryption type.

## Data 
We use two datasets: [HIGGS](https://mlphysics.ics.uci.edu/data/higgs/) and [creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
to perform the experiments. Both are binary classification tasks but of significantly different scales:
The HIGGS dataset contains 11 million instances, each with 28 attributes, while the creditcardfraud dataset contains 284,807 instances, each with 30 attributes.

We use the HIGGS dataset to compare the performance of different federated learning settings due to its large scale, 
and the creditcardfraud dataset to demonstrate secure federated learning with HE for computational efficiency.
Please note that the websites may experience occasional downtime.

First, download the datasets from the links above: a single zipped `HIGGS.csv.gz` file and a single `creditcard.csv` file.
By default, we assume the datasets are downloaded, uncompressed, and stored in `DATASET_ROOT/HIGGS.csv` and `DATASET_ROOT/creditcard.csv`.
Each row corresponds to a data sample, and each column corresponds to a feature. 

To simulate the above collaboration modes, we split both datasets horizontally and vertically, assigning the label column to site-1 for simplicity.

## Federated Training of XGBoost
Continue with this example for two scenarios:
### [Federated XGBoost without Encryption](./fedxgb/README.md)
This example includes instructions for running federated XGBoost without encryption under histogram-based and tree-based horizontal 
collaboration, as well as histogram-based vertical collaboration.

### [Secure Federated XGBoost with Homomorphic Encryption](./fedxgb_secure/README.md)
This example includes instructions for running secure federated XGBoost with HE under 
histogram-based horizontal and vertical collaboration. Note that tree-based collaboration does not have security concerns 
that can be addressed by encryption.

## Disclaimer and Future Outlook
Other security assumption scenarios listed below are not included in our current secure XGBoost solution. Histogram communication in plaintext can reveal data distribution information, and users should be aware of its implications for potential data reconstruction and choose appropriate actions accordingly:

| Collaboration Mode | Algorithm | Security Category | Security Assumptions | Possible Security Measures | Notes |
|--------------------|-----------|-------------------|----------------------|----------------------------|-------|
| **Horizontal** | Histogram-based | Histogram leakage | Trust server, no trust in other clients | Perform most calculations on the server, only distribute the final splits to clients | Such an assumption (trusting the server with data distributions) is rare and not common in practice |
| **Horizontal** | Histogram-based | Histogram leakage | No trust in server, no trust in other clients | Confidential computing | HE compatibility issue [*](#he-compatibility-note) |
| **Vertical** | Histogram-based | Histogram leakage | Trust passive parties, no trust in active party | Perform most calculations on passive parties, only send the final splits to active party | Such an assumption (trusting passive parties with sample labels) is rare and not common in practice |
| **Vertical** | Histogram-based | Histogram + Gradient leakage | No trust in passive parties, no trust in active party | Local secret data preprocessing and anonymization, Confidential computing | HE compatibility issue [*](#he-compatibility-note) |

<a id="he-compatibility-note"></a>
**[*] HE Compatibility Note:** HE (Homomorphic Encryption) is currently not compatible with performing calculations until splits because it would require support for operations like ciphertext division and argmax, which are not efficiently supported by current HE schemes. Therefore, the existing HE solution cannot co-exist with methods that require computation on encrypted data - "Perform most calculations on the server / passive parties".
