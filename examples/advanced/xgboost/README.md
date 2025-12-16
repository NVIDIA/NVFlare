# Federated Learning for XGBoost

This guide demonstrates how to use NVIDIA FLARE (NVFlare) to train XGBoost models in a federated learning environment. It showcases multiple collaboration strategies with varying levels of security.

NVFlare provides the following advantages:

- Secure training with Homomorphic Encryption (HE), protecting local histograms and gradients from the federated server and passive parties.
- Lifecycle management of XGBoost processes
- Reliable messaging that can overcome network glitches
- Training over complex networks with relays

This guide covers several federated XGBoost configurations:
- **Horizontal Collaboration**: Histogram-based and tree-based approaches (non-secure and secure)
- **Vertical Collaboration**: Histogram-based approach (non-secure and secure with Homomorphic Encryption)

## What is XGBoost?

XGBoost (eXtreme Gradient Boosting) is a powerful machine learning algorithm that uses decision/regression trees for classification and regression tasks. It excels particularly with tabular data and remains widely used due to its:
- **High performance** on structured data
- **Explainability** of predictions
- **Computational efficiency**

These examples use [DMLC XGBoost](https://github.com/dmlc/xgboost), which provides:
- GPU acceleration capabilities
- Distributed and federated learning support
- Optimized gradient boosting implementations

---

## Federated Learning Modes

### Horizontal Federated Learning
In horizontal collaboration, each participant has:
- **Same features** (columns) across all sites
- **Different data samples** (rows) at each site
- **Equal status** as label owners

**Example**: Multiple hospitals each have complete patient records (all features), but different patients.

### Vertical Federated Learning
In vertical collaboration, each participant has:
- **Different features** (columns) at each site
- **Same data samples** (rows) across all sites
- **One "active party"** (label owner) and multiple "passive parties"

**Example**: A bank and a retailer have data about the same customers, but different attributes (financial vs. shopping behavior).

---

## Security Risks and Mitigations

### Risks

Federated XGBoost faces three main security risks:

1. **Model Statistics Leakage**: The default XGBoost JSON model contains "sum_hessian" statistics that enable model inversion attacks to recover data distributions. (Reference: [TimberStrike](https://arxiv.org/abs/2506.07605))

2. **Histogram Leakage**: Gradient histograms can be exploited to reconstruct data distributions. The same model statistics of "sum_hessian" can be derived from histograms. (Reference: [TimberStrike](https://arxiv.org/abs/2506.07605))

3. **Gradient Leakage**: Sample-wise gradients may reveal label information. (Reference: [SecureBoost](https://arxiv.org/abs/1901.08755))


### Attack Surface

The attack surface varies by collaboration mode and party role:

**Server**: Depending on the collaboration mode, the server may have access to 
1. The local model:
   - Horizontal tree-based:
      - **Model Statistics Leakage** over each client's data distribution
2. Local histograms:
   - Horizontal histogram-based / vertical histogram-based:
      - **Histogram Leakage** over each client / passive party's data distribution
3. Sample-wise gradients:
   - Vertical histogram-based:
      - **Gradient Leakage** over active party's label information

**Clients**: Depending on the collaboration mode, the clients may have access to
1. The aggregated global model:
   - Horizontal tree-based:
      - **Model Statistics Leakage** over global data distribution 
2. Global histograms:
   - Horizontal histogram-based:
      - **Histogram Leakage** over global data distribution 
3. Local histograms:
   - Vertical histogram-based: 
      - **Histogram Leakage** over each passive party's data distribution on active party
3. Sample-wise gradients:
   - **Gradient Leakage** over active party's label information on passive parties

### Mitigations

The following table summarizes the available mitigations for different collaboration scenarios:

| Collaboration Mode | Algorithm | Data Exchange | Risk Mitigated | Security Measure | Implementation |
|-------------------|-----------|---------------|---------------|------------------|----------------|
| **Horizontal** | Tree-based | Clients send locally boosted trees to server; server combines and distributes trees back to clients | **Model statistics leakage** on both server and clients | Remove "sum_hessian" values from JSON model | Removed before clients send local trees to server |
| **Horizontal** | Histogram-based | Clients send local histograms to server; server aggregates to global histogram and distributes it back to clients | **Histogram leakage** on server (client-side remain) | Encrypt histograms | Local histograms encrypted before transmission |
| **Vertical** | Histogram-based | Active party computes gradients; routed by server, passive parties receive gradients, compute histograms, and send them back to active party through server | **Histogram leakage** on server (active party-side remain), **Gradient leakage** on both server and passive parties | **Primary**: Encrypt gradients<br>**Secondary**: Mask feature ownership in split values | Gradients encrypted before sending out to passive parties |

**Notes**
- **Vertical histogram-based**: 
   - **Primary goal**: Protect sample gradients from passive parties (critical)
   - **Secondary goal**: Hide split values from non-feature owners (desirable but lower risk)
- **The remaining two risks** will be discussed in the [last section](./README.md#advanced-topics-future-security-scenarios).

---

## GPU Acceleration

Federated XGBoost supports two levels of GPU acceleration:

### 1. XGBoost GPU Training
Enable GPU-accelerated training by setting `tree_method='gpu_hist'` when initializing the XGBoost model.
- **Performance**: Up to **4.15x speedup** vs. CPU training ([GPU XGBoost Blog](https://developer.nvidia.com/blog/gradient-boosting-decision-trees-xgboost-cuda/))

### 2. GPU-Accelerated Homomorphic Encryption (HE)
NVFlare provides GPU acceleration for HE operations using specialized encryption plugins.
- **Performance**: Up to **36.5x speedup** vs. CPU encryption ([NVFlare Secure XGBoost Blog](https://developer.nvidia.com/blog/security-for-data-privacy-in-federated-learning-with-cuda-accelerated-homomorphic-encryption-in-xgboost/))

### Security Implementation Matrix

The following table shows which security measures are supported across different hardware configurations:

| Collaboration Mode | Security Goal | CPU XGBoost<br>+ CPU Encryption | CPU XGBoost<br>+ GPU Encryption | GPU XGBoost<br>+ CPU Encryption | GPU XGBoost<br>+ GPU Encryption |
|-------------------|---------------|:-------------------------------:|:-------------------------------:|:-------------------------------:|:-------------------------------:|
| **Horizontal** | Histogram protection against server | ✅ | N/A* | ✅ | N/A* |
| **Vertical** | **Primary**: Gradient protection | ✅ | ✅ | ✅ | ✅ |
| **Vertical** | **Secondary**: Split value masking | ✅ | ✅ | ❌ | ❌ |

**\*Note**: Horizontal histogram encryption is not computationally intensive (encrypting histogram vectors), so GPU encryption is not needed.

**Implementation Notes**:
- **Vertical mode primary goal** (gradient protection): Fully supported across all configurations
- **Vertical mode secondary goal** (split value masking): Only supported with CPU XGBoost

---
## Prerequisites

Before running the examples, set up your environment:

1. Create a virtual environment and install Jupyterlab (see [example root README](../../README.md))
2. Install additional requirements:
   ```bash
   python3 -m pip install -r requirements.txt
   ```
3. Set up encryption plugins by following the [encryption plugins instructions](../../../integration/xgboost/encryption_plugins)

## Datasets

We use two binary classification datasets with different scales:

### 1. HIGGS Dataset
- **Source**: [UCI Machine Learning Repository](https://mlphysics.ics.uci.edu/data/higgs/)
- **Size**: 11 million instances, 28 features
- **Use**: Performance comparison of different federated learning settings

### 2. Credit Card Fraud Dataset
- **Source**: [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 instances, 30 features
- **Use**: Demonstrating secure federated learning with HE (computational efficiency)

### Data Preparation

1. Download the datasets:
   - HIGGS: `HIGGS.csv.gz` (uncompress after download)
   - Credit Card Fraud: `creditcard.csv`

2. Place the files in your dataset directory:
   - `DATASET_ROOT/HIGGS.csv`
   - `DATASET_ROOT/creditcard.csv`

3. Data format: Each row = data sample, each column = feature

4. Data splitting: Both datasets are split horizontally and vertically to simulate different collaboration modes. The label column is assigned to site-1 by default.

**Note**: Dataset hosting websites may experience occasional downtime.

---

## Getting Started

Choose your scenario:

### [Federated XGBoost without Encryption](./fedxgb/README.md)
Run federated XGBoost in non-secure mode:
- Histogram-based horizontal collaboration
- Tree-based horizontal collaboration
- Histogram-based vertical collaboration

### [Secure Federated XGBoost with Homomorphic Encryption](./fedxgb_secure/README.md)
Run secure federated XGBoost with HE:
- Histogram-based horizontal collaboration (encrypted)
- Histogram-based vertical collaboration (encrypted)

**Note**: Tree-based collaboration does not require encryption-based security measures.

---

## Advanced Topics: Future Security Scenarios

The following security scenarios are not currently implemented in our solution. Users should be aware that **plaintext histogram communication** can reveal data distribution information, which may enable data reconstruction attacks as stated above. On the other hand, similar statistics can also be derived from common practices such as [federated statistics](../federated-statistics/README.md). As the attack potency depends on multiple factors including data complexity, model hyperparameters, and the data distribution information that can be utilized, the corresponding indications of a certain type of attack can vary significantly. This is still an open and active research area.   

### Potential Future Enhancements to Protect Against All Parties

| Collaboration Mode | Algorithm | Remaining Security Risk | Possible Approach | Challenges |
|--------------------|-----------|---------------|-------------------|------------|
| **Horizontal** | Histogram-based | Histogram leakage over global data distribution on clients (in addition to server as addressed above) | Confidential computing,<br>advanced HE | HE compatibility issue* with server performing calculations and distributing only final splits |
| **Vertical** | Histogram-based | Histogram leakage over each passive party's data distribution on active party (in addition to Histogram leakage on server, and Gradient leakage on server and passive parties as addressed above) | Local data preprocessing and anonymization,<br>confidential computing,<br>advanced HE | HE compatibility issue* with passive parties performing calculations and sending only final splits |

**\*HE Compatibility Challenge**: Current Homomorphic Encryption schemes do not efficiently support operations like ciphertext division and argmax, which are required for performing split calculations on encrypted data. Advanced HE features are needed to support approaches that "perform calculations until splits on the server/passive parties."

---

## Additional Resources

- [NVIDIA FLARE Documentation](https://nvflare.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [GPU XGBoost Blog](https://developer.nvidia.com/blog/gradient-boosting-decision-trees-xgboost-cuda/)
- [NVFlare Secure XGBoost Blog](https://developer.nvidia.com/blog/security-for-data-privacy-in-federated-learning-with-cuda-accelerated-homomorphic-encryption-in-xgboost/)

---

## Questions or Issues?

If you encounter problems or have questions:
1. Check the [FAQ](../../../docs/faq.rst)
2. Review the [NVIDIA FLARE documentation](https://nvflare.readthedocs.io/)
3. Open an issue on the [GitHub repository](https://github.com/NVIDIA/NVFlare)
