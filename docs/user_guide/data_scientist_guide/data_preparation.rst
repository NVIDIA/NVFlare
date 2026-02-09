.. _data_preparation:

###################################
Data Preparation & Heterogeneity
###################################

.. note::
   This guide is coming soon. It will cover best practices for preparing data
   for federated learning, handling data heterogeneity, and validating data quality
   across participating sites.

Overview
========

Data preparation in federated learning differs from centralized ML because:

- Each site has its own local dataset that cannot be inspected centrally
- Data distributions across sites are often **non-IID** (not identically distributed)
- Feature schemas must be aligned without sharing raw data
- Data quality varies across sites

This guide covers practical approaches to these challenges.

Data Heterogeneity (Non-IID Data)
=================================

In federated learning, data across sites is often **non-IID** (not independently and identically distributed).
Sites may have different label distributions, feature distributions, or dataset sizes.
This heterogeneity can slow convergence and reduce model accuracy compared to centralized training.

**Mitigation strategies with examples:**

- `FedProx <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/cifar10/pt/cifar10-sim/cifar10_fedprox>`_ -- Adds a proximal regularization term to prevent client models from drifting too far from the global model during local training
- `SCAFFOLD <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/cifar10/pt/cifar10-sim/cifar10_scaffold>`_ -- Uses control variates to correct for client drift, significantly improving convergence on highly heterogeneous data

Federated Data Exploration
===========================

Before training, use **Federated Statistics** to understand data distributions across
sites without sharing raw data:

- :doc:`Hello Tabular Statistics </hello-world/hello-tabular-stats/index>` -- Compute statistics (mean, std, histogram) across federated tabular data
- `Federated Image Statistics <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/federated-statistics/image_stats>`_ -- Compute image histogram statistics across sites

User Alignment for Vertical FL
===============================

In vertical federated learning, different sites hold different features for overlapping users.
Before training, sites must identify their common users without revealing their full datasets.

- `Private Set Intersection (PSI) <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/psi>`_ -- User alignment for vertical federated learning; find common users/entities across sites without exposing private data

Data Quality Validation
========================

*Coming soon.* Will cover:

- Pre-training data quality checks
- Detecting and handling outliers across sites
- Monitoring data drift during training
- Using FLARE filters for data validation
