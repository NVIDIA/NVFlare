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

*Coming soon.* Will cover:

- What is non-IID data and why it matters for FL
- Types of heterogeneity: feature skew, label skew, quantity skew
- Impact on model convergence and accuracy
- Mitigation strategies (FedProx, SCAFFOLD, personalization)

Federated Data Exploration
===========================

Before training, use **Federated Statistics** to understand data distributions across
sites without sharing raw data:

- :doc:`Hello Tabular Statistics </hello-world/hello-tabular-stats/index>` -- compute
  statistics (mean, std, histogram) across federated tabular data

Feature Alignment
=================

*Coming soon.* Will cover:

- Ensuring consistent feature schemas across sites
- Handling missing features at some sites
- Feature encoding consistency
- Private Set Intersection (PSI) for entity alignment

Data Quality Validation
========================

*Coming soon.* Will cover:

- Pre-training data quality checks
- Detecting and handling outliers across sites
- Monitoring data drift during training
- Using FLARE filters for data validation
