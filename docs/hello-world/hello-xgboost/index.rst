.. _hello_xgboost:

#####################
Hello XGBoost
#####################

Overview
========

This example demonstrates how to run federated XGBoost training using NVIDIA FLARE.
XGBoost is one of the most popular gradient boosting frameworks, widely used in
tabular data applications including fraud detection, credit scoring, and healthcare.

NVIDIA FLARE supports multiple XGBoost federation modes:

- **Horizontal (row-split)** -- Each site has different samples with the same features
- **Vertical (column-split)** -- Each site has different features for the same samples
- **Histogram-based** -- Federated histogram aggregation for tree construction

For the comprehensive XGBoost guide, see :doc:`/user_guide/data_scientist_guide/federated_xgboost/federated_xgboost`.

Examples
========

- `Federated XGBoost (horizontal) <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/xgboost/fedxgb>`_ -- Standard federated XGBoost with histogram-based aggregation
- `Secure Federated XGBoost <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/xgboost/fedxgb_secure>`_ -- XGBoost with encryption for secure aggregation
