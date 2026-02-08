.. _hello_xgboost:

#####################
Hello XGBoost
#####################

.. note::
   This example is coming soon. It will provide a complete walkthrough of
   federated XGBoost training with NVIDIA FLARE.

Overview
========

This example demonstrates how to run federated XGBoost training using NVIDIA FLARE.
XGBoost is one of the most popular gradient boosting frameworks, widely used in
tabular data applications including fraud detection, credit scoring, and healthcare.

NVIDIA FLARE supports multiple XGBoost federation modes:

- **Horizontal (row-split)** -- Each site has different samples with the same features
- **Vertical (column-split)** -- Each site has different features for the same samples
- **Histogram-based** -- Federated histogram aggregation for tree construction

*Full example walkthrough coming soon.*

For now, see:

- :doc:`/user_guide/data_scientist_guide/federated_xgboost/federated_xgboost` -- Comprehensive XGBoost guide
- `XGBoost examples on GitHub <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/xgboost>`_ -- Code examples
