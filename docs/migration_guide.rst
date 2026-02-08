.. _migration_guide:

############################################
Migrating from Centralized to Federated ML
############################################

.. note::
   This guide is coming soon. It will provide step-by-step patterns for converting
   existing centralized ML training code to federated learning with NVIDIA FLARE.

Overview
========

If you have existing ML training code (PyTorch, TensorFlow, XGBoost, scikit-learn, etc.)
and want to run it in a federated setting, NVIDIA FLARE's **Client API** makes this
straightforward -- typically requiring only 5-10 lines of code changes.

This guide will cover:

- When federated learning is (and isn't) the right approach
- Step-by-step conversion of centralized training scripts
- Handling data loading and preprocessing in federated settings
- Common pitfalls and how to avoid them
- Testing your federated code with the FL Simulator

When to Use Federated Learning
==============================

Federated learning is most beneficial when:

- **Data cannot be centralized** due to privacy regulations (HIPAA, GDPR), competitive
  sensitivity, or data sovereignty requirements
- **Data is naturally distributed** across organizations, devices, or geographies
- **Model quality improves** with access to broader, more diverse data
- **Collaboration** between multiple parties is desired without raw data sharing

Federated learning may not be the best fit when:

- Data can easily and legally be centralized
- The dataset at each site is too small to produce meaningful gradients
- Real-time or online learning is required with sub-second latency

Quick Conversion Example
=========================

*Coming soon.* This section will show a side-by-side comparison of centralized
vs. federated code for PyTorch, TensorFlow, and XGBoost.

For now, see:

- :ref:`Client API Guide <client_api>` -- Core conversion patterns
- :doc:`Hello PyTorch </hello-world/hello-pt/index>` -- Complete PyTorch example
- :doc:`Hello TensorFlow </hello-world/hello-tf/index>` -- Complete TensorFlow example

Data Preparation Considerations
================================

*Coming soon.* This section will cover:

- Data partitioning strategies (IID vs non-IID)
- Feature alignment across sites
- Data quality validation before federated training
- Privacy-preserving data exploration with :doc:`federated statistics </hello-world/hello-tabular-stats/index>`

Testing Your Federated Code
============================

Before deploying to production, always test with the FL Simulator:

.. code-block:: bash

   nvflare simulator -w /tmp/sim_workspace -n 2 -t 2 my_job

See the :ref:`FL Simulator <fl_simulator>` guide for detailed usage.
