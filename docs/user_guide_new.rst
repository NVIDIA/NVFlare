.. _user_guide:

##########
User Guide
##########

This guide is for data scientists and ML engineers who want to use NVIDIA FLARE for federated learning applications.

Choosing the Right API
======================

FLARE provides several APIs at different levels of abstraction. Here's how to choose:

- **Client API** -- Start here. Add a few lines to your existing training script to make it federated. See :ref:`client_api`.
- **Job Recipe API** -- Pick a pre-built FL algorithm (FedAvg, SCAFFOLD, XGBoost, etc.) and run it immediately with minimal configuration. See :ref:`job_recipe`.
- **FLARE API** -- Submit, monitor, and manage FL jobs programmatically from Python or Jupyter notebooks. Ideal for experiment workflows. See :ref:`flare_api`.
- **FedJob API** -- Compose jobs programmatically with full control over components. For advanced users who need custom configurations. See :ref:`fed_job_api` in the Developer Guide.

Most users only need the **Client API** + **Job Recipe API** to get productive quickly.

For low-level APIs and system internals, see the :ref:`Architecture & Developer Guide <developer_guide>`.
