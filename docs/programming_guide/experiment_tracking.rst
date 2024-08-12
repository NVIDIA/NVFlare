.. _experiment_tracking:

###################
Experiment Tracking
###################

FLARE seamlessly integrates with leading experiment tracking systems—MLflow, Weights & Biases, and TensorBoard—to facilitate comprehensive monitoring of metrics.

You can choose between decentralized and centralized tracking configurations:

- **Decentralized tracking**: Each client manages its own metrics and experiment tracking server locally, maintaining training metric privacy. However, this setup limits the ability to compare data across different sites.
- **Centralized tracking**: All metrics are streamed to a central FL server, which then pushes the data to a selected tracking system. This setup supports effective cross-site metric comparisons

We provide solutions for different client execution types. For the Client API, use the corresponding experiment tracking APIs. For Executors or Learners, use the experiment tracking LogWriters.

.. toctree::
   :maxdepth: 1

   experiment_tracking/experiment_tracking_apis
   experiment_tracking/experiment_tracking_log_writer
