# NVFLARE JOB TEMPLATE REGISTRY

This directory contains NVFLARE job templates. 

## Introduction

Each job template contains the following informations

* client-side configuration: config_fed_client.conf
* server-side configuration: config_fed_server.conf
* job meta info: meta.conf
* (Optional) data exchange configuration: config_exchange.conf. This is only used with the new FLARE ML to FL transition Client API
* information card: info.md for display purpose
* information config: used by program

Refer to the [Job CLI Documentation](https://nvflare.readthedocs.io/en/main/user_guide/nvflare_cli/job_cli.html) for details on how to use the Job Templates with the Job CLI.

## Configuration format

Configurations are written in HOCON (human optimized object Notation). As a variant of JSON, .conf can also use json format.
The pyhocon format allows for comments, and you can remove many of the double quotes as well as replace ":" with "=" to make the configurations look cleaner.
You can find details in [pyhocon: HOCON Parser for python](https://github.com/chimpler/pyhocon).

## List of Job Templates

View all the available job templates with the following command:

```nvflare job list_templates```

| Example | Controller-Type | Execution API Type | Description |
|---------|-----------------|-----------------|-------------|
| [cyclic_cc_pt](./cyclic_cc_pt)                  | client           | client_api           | client-controlled cyclic workflow with PyTorch ClientAPI trainer |
| [cyclic_pt](./cyclic_pt)                        | server           | client_api           | server-controlled cyclic workflow with PyTorch ClientAPI trainer |
| [psi_csv](./psi_csv)                            | server           | Executor             | private-set intersection for csv data                |
| [sag_cross_np](./sag_cross_np)                  | server           | client executor      | scatter & gather and cross-site validation using numpy |
| [sag_cse_pt](./sag_cse_pt)                      | server           | client_api           | scatter & gather workflow and cross-site evaluation with PyTorch |
| [sag_gnn](./sag_gnn)                            | server           | client_api           | scatter & gather workflow for gnn learning           |
| [sag_nemo](./sag_nemo)                          | server           | client_api           | Scatter and Gather Workflow for NeMo                 |
| [sag_np](./sag_np)                              | server           | client_api           | scatter & gather workflow using numpy                |
| [sag_np_cell_pipe](./sag_np_cell_pipe)          | server           | client_api           | scatter & gather workflow using numpy                |
| [sag_np_metrics](./sag_np_metrics)              | server           | client_api           | scatter & gather workflow using numpy                |
| [sag_pt](./sag_pt)                              | server           | client_api           | scatter & gather workflow using pytorch              |
| [sag_pt_deploy_map](./sag_pt_deploy_map)        | server           | client_api           | SAG workflow with pytorch, deploy_map, site-specific configs |
| [sag_pt_executor](./sag_pt_executor)            | server           | Executor             | scatter & gather workflow and cross-site evaluation with PyTorch |
| [sag_pt_he](./sag_pt_he)                        | server           | client_api           | scatter & gather workflow using pytorch and homomorphic encryption |
| [sag_pt_mlflow](./sag_pt_mlflow)                | server           | client_api           | scatter & gather workflow using pytorch with MLflow tracking |
| [sag_pt_model_learner](./sag_pt_model_learner)  | server           | ModelLearner          | scatter & gather workflow and cross-site evaluation with PyTorch |
| [sag_tf](./sag_tf)                              | server           | client_api           | scatter & gather workflow using TensorFlow           |
| [sklearn_kmeans](./sklearn_kmeans)              | server           | client_api           | scikit-learn KMeans model                            |
| [sklearn_linear](./sklearn_linear)              | server           | client_api           | scikit-learn linear model                            |
| [sklearn_svm](./sklearn_svm)                    | server           | client_api           | scikit-learn SVM model                               |
| [stats_df](./stats_df)                          | server           | stats executor       | FedStats: tabular data with pandas                   |
| [stats_image](./stats_image)                    | server           | stats executor       | FedStats: image intensity histogram                  |
| [swarm_cse_pt](./swarm_cse_pt)                  | client           | client_api           | Swarm Learning with Cross-Site Evaluation with PyTorch |
| [swarm_cse_pt_model_learner](./swarm_cse_pt_model_learner)  | client           | ModelLearner          | Swarm Learning with Cross-Site Evaluation with PyTorch ModelLearner |
| [vertical_xgb](./vertical_xgb)                  | server           | Executor             | vertical federated xgboost                           |
| [xgboost_tree](./xgboost_tree)                  | server           | client_api           | xgboost horizontal tree-based collaboration model  |
