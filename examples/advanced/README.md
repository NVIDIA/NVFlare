# NVFlare advanced examples

This folder contains advanced examples for NVFlare.

Please make sure you set up a virtual environment and install JupyterLab following the [example root readme](../README.md).

Please also install "./requirements.txt" in each example folder.

## FL algorithms
* [Federated Learning with CIFAR-10](./cifar10/README.md)
  * [Simulated Federated Learning with CIFAR-10](./cifar10/cifar10-sim/README.md)
    * This example includes instructions on running [FedAvg](https://arxiv.org/abs/1602.05629), 
  [FedProx](https://arxiv.org/abs/1812.06127), [FedOpt](https://arxiv.org/abs/2003.00295), 
  and [SCAFFOLD](https://arxiv.org/abs/1910.06378) algorithms using NVFlare's FL simulator.
  * [Real-world Federated Learning with CIFAR-10](./cifar10/cifar10-real-world/README.md)
    * Includes instructions on running [FedAvg](https://arxiv.org/abs/1602.05629) with streaming 
  of TensorBoard metrics to the server during training 
  and [homomorphic encryption](https://developer.nvidia.com/blog/federated-learning-with-homomorphic-encryption/).
* [Federated XGBoost](./xgboost/README.md)
  * Includes examples of [histogram-based](./xgboost/histogram-based/README.md) algorithm, [tree-based](./xgboost/tree-based/README.md).
    Tree-based algorithms also include [bagging](./xgboost/tree-based/jobs/bagging_base) and [cyclic](./xgboost/tree-based/jobs/cyclic_base) approaches.

## Traditional ML examples
* [Federated Linear Model with Scikit-learn](./sklearn-linear/README.md)
  * Shows how to use the NVIDIA FLARE with [scikit-learn](https://scikit-learn.org/), a widely used open-source machine learning library.
* [Federated K-Means Clustering with Scikit-learn](./sklearn-kmeans/README.md)
  * NVIDIA FLARE with [scikit-learn](https://scikit-learn.org/) and k-Means.
* [Federated SVM with Scikit-learn](./sklearn-svm/README.md)
  * NVIDIA FLARE with [scikit-learn](https://scikit-learn.org/) and [SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).
* [Federated Learning for Random Forest based on XGBoost](./random_forest/README.md)
  * Example of using NVIDIA FLARE with [scikit-learn](https://scikit-learn.org/) and Random Forest.

## Medical Image Analysis
* [Federated Learning with Differential Privacy for BraTS18 segmentation](./brats18/README.md)
   * Illustrates the use of differential privacy for training brain tumor segmentation models using federated learning.
* [Federated Learning for Prostate Segmentation from Multi-source Data](./prostate/README.md)
  * Example of training a multi-institutional prostate segmentation model using [FedAvg](https://arxiv.org/abs/1602.05629), [FedProx](https://arxiv.org/abs/1812.06127), and [Ditto](https://arxiv.org/abs/2012.04221).

## Finance
* [Financial Application with Federated XGBoost Methods](./finance/README.md)
   * Illustrates the use of NVFlare on a financial application using XGBoost to train a model in a federated manner.

## Swarm Learning
* [Swarm Learning](./swarm_learning/README.md)
   * Example of swarm learning with NVIDIA FLARE using PyTorch with the CIFAR-10 dataset.

## Distributed Optimization / P2P algorithms
* [Distributed Optimization](./distributed_optimization/README.md)
   * Example of using the low-level NVFlare APIs to implement and run P2P distributed optimization algorithms.

## Vertical Federated Learning
* [Vertical Federated Learning](./vertical_federated_learning/README.md)
   * Example of running split learning using the CIFAR-10 dataset.
* [Vertical Federated XGBoost](./vertical_xgboost/README.md)
   * Example of vertical federated learning with NVIDIA FLARE on tabular data.

## Federated Statistics
* [Federated Statistic Overview](./federated-statistics/README.md)
  * Discuss the overall federated statistics features 
* [Federated Statistics for Medical Imaging](./federated-statistics/image_stats/README.md)
  * Example of gathering local image histogram to compute the global dataset histograms.
* [Federated Statistics for DataFrame](./federated-statistics/df_stats/README.md)
  * Example of gathering local statistics summary from Pandas DataFrame to compute the global dataset statistics.
* [Federated Hierarchical Statistics](./federated-statistics/hierarchical_stats/README.md)
  * Example of generating federated hierarchical statistics for data that can be represented as Pandas DataFrame.

## Federated Policies
* [Federated Policies](./federated-policies/README.rst) 
  * Discuss the federated site policies for authorization, resource and data privacy management

## Custom Authentication
* [Custom Authentication](./custom_authentication/README.rst) 
  * Example demonstrating custom authentication policy

## Job-level Authorization
* [Job-level Authorization](./job-level-authorization/README.md) 
  * Example demonstrating job-level authorization policy

## Experiment tracking
* [Hello PyTorch with TensorBoard Streaming](./experiment-tracking/tensorboard/README.md)
  * Example building upon [Hello PyTorch](../hello-world/hello-pt/README.md) showcasing the [TensorBoard](https://tensorflow.org/tensorboard) streaming capability from the clients to the server.
* [Experiment Tracking with MLflow and Weights and Biases](./experiment-tracking/README.md)
  * Example showing the use of the Writers and Receivers in NVFlare to write to different experiment tracking systems.

## Federated Learning Hub

* [FL Hub](./fl_hub/README.md) 
  * Allow hierarchical interaction between several levels of nvflare FL systems, e.g. Tier-1 (hub) and Tier-2 (sub-systems).
