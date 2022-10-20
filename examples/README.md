# NVIDIA FLARE Examples

[NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) provides several examples to help you get started using federated learning for your own applications.

The provided examples cover different aspects of [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html), such as using the provided [Controllers](https://nvflare.readthedocs.io/en/main/programming_guide/controllers.html) for "scatter and gather" or "cyclic weight transfer" workflows and example [Executors](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.apis.executor.html) to implement your own training and validation pipelines. Some examples use the provided "task data" and "task result" [Filters](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.apis.html?#module-nvflare.apis.filter) for homomorphic encryption and decryption or differential privacy. Furthermore, we show how to use different components for FL algorithms such as [FedAvg](https://arxiv.org/abs/1602.05629), [FedProx](https://arxiv.org/abs/1812.06127), and [FedOpt](https://arxiv.org/abs/2003.00295). We also provide domain-specific examples for deep learning and medical image analysis.

> **_NOTE:_** To run examples, please follow the instructions for [Installation](https://nvflare.readthedocs.io/en/main/quickstart.html) and any additional steps specified in the example readmes.

## 0. Quickstart
To get started with these examples, please follow the [Quickstart](https://nvflare.readthedocs.io/en/main/quickstart.html)in the NVIDIA FLARE Documentation.  This walks you through installation, creating a POC workspace, and deploying your first NVIDIA FLARE Application.  The following examples will detail any additional requirements in their READMEs.
## 1. Hello World Examples
### 1.1 Workflows
* [Hello Scatter and Gather](./hello-numpy-sag/README.md)
    * Example using "[ScatterAndGather](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.workflows.scatter_and_gather.html)" controller workflow.
* [Hello Cross-Site Validation](./hello-numpy-cross-val/README.md)
    * Example using [CrossSiteModelEval](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.workflows.cross_site_model_eval.html) controller workflow.
* [Hello Cyclic Weight Transfer](./hello-cyclic/README.md)
    * Example using [CyclicController](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.workflows.cyclic_ctl.html) controller workflow to implement [Cyclic Weight Transfer](https://pubmed.ncbi.nlm.nih.gov/29617797/).
### 1.2 Deep Learning
* [Hello PyTorch](./hello-pt/README.md)
  * Example using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) an image classifier using [FedAvg]([FedAvg](https://arxiv.org/abs/1602.05629)) and [PyTorch](https://pytorch.org/) as the deep learning training framework.
* [Hello PyTorch with TensorBoard](./hello-pt-tb/README.md)
  * Example building upon [Hello PyTorch](./hello-pt/README.md) showcasing the [TensorBoard](https://tensorflow.org/tensorboard) streaming capability from the clients to the server.
* [Hello TensorFlow](./hello-tf2/README.md)
  * Example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) an image classifier using [FedAvg]([FedAvg](https://arxiv.org/abs/1602.05629)) and [TensorFlow](https://tensorflow.org/) as the deep learning training framework.

## 2. FL algorithms
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
  * Includes examples of [histogram-based](./xgboost/histogram-based/README.md) algorithm, [tree-based](./xgboost/tree-based/README.md). Tree-based algorithms also includes [bagging](./xgboost/tree-based/job_configs/bagging_base) and [cyclic](./xgboost/tree-based/job_configs/cyclic_base) approaches 

## 3. Medical Image Analysis
* [Hello MONAI Bundle](./hello-monai-bundle/README.md)
  * For an example of using NVIDIA FLARE to train a 3D medical image analysis model using federated averaging (FedAvg) and MONAI Bundle, see [../integration/monai/examples/README.md](../integration/monai/examples/README.md).
* [Federated Learning with Differential Privacy for BraTS18 segmentation](./brats18/README.md)
   * Illustrates the use of differential privacy for training brain tumor segmentation models using federated learning.
* [Federated Learning for Prostate Segmentation from Multi-source Data](./prostate/README.md)
  * Example of training a multi-institutional prostate segmentation model using [FedAvg](https://arxiv.org/abs/1602.05629), [FedProx](https://arxiv.org/abs/1812.06127), and [Ditto](https://arxiv.org/abs/2012.04221).

## 4. Federated Statistics
* [Federated Statistic Overview](federated_statistics/README.md)
  * Discuss the overall federated statistics features 
* [Federated Statistics for Medical Imaging](federated_statistics/image_stats/README.md)
  * Example of gathering local image histogram to compute the global dataset histograms.
* [Federated Statistics for DataFrame](federated_statistics/df_stats/README.md)
  * Example of gathering local statistics summary from Pandas DataFrame to compute the global dataset statistics.

## 5. Federated Policies
* [Federated Policies](./federated-policies/README.rst) 
  * Discuss the federated site policies for authorization, resource and data privacy management
