# NVIDIA FLARE Examples

[NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) provides several examples to help you get started using federated learning for your own applications.

The provided examples cover different aspects of [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html), such as using the provided [Controllers](https://nvflare.readthedocs.io/en/main/programming_guide/controllers.html) for "scatter and gather" or "cyclic weight transfer" workflows and example [Executors](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.apis.executor.html) to implement your own training and validation pipelines. Some examples use the provided "task data" and "task result" [Filters](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.apis.html?#module-nvflare.apis.filter) for homomorphic encryption and decryption or differential privacy. Furthermore, we show how to use different components for FL algorithms such as [FedAvg](https://arxiv.org/abs/1602.05629), [FedProx](https://arxiv.org/abs/1812.06127), and [FedOpt](https://arxiv.org/abs/2003.00295). We also provide domain-specific examples for deep learning and medical image analysis.

> **_NOTE:_** To run examples, please follow the instructions for [Installation](https://nvflare.readthedocs.io/en/main/quickstart.html) and any additional steps specified in the example readmes.

## Getting started
To get started with NVIDIA FLARE, please follow the [Getting Started Guide](https://nvflare.readthedocs.io/en/main/getting_started.html) in the documentation.
This walks you through installation, creating a POC workspace, and deploying your first NVIDIA FLARE Application.
The following examples will detail any additional requirements in their `requirements.txt`.

## Set up a virtual environment
We recommend setting up a virtual environment before installing the dependencies of the examples.
```
python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv
```
(If needed) make all shell scripts executable using
```
find . -name ".sh" -exec chmod +x {} \;
```
activate your virtual environment.
```
source ./set_env.sh
```
within each example folder, install required packages for training
```
pip install --upgrade pip
pip install -r requirements.txt
```
(optional) some examples contains script for plotting the TensorBoard event files, if needed, please also install
```
pip install -r plot-requirements.txt
```

## Notebooks
To run examples including notebooks, we recommend using [JupyterLab](https://jupyterlab.readthedocs.io).

After activating your virtual environment, install JupyterLab
```
pip install jupyterlab
```
Register the virtual environment kernel
```
python -m ipykernel install --user --name="nvflare_example"
```
Start a Jupyter Lab
```
jupyter lab .
```
When you open a notebook, select the kernel `nvflare_example` using the dropdown menu at the top right.
![Selecting a JupyterLab kernel](./jupyterlab_kernel.png)

## 1. Hello World Examples
### 1.1 Workflows
* [Hello Scatter and Gather](./hello-world/hello-numpy-sag/README.md)
    * Example using "[ScatterAndGather](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.workflows.scatter_and_gather.html)" controller workflow.
* [Hello Cross-Site Validation](./hello-world/hello-numpy-cross-val/README.md)
    * Example using [CrossSiteModelEval](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.workflows.cross_site_model_eval.html) controller workflow.
* [Hello Cyclic Weight Transfer](./hello-world/hello-cyclic/README.md)
    * Example using [CyclicController](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.workflows.cyclic_ctl.html) controller workflow to implement [Cyclic Weight Transfer](https://pubmed.ncbi.nlm.nih.gov/29617797/).
### 1.2 Deep Learning
* [Hello PyTorch](./hello-world/hello-pt/README.md)
  * Example using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) an image classifier using [FedAvg]([FedAvg](https://arxiv.org/abs/1602.05629)) and [PyTorch](https://pytorch.org/) as the deep learning training framework.
* [Hello TensorFlow](./hello-world/hello-tf2/README.md)
  * Example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) an image classifier using [FedAvg]([FedAvg](https://arxiv.org/abs/1602.05629)) and [TensorFlow](https://tensorflow.org/) as the deep learning training framework.

## 2. Tutorial notebooks
* [Intro to the FL Simulator](./tutorials/flare_simulator.ipynb)
  * Shows how to use the FLARE Simulator to run a local simulation.
* [Intro to the FL Simulator](./tutorials/flare_api.ipynb)
  * Goes through the different commnads of the FLARE API.
* [Intro to the FL Simulator](./tutorials/setup_poc.ipynb)
  * Shows how to use POC mode.

## 3. FL algorithms
* [Federated Learning with CIFAR-10](./advanced/cifar10/README.md)
  * [Simulated Federated Learning with CIFAR-10](./advanced/cifar10/cifar10-sim/README.md)
    * This example includes instructions on running [FedAvg](https://arxiv.org/abs/1602.05629), 
  [FedProx](https://arxiv.org/abs/1812.06127), [FedOpt](https://arxiv.org/abs/2003.00295), 
  and [SCAFFOLD](https://arxiv.org/abs/1910.06378) algorithms using NVFlare's FL simulator.
  * [Real-world Federated Learning with CIFAR-10](./advanced/cifar10/cifar10-real-world/README.md)
    * Includes instructions on running [FedAvg](https://arxiv.org/abs/1602.05629) with streaming 
  of TensorBoard metrics to the server during training 
  and [homomorphic encryption](https://developer.nvidia.com/blog/federated-learning-with-homomorphic-encryption/).
* [Federated XGBoost](./advanced/xgboost/README.md)
  * Includes examples of [histogram-based](./advanced/xgboost/histogram-based/README.md) algorithm, [tree-based](./advanced/xgboost/tree-based/README.md). Tree-based algorithms also includes [bagging](./advanced/xgboost/tree-based/job_configs/bagging_base) and [cyclic](./advanced/xgboost/tree-based/job_configs/cyclic_base) approaches 

## 4. Traditional ML examples
* [Federated Linear Model with Scikit-learn](./advanced/sklearn-linear/README.md)
  * Shows how to use the NVIDIA FLARE with [scikit-learn](https://scikit-learn.org/), a widely used open-source machine learning library.
* [Federated K-Means Clustering with Scikit-learn](./advanced/sklearn-kmeans/README.md)
  * NVIDIA FLARE with [scikit-learn](https://scikit-learn.org/) and k-Means.
* [Federated SVM with Scikit-learn](./advanced/sklearn-svm/README.md)
  * NVIDIA FLARE with [scikit-learn](https://scikit-learn.org/) and [SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).
* [Federated Learning for Random Forest based on XGBoost](./advanced/random_forest/README.md)
  * Example of using NVIDIA FLARE with [scikit-learn](https://scikit-learn.org/) and Random Forest.

## 5. Medical Image Analysis
* [NVFlare + MONAI integration](../integration/monai/README.md)
  * For an example of using NVIDIA FLARE to train a 3D medical image analysis model using federated averaging (FedAvg) and MONAI Bundle, see [../integration/monai/examples/README.md](../integration/monai/examples/README.md).
* [Federated Learning with Differential Privacy for BraTS18 segmentation](./advanced/brats18/README.md)
   * Illustrates the use of differential privacy for training brain tumor segmentation models using federated learning.
* [Federated Learning for Prostate Segmentation from Multi-source Data](./advanced/prostate/README.md)
  * Example of training a multi-institutional prostate segmentation model using [FedAvg](https://arxiv.org/abs/1602.05629), [FedProx](https://arxiv.org/abs/1812.06127), and [Ditto](https://arxiv.org/abs/2012.04221).

## 6. Federated Statistics
* [Federated Statistic Overview](./advanced/federated-statistics/README.md)
  * Discuss the overall federated statistics features 
* [Federated Statistics for Medical Imaging](./advanced/federated-statistics/image_stats/README.md)
  * Example of gathering local image histogram to compute the global dataset histograms.
* [Federated Statistics for DataFrame](./advanced/federated-statistics/df_stats/README.md)
  * Example of gathering local statistics summary from Pandas DataFrame to compute the global dataset statistics.

## 7. Federated Policies
* [Federated Policies](./advanced/federated-policies/README.rst) 
  * Discuss the federated site policies for authorization, resource and data privacy management

## 8. Experiment tracking
* [Hello PyTorch with TensorBoard Streaming](./advanced/tensorboard-streaming/README.md)
  * Example building upon [Hello PyTorch](./basic/hello-pt/README.md) showcasing the [TensorBoard](https://tensorflow.org/tensorboard) streaming capability from the clients to the server.
