# NVIDIA FLARE Examples

[NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) provides several examples to help you get started using federated learning for your own applications.

The provided examples cover different aspects of [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html),
such as using the provided [Controllers](https://nvflare.readthedocs.io/en/main/programming_guide/controllers.html)
for "scatter and gather" or "cyclic weight transfer" workflows
and different [Executors](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.apis.executor.html)
to implement your own training and validation pipelines.
Some examples use the provided "task data" and "task result" [Filters](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.apis.html?#module-nvflare.apis.filter) for homomorphic encryption and decryption or differential privacy.
Furthermore, we show how to use different components for FL algorithms such as [FedAvg](https://arxiv.org/abs/1602.05629), [FedProx](https://arxiv.org/abs/1812.06127), and [FedOpt](https://arxiv.org/abs/2003.00295).
We also provide domain-specific examples for deep learning and medical image analysis.

## Getting started
To get started with NVIDIA FLARE, please follow the [Getting Started Guide](https://nvflare.readthedocs.io/en/main/getting_started.html) in the documentation.
This walks you through installation, creating a POC workspace, and deploying your first NVIDIA FLARE Application.
The following examples will detail any additional requirements in their `requirements.txt`.

## Set up a virtual environment
We recommend setting up a virtual environment before installing the dependencies of the examples.
**You need to set up the virtual environment and install nvflare and set additional `PYTHONPATH` before launch the jupyter lab.**

Install dependencies for a virtual environment with:

```shell
python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv
```

(If needed) make all shell scripts executable using:
```shell
find . -name ".sh" -exec chmod +x {} \;
```

Create and activate your virtual environment with the `set_env.sh` script:
```shell
source ./set_env.sh
```

Install nvflare
```shell
(nvflare_example)$ pip install nvflare
```

In each example folder, install required packages for training:
```shell
(nvflare_example)$ pip install --upgrade pip
(nvflare_example)$ pip install -r requirements.txt
```

(optional) some examples contains script for plotting the TensorBoard event files, if needed, please also install:
```shell
(nvflare_example)$ pip install -r plot-requirements.txt
```

## Set up JupyterLab for notebooks
To run examples including notebooks, we recommend using [JupyterLab](https://jupyterlab.readthedocs.io).
**You need to set up the virtual environment and install nvflare and set additional `PYTHONPATH` before launch the jupyter lab.**

After activating your virtual environment, install JupyterLab:
```shell
(nvflare_example)$ pip install jupyterlab
```
You can register the virtual environment you created, so it is usable in JupyterLab:
```shell
(nvflare_example)$ python3 -m ipykernel install --user --name="nvflare_example"
```
Start a Jupyter Lab:
```shell
(nvflare_example)$ jupyter lab .
```
When you open a notebook, select the kernel `nvflare_example` using the dropdown menu at the top right.
![Selecting a JupyterLab kernel](./jupyterlab_kernel.png)

## 1. Hello World Examples
| Models                                                                                                                                 | Framework    | Notebooks                                           | Notes                  | Summary                                                                                                                                                         |
|----------------------------------------------------------------------------------------------------------------------------------------|--------------|-----------------------------------------------------|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Hello Scatter and Gather](./hello-world/hello-numpy-sag/README.md)                                                                    | Numpy        | [Yes](./hello-world/hello_world.ipynb)              | Workflow example       | Example using [ScatterAndGather](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.workflows.scatter_and_gather.html) controller workflow.      |
| [Hello Cross-Site Validation](./hello-world/hello-numpy-cross-val/README.md)                                                           | Numpy        | [Yes](./hello-world/hello_world.ipynb)              | Workflow example       | Example using [CrossSiteModelEval](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.workflows.cross_site_model_eval.html) controller workflow. |
| [Hello Cyclic Weight Transfer](./hello-world/hello-cyclic/README.md)                                                                   | PyTorch      | [Yes](./hello-world/hello_world.ipynb)              | Workflow example       | Example using [CyclicController](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.workflows.cyclic_ctl.html) controller workflow to implement [Cyclic Weight Transfer](https://pubmed.ncbi.nlm.nih.gov/29617797/). |
| [Hello PyTorch](./hello-world/hello-pt/README.md)                                                                                      | PyTorch      | [Yes](./hello-world/hello_world.ipynb)              | Deep Learning          | Example using an image classifier using [FedAvg](https://arxiv.org/abs/1602.05629) and [PyTorch](https://pytorch.org/) as the deep learning training framework. |
| [Hello TensorFlow](./hello-world/hello-tf2/README.md)                                                                                  | TensorFlow2  | [Yes](./hello-world/hello_world.ipynb)              | Deep Learning          | Example of using an image classifier using [FedAvg](https://arxiv.org/abs/1602.05629) and [TensorFlow](https://tensorflow.org/) as the deep learning training framework. |

## 2. Tutorial notebooks
| Notebook                                                                                                                               | Framework    | Notebooks                                           | Notes                  | Summary                                                                                                                                                         |
|----------------------------------------------------------------------------------------------------------------------------------------|--------------|-----------------------------------------------------|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Intro to the FL Simulator](./tutorials/flare_simulator.ipynb)                                                                         | -            | [Yes](./tutorials/flare_simulator.ipynb)            | -                      | Shows how to use the FLARE Simulator to run a local simulation.                                                                                                 |
| [Hello FLARE API](./tutorials/flare_api.ipynb)                                                                                         | -            | [Yes](./tutorials/flare_api.ipynb)                  | -                      | Goes through the different commnads of the FLARE API.                                                                                                           |
| [NVFLARE in POC Mode](./tutorials/setup_poc.ipynb)                                                                                     | -            | [Yes](./tutorials/setup_poc.ipynb)                  | -                      | Shows how to use POC mode.                                                                                                                                      |

## 3. FL algorithms
| Example                                                                         | Subsection                                                                                     | Notebooks                                                                                                            | Summary                                                                                                                                                                                                                                                                        |
|---------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Federated Learning with CIFAR-10](./advanced/cifar10/README.md)                | [Simulated Federated Learning with CIFAR-10](./advanced/cifar10/cifar10-sim/README.md)         | -                                                                                                                    | This example includes instructions on running [FedAvg](https://arxiv.org/abs/1602.05629), [FedProx](https://arxiv.org/abs/1812.06127), [FedOpt](https://arxiv.org/abs/2003.00295), and [SCAFFOLD](https://arxiv.org/abs/1910.06378) algorithms using NVFlare's FL simulator.   |
| Federated Learning with CIFAR-10                                                | [Real-world Federated Learning with CIFAR-10](./advanced/cifar10/cifar10-real-world/README.md) | -                                                                                                                    | Includes instructions on running [FedAvg](https://arxiv.org/abs/1602.05629) with streaming of TensorBoard metrics to the server during training and [homomorphic encryption](https://developer.nvidia.com/blog/federated-learning-with-homomorphic-encryption/).               |
| [Federated XGBoost](./advanced/xgboost/README.md)                               | -                                                                                              | [Federated Learning for XGBoost - Data and Job Configs](./advanced/xgboost/data_job_setup.ipynb)                     | Includes examples of histogram-based and tree-based algorithms, with details below.                                                                                                                                                                                            |
| Federated XGBoost                                                               | [Histogram-based FL for XGBoost](./advanced/xgboost/histogram-based/README.md)                 | [Histogram-based FL for XGBoost on HIGGS Dataset](./advanced/xgboost/histogram-based/xgboost_histogram_higgs.ipynb)  | Histogram-based algorithm                                                                                                                                                                                                                                                      |
| Federated XGBoost                                                               | [Tree-based Federated Learning for XGBoost ](./advanced/xgboost/tree-based/README.md)          | [Tree-based FL for XGBoost on HIGGS Dataset](./advanced/xgboost/tree-based/README.md)                                | Tree-based algorithms includes [bagging](./advanced/xgboost/tree-based/jobs/bagging_base) and [cyclic](./advanced/xgboost/tree-based/jobs/cyclic_base) approaches.                                                                                                             |

## 4. Traditional ML examples
| Example                                                                                                                                | Framework         | Notebooks                                                                                                             | Notes                  | Summary                                                                                                                                                         |
|----------------------------------------------------------------------------------------------------------------------------------------|-------------------|-----------------------------------------------------------------------------------------------------------------------|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Federated Linear Model with Scikit-learn](./advanced/sklearn-linear/README.md)                                                        | scikit-learn      | [FL Model with Scikit-learn on HIGGS Dataset](./advanced/sklearn-linear/sklearn_linear_higgs.ipynb)                   | -                      | Shows how to use the NVIDIA FLARE with [scikit-learn](https://scikit-learn.org/), a widely used open-source machine learning library.                           |
| [Federated K-Means Clustering with Scikit-learn](./advanced/sklearn-kmeans/README.md)                                                  | scikit-learn      | [Federated K-Means Clustering with Scikit-learn on Iris Dataset](./advanced/sklearn-kmeans/sklearn_kmeans_iris.ipynb) | -                      | NVIDIA FLARE with [scikit-learn](https://scikit-learn.org/) and k-Means.                                                                                        |
| [Federated SVM with Scikit-learn](./advanced/sklearn-svm/README.md)                                                                    | scikit-learn      | [Federated SVM with Scikit-learn on Breast Cancer Dataset](./advanced/sklearn-svm/sklearn_svm_cancer.ipynb)           | -                      | NVIDIA FLARE with [scikit-learn](https://scikit-learn.org/) and [SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).                  |
| [Federated Learning for Random Forest based on XGBoost](./advanced/random_forest/README.md)                                            | XGBoost           | [Federated Random Forest on HIGGS Dataset](./advanced/random_forest/random_forest.ipynb)                              | -                      | Example of using NVIDIA FLARE with [scikit-learn](https://scikit-learn.org/) and Random Forest.                                                                 |

## 5. Medical Image Analysis
| Example                                                                                                                                | Framework    | Notebooks                                           | Notes                                     | Summary                                                                                                                                                         |
|----------------------------------------------------------------------------------------------------------------------------------------|--------------|-----------------------------------------------------|-------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [NVFlare + MONAI integration](../integration/monai/README.md)                                                                          | MONAI        | -                                                   | -                                         | For an example of using NVIDIA FLARE to train a 3D medical image analysis model using federated averaging (FedAvg) and MONAI Bundle, see [here](../integration/monai/examples/README.md). |
| [Federated Learning with Differential Privacy for BraTS18 segmentation](./advanced/brats18/README.md)                                  | MONAI        | -                                                   | see requirements.txt                      | Illustrates the use of differential privacy for training brain tumor segmentation models using federated learning.                                              |
| [Federated Learning for Prostate Segmentation from Multi-source Data](./advanced/prostate/README.md)                                   | MONAI        | -                                                   | see requirements.txt                      | Example of training a multi-institutional prostate segmentation model using [FedAvg](https://arxiv.org/abs/1602.05629), [FedProx](https://arxiv.org/abs/1812.06127), and [Ditto](https://arxiv.org/abs/2012.04221). |

## 6. Federated Statistics
| Example                                                                                                                                | Framework    | Notebooks                                                                                                                                                              | Notes                  | Summary                                                                                                                                                         |
|----------------------------------------------------------------------------------------------------------------------------------------|--------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Federated Statistics Overview](./advanced/federated-statistics/README.md)                                                             | -            | -                                                                                                                                                                      | -                      | Discuss the overall federated statistics features.                                                                                                              |
| [Federated Statistics for Medical Imaging](./advanced/federated-statistics/image_stats/README.md)                                      | -            | [Image Histograms](./advanced/federated-statistics/image_stats.ipynb)                                                                                                  | see requirements.txt   | Example of gathering local image histogram to compute the global dataset histograms.                                                                            |
| [Federated Statistics for DataFrame](./advanced/federated-statistics/df_stats/README.md)                                               | -            | [Data Frame Federated Statistics](./advanced/federated-statistics/df_stats.ipynb), [Visualization](./advanced/federated-statistics/df_stats/demo/visualization.ipynb)  | see requirements.txt   | Example of gathering local statistics summary from Pandas DataFrame to compute the global dataset statistics.                                                   |

## 7. Federated Policies
| Example                                                                                                                                | Framework    | Notebooks                                           | Notes                  | Summary                                                                                                                                                         |
|----------------------------------------------------------------------------------------------------------------------------------------|--------------|-----------------------------------------------------|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Federated Policies](./advanced/federated-policies/README.rst)                                                                         | -            | -                                                   | -                      | Discuss the federated site policies for authorization, resource and data privacy management.                                                                    |

## 8. Experiment tracking
| Example                                                                                                                                | Framework    | Notebooks                                           | Notes                     | Summary                                                                                                                                                         |
|----------------------------------------------------------------------------------------------------------------------------------------|--------------|-----------------------------------------------------|---------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Hello PyTorch with TensorBoard Streaming](./advanced/experiment-tracking/tensorboard-streaming/README.md)                             | PyTorch      | -                                                   | Also requires tensorboard | Example building upon [Hello PyTorch](./hello-world/hello-pt/README.md) showcasing the [TensorBoard](https://tensorflow.org/tensorboard) streaming capability from the clients to the server.  |

## 9. NLP
| Example                                                                               | Framework     | Notebooks                                           | Notes                     | Summary                                                                                                                                                         |
|---------------------------------------------------------------------------------------|---------------|-----------------------------------------------------|---------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [NLP-NER](./advanced/nlp-ner/README.md)                                               | see details   | -                                                   | -                         | Illustrates both [BERT](https://github.com/google-research/bert) and [GPT-2](https://github.com/openai/gpt-2) models from [Hugging Face](https://huggingface.co/) ([BERT-base-uncased](https://huggingface.co/bert-base-uncased), [GPT-2](https://huggingface.co/gpt2)) on a Named Entity Recognition (NER) task using the [NCBI disease dataset](https://pubmed.ncbi.nlm.nih.gov/24393765/).  |
