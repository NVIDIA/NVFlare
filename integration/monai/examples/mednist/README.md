## Converting MONAI Code to a Federated Learning Setting

In this tutorial, we will introduce how simple it can be to run an end-to-end classification pipeline with MONAI 
and deploy it in a federated learning setting using NVFlare.

### 1. Standalone training with MONAI
[monai_101.ipynb](./monai_101.ipynb) is based on the [MONAI 101 classification tutorial](https://github.com/Project-MONAI/tutorials/blob/main/2d_classification/monai_101.ipynb) and shows each step required in only a  few lines of code, including

- Dataset download
- Data pre-processing
- Define a DenseNet-121 and run training
- Check the results on test dataset

### 2. Federated learning with MONAI
[monai_101_fl.ipynb](./monai_101_fl.ipynb) shows how we can simply put the code introduced above into a Python script and convert it to running in an FL scenario using NVFlare.

To achieve this, we utilize the [`FedAvg`](https://arxiv.org/abs/1602.05629) algorithm and NVFlare's [Client
API](https://nvflare.readthedocs.io/en/main/programming_guide/execution_api_type.html#client-api).
