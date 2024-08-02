# Getting Started with NVFlare (TensorFlow)
[![TensorFlow Logo](https://upload.wikimedia.org/wikipedia/commons/a/ab/TensorFlow_logo.svg)](https://tensorflow.org/)

We provide several examples to quickly get you started using NVFlare's Job API. 
All examples in this folder are based on using [TensorFlow](https://tensorflow.org/) as the model training framework.

## Setup environment
First, install nvflare and dependencies:
```commandline
pip install -r requirements.txt
```

## Tutorials
A good starting point for understanding the Job API scripts and NVFlare components is the following tutorial.
### 1. [Federated averaging using script executor](./nvflare_tf_getting_started.ipynb)
Tutorial on [FedAvg](https://arxiv.org/abs/1602.05629) using the [Client API](https://nvflare.readthedocs.io/en/main/programming_guide/execution_api_type/client_api.html).

## Examples
You can also run any of the below scripts directly using
```commandline
python "script_name.py"
```
### 1. [Federated averaging using script executor](./tf_fedavg_script_executor_cifar10.py)
Implementation of [FedAvg](https://arxiv.org/abs/1602.05629) using the [Client API](https://nvflare.readthedocs.io/en/main/programming_guide/execution_api_type/client_api.html).
```commandline
python tf_fedavg_script_executor_cifar10.py
```
