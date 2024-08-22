# Getting Started with NVFlare (PyTorch)
[![PyTorch Logo](https://upload.wikimedia.org/wikipedia/commons/c/c6/PyTorch_logo_black.svg)](https://pytorch.org)

We provide several examples to quickly get you started using NVFlare's Job API. 
All examples in this folder are based on using [PyTorch](https://pytorch.org/) as the model training framework.
Furthermore, we support [PyTorch Lightning](https://lightning.ai).

## Setup environment
First, install nvflare and dependencies:
```commandline
pip install -r requirements.txt
```

## Tutorials
A good starting point for understanding the Job API scripts and NVFlare components are the following tutorials.
### 1. [Federated averaging using script executor](./nvflare_pt_getting_started.ipynb)
Tutorial on [FedAvg](https://arxiv.org/abs/1602.05629) using the [Client API](https://nvflare.readthedocs.io/en/main/programming_guide/execution_api_type/client_api.html).

### 2. [Federated averaging using script executor with Lightning API](./nvflare_lightning_getting_started.ipynb)
Tutorial on [FedAvg](https://arxiv.org/abs/1602.05629) using the [Lightning Client API](https://nvflare.readthedocs.io/en/main/programming_guide/execution_api_type/client_api.html#id4)

## Examples
You can also run any of the below scripts directly using
```commandline
python "script_name.py"
```
### 1. [Federated averaging using script executor](./fedavg_script_runner_cifar10.py)
Implementation of [FedAvg](https://arxiv.org/abs/1602.05629) using the [Client API](https://nvflare.readthedocs.io/en/main/programming_guide/execution_api_type/client_api.html).
```commandline
python fedavg_script_runner_cifar10.py
```
### 2. [Federated averaging using script executor with Lightning API](./fedavg_script_runner_lightning_cifar10.py)
Implementation of [FedAvg](https://arxiv.org/abs/1602.05629) using the [Lightning Client API](https://nvflare.readthedocs.io/en/main/programming_guide/execution_api_type/client_api.html#id4)
```commandline
python fedavg_script_runner_lightning_cifar10.py
```
### 3. [Federated averaging using the script executor for all clients](./fedavg_script_runner_cifar10_all.py)
Implementation of [FedAvg](https://arxiv.org/abs/1602.05629) using the [Client API](https://nvflare.readthedocs.io/en/main/programming_guide/execution_api_type/client_api.html).
Here, we deploy the same configuration to all clients.
```commandline
python fedavg_script_runner_cifar10_all.py
```
### 4. [Federated averaging using script executor and differential privacy filter](./fedavg_script_runner_dp_filter_cifar10.py)
Implementation of [FedAvg](https://arxiv.org/abs/1602.05629) using the [Client API](https://nvflare.readthedocs.io/en/main/programming_guide/execution_api_type/client_api.html)
with additional [differential privacy filters](https://arxiv.org/abs/1910.00962) on the client side.
```commandline
python fedavg_script_runner_dp_filter_cifar10.py
```
### 5. [Swarm learning using script executor](./swarm_script_runner_cifar10.py)
Implementation of [swarm learning](https://www.nature.com/articles/s41586-021-03583-3) using the [Client API](https://nvflare.readthedocs.io/en/main/programming_guide/execution_api_type/client_api.html)
```commandline
python swarm_script_runner_cifar10.py
```
### 6. [Cyclic weight transfer using script executor](./cyclic_cc_script_runner_cifar10.py)
Implementation of [cyclic weight transfer](https://arxiv.org/abs/1709.05929) using the [Client API](https://nvflare.readthedocs.io/en/main/programming_guide/execution_api_type/client_api.html)
```commandline
python cyclic_cc_script_runner_cifar10.py
```
### 7. [Federated averaging using model learning](./fedavg_model_learner_xsite_val_cifar10.py))
Implementation of [FedAvg](https://arxiv.org/abs/1602.05629) using the [model learner class](https://nvflare.readthedocs.io/en/main/programming_guide/execution_api_type/model_learner.html),
followed by [cross site validation](https://nvflare.readthedocs.io/en/main/programming_guide/controllers/cross_site_model_evaluation.html)
for federated model evaluation.
```commandline
python fedavg_model_learner_xsite_val_cifar10.py
```

> [!NOTE]
> More examples can be found at https://nvidia.github.io/NVFlare.
