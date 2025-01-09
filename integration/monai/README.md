# MONAI Integration

## Objective
Integration with [MONAI](https://monai.io/)'s federated learning capabilities.

Add `ClientAlgoExecutor` class to allow using MONAI's `ClientAlgo` class in federated scenarios.

### Goals:

Allow the use of bundles from the MONAI [model zoo](https://github.com/Project-MONAI/model-zoo) or custom configurations with NVFlare.

## Background
MONAI allows the definition of AI models using the "[bundle](https://docs.monai.io/en/latest/bundle.html)" concept. 
It allows for easy experimentation and sharing of models that have been developed using MONAI.
Using the bundle configurations, we can use MONAI's `MonaiAlgo` (the implementation of `ClientAlgo`) to execute a bundle model in a federated scenario using NVFlare.

![Federated Learning Module in MONAI (https://docs.monai.io/en/stable/modules.html#federated-learning)](https://docs.monai.io/en/stable/_images/federated.svg)

## Description
NVFlare executes the `ClientAlgo` class using the `ClientAlgoExecutor` class provided with this package.

### Examples

For an example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) to train
a medical image analysis model using federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629))
and [MONAI Bundle](https://docs.monai.io/en/latest/mb_specification.html),
see the [examples](./examples/README.md).

## Requirements

We recommend following the instructions for setting up a [virtual environment](../../examples/README.md#set-up-a-virtual-environment),
and using it in [JupyterLab](../../examples/README.md#set-up-jupyterlab-for-notebooks) for running the notebooks the MONAI integration examples.

Install MONAI-NVFlare integration from [PyPI](https://pypi.org/):
```
pip install monai_nvflare
```

(Optional) Install MONAI-NVFlare integration from source:
```
pip install -e .
```
