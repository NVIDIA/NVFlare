# MONAI Integration

## Objective
Integration with [MONAI](https://monai.io/)'s federated learning capabilities.

Add `Executor` class to allow using [MONAI bundle](https://docs.monai.io/en/latest/bundle.html) configuration files with MONAI's `ClientAlgo` class.

### Goals:

Allow the use of bundles from the MONAI [model zoo](https://github.com/Project-MONAI/model-zoo) or custom configurations with NVFlare.

### Non-goals:

n/a

## Background
MONAI allows the definition of AI models using the "bundle" concept. 
It allows for easy experimentation and sharing of models that have been developed using MONAI.
Using the bundle configurations, we can use MONAI's `MonaiAlgo` class to execute a bundle model in a federated scenario using NVFlare.

## Description
NVFlare executes the `MonaiAlgo` class using the `ClientAlgoExecutor` class provided with this package.

### Examples

For an example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) to train a medical image analysis model using federated averaging ([FedAvg]([FedAvg](https://arxiv.org/abs/1602.05629))) and [MONAI Bundle](https://docs.monai.io/en/latest/mb_specification.html), see the [examples/spleen_ct_segmentation](./examples/spleen_ct_segmentation).

## Requirements

Install MONAI-NVFlare integration from [PyPI](https://pypi.org/):
```
pip install monai_nvflare
```

(Optional) Install MONAI-NVFlare integration from source:
```
pip install -e .
```
