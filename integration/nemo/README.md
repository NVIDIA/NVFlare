# NeMo Integration

## Objective
Execute [NVIDIA NeMo™](https://developer.nvidia.com/nemo) in federated environments.

### Goals:

Allow NeMo models to be trained and adapted with NVFlare.

### Non-goals:

n/a

## Background
NVIDIA NeMo™ is an end-to-end cloud-native enterprise framework for developers to 
build, customize, and deploy generative AI models with billions of parameters.

## Description
NVFlare utilizes features from NeMo, such as prompt learning to run LLM tasks in federated environments.

### Examples

For an example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) with NeMo for prompt learning, 
see [examples/prompt_learning](examples/prompt_learning/README.md) 

## Requirements

### Using docker
For simplicity, we recommend using NVIDIA's docker containers that include all the requirements for running NeMo models.
```
docker pull nvcr.io/nvidia/nemo:23.02
```

### Install NeMo-NVFlare package

<!---
#### Pip 
Install NeMo-NVFlare integration from [PyPI](https://pypi.org/):
```
pip install nemo_nvflare
```
-->

#### Mount the source code
For easy development with NeMo, install NVFlare and mount the code inside the folder.
```
pip install nvflare>=2.3.0
export PYTHONPATH=${PWD}
```

<!---
#### From source
To install the package from source code, use:
```
pip install -e .
```
-->

### Installation in a virtual environment

If preferred to install dependencies locally, 
we recommend following the instructions for setting up a 
[virtual environment](../../examples/README.md#set-up-a-virtual-environment),
and using it in [JupyterLab](../../examples/README.md#notebooks) for running 
the notebooks in the NeMo integration examples.

Follow the NeMo installation steps [here](https://github.com/NVIDIA/NeMo#installation)
before installing the NeMo-NVFlare package.
