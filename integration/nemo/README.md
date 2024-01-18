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

For an example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) with NeMo for supervised fine-tuning (SFT), 
see [examples/supervised_fine_tuning](examples/supervised_fine_tuning/README.md) 

## Requirements

### Using docker (Recommended)
For simplicity, we recommend using NVIDIA's [NeMo docker containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo) that include all the requirements for running NeMo models.

> Note: each example in this folder might require different container version. Please check their Readmes for details. 

### Installation in a virtual environment

If preferred to install dependencies locally, 
we recommend following the instructions for setting up a 
[virtual environment](../../examples/README.md#set-up-a-virtual-environment),
and using it in [JupyterLab](../../examples/README.md#notebooks) for running 
the notebooks in the NeMo integration examples.

Follow the NeMo installation steps [here](https://github.com/NVIDIA/NeMo#installation)
before installing NVFlare and adding the source to the PYTHONPATH.
