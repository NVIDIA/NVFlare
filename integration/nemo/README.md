# NeMo Integration

[NVIDIA NeMo™](https://developer.nvidia.com/nemo) is an end-to-end cloud-native enterprise framework for developers to 
build, customize, and deploy generative AI models with billions of parameters.

Here, we show how NVFlare utilizes features from NeMo to run LLM tasks in federated environments with several [examples](./examples).

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
