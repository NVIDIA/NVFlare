# BioNeMo

[BioNeMo](https://www.nvidia.com/en-us/clara/bionemo/) is NVIDIA's generative AI platform for drug discovery.

This directory contains examples of running BioNeMo in a federated learning environment using [NVFlare](https://github.com/NVIDIA/NVFlare).

## Notebooks

In this repo you will find two notebooks under the `task_fitting` and `downstream` folders respectively: 
1. The [task_fitting](./task_fitting/task_fitting.ipynb) notebook example includes a notebook that shows how to obtain protein-learned representations in the form of embeddings using an ESM-2 pre-trained model. 

2. The [downstream](./downstream/downstream_nvflare.ipynb) notebook example shows three different downstream tasks for fine-tuning a BioNeMo ESM-style model.

## Requirements

Download and run the [BioNeMo docker container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/containers/bionemo-framework).
> **Note:** The examples here were tested with `nvcr.io/nvidia/clara/bionemo-framework:2.5`

We recommend following the [User Guide](https://docs.nvidia.com/bionemo-framework/latest/user-guide/)
on how to get started with BioNeMo 2.

Start the container and Jupyter Lab to run the NVFlare experiments with NVFlare using
```commandline
./start_bionemo.sh
```
It will start Jupyter Lab at `http://hostname:8888`.

For information about how to get started with BioNeMo refer to the [documentation](https://docs.nvidia.com/bionemo-framework/latest).