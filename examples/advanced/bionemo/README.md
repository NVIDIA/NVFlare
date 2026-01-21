# BioNeMo

[BioNeMo](https://www.nvidia.com/en-us/clara/bionemo/) is NVIDIA's generative AI platform for drug discovery.

This directory contains examples of running BioNeMo in a federated learning environment using [NVFlare](https://github.com/NVIDIA/NVFlare).

## Notebooks

In this repo you will find two notebooks under the `task_fitting` and `downstream` folders respectively: 
1. The [task_fitting](./task_fitting/task_fitting.ipynb) notebook example includes a notebook that shows how to obtain protein-learned representations in the form of embeddings using an ESM-2 pre-trained model. 

2. The [downstream](./downstream/downstream_nvflare.ipynb) notebook example shows three different downstream tasks for fine-tuning a BioNeMo ESM-style model.

## Requirements

<div class="alert alert-block alert-info"> <b>NOTE:</b> This notebook is designed to run inside the BioNeMo Framework Docker container. Follow these <a href="https://docs.nvidia.com/ai-enterprise/deployment/vmware/latest/docker.html">instructions</a> to set up your Docker environment and execute the following bash script before opening this notebook.</div>

To set up your environment, simply run (outside this notebook):

```bash
./start_bionemo.sh
```

This script will automatically pull the [BioNeMo Docker container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/containers/bionemo-framework) (tested with version nvcr.io/nvidia/clara/bionemo-framework:2.5) and launch Jupyter Lab. The Jupyter Lab interface will be available at `http://<your-hostname>:8888` where `<your-hostname>` should be replaced with your machine's hostname or IP address. Open that URL in your browser and access this notebook.

For detailed setup guidance, refer to the [BioNeMo User Guide](https://docs.nvidia.com/bionemo-framework).
