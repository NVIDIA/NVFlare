# BioNeMo

[BioNeMo](https://www.nvidia.com/en-us/clara/bionemo/) is NVIDIA's generative AI platform for drug discovery.

This directory contains examples of running BioNeMo in a federated learning environment using [NVFlare](https://github.com/NVIDIA/NVFlare).

## Notebooks

1. The [task_fitting](./task_fitting/task_fitting.ipynb) notebook shows how to obtain protein-learned
   representations as embeddings with a pretrained ESM-2 model.
2. The [downstream](./downstream/downstream_nvflare.ipynb) notebook shows three downstream tasks for
   federated fine-tuning of a BioNeMo ESM-style model.
3. The [Evo2](./evo2/walkthrough.ipynb) notebook shows parameter-efficient federated fine-tuning of
   Evo2-1B for splice-site classification. See the [Evo2 README](./evo2/README.md) for its separately
   pinned BioNeMo Recipes and Megatron Bridge environment.

## Requirements

### ESM2 examples

<div class="alert alert-block alert-info"> <b>NOTE:</b> The ESM2 notebooks are designed to run inside the BioNeMo Framework Docker container. Follow these <a href="https://docs.nvidia.com/ai-enterprise/deployment/vmware/latest/docker.html">instructions</a> to set up your Docker environment and execute the following bash script before opening them. The Evo2 example uses the separate pinned environment documented in its README.</div>

To set up your environment, simply run (outside this notebook):

```bash
./start_bionemo.sh
```

This script will automatically pull the [BioNeMo Docker container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/containers/bionemo-framework) (tested with version nvcr.io/nvidia/clara/bionemo-framework:2.5) and launch Jupyter Lab. The Jupyter Lab interface will be available at `http://<your-hostname>:8888` where `<your-hostname>` should be replaced with your machine's hostname or IP address. Open that URL in your browser and access this notebook.

For detailed setup guidance, refer to the [BioNeMo User Guide](https://docs.nvidia.com/bionemo-framework).

### Evo2 example

Build the dedicated image and run its container helper from the Evo2 directory. The
[Evo2 setup instructions](./evo2/README.md#requirements) pin BioNeMo Recipes, Megatron Bridge, and the PyTorch
base image used by that example.
