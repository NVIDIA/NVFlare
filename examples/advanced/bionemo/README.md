# BioNeMo

[BioNeMo](https://www.nvidia.com/en-us/clara/bionemo/) is NVIDIA's generative AI platform for drug discovery.

This directory contains examples of running BioNeMo in a federated learning environment using [NVFlare](https://github.com/NVIDIA/NVFlare).

1. The [task_fitting](./task_fitting/README.md) example includes a notebook that shows how to obtain protein learned representations in the form of embeddings using the ESM-1nv pre-trained model. 
The model is trained with NVIDIA's BioNeMo framework for Large Language Model training and inference.
2. The [downstream](./downstream/README.md) example shows three different downstream tasks for fine-tuning a BioNeMo ESM-style model.

## Requirements

Download and run the latest [BioNeMo docker container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/containers/bionemo-framework).

We recommend following the [Quickstart Guide](https://docs.nvidia.com/bionemo-framework/latest/quickstart-fw.html#docker-container-access) 
on how to get the BioNeMo container. 

First, copy the NeMo code to a local directory and configure the launch script so that downloaded models can be reused 
```commandline
CONTAINER="nvcr.io/nvidia/clara/bionemo-framework:latest"
DEST_PATH="."
CONTAINER_NAME=bionemo
docker run --name $CONTAINER_NAME -itd --rm $CONTAINER bash
docker cp $CONTAINER_NAME:/opt/nvidia/bionemo $DEST_PATH
docker kill $CONTAINER_NAME
```

Next, download the pre-trained models using
```commandline
cd ./bionemo
./launch.sh download
cd ..
```

Then, start the container and Jupyter Lab to run the NVFlare experiments with NVFlare using
```commandline
./start_bionemo.sh
```

**Note:** The examples here were tested with `nvcr.io/nvidia/clara/bionemo-framework:1.0`

For information about how to get started with BioNeMo refer to the [documentation](https://docs.nvidia.com/bionemo-framework/latest).
