# Federated Protein Embeddings and Task Model Fitting with BioNeMo

This example notebook shows how to obtain protein learned representations in the form of embeddings using the ESM-1nv pre-trained model. 
The model is trained with NVIDIA's BioNeMo framework for Large Language Model training and inference. 

This example is based on NVIDIA BioNeMo Service [example](https://github.com/NVIDIA/BioNeMo/blob/main/examples/service/notebooks/task-fitting-predictor.ipynb) 
but runs inference locally (on the FL clients) instead of using BioNeMo's cloud API.

## 1. Install requirements

Follow the instructions provide [here](../README.md#requirements) on how to start the BioNeMo container.

Install required packages for training inside the BioNeMo container (normally using a JupyterLab terminal window).
```
pip install --upgrade pip
pip install -r ./requirements.txt
```

## 2. Run experiments

Open the example notebook [task_fitting.ipynb](./task_fitting.ipynb).
