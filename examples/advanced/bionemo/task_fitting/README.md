# Task Fitting: Federated Protein Embeddings and MLP Training

This example shows how to obtain protein-learned representations in the form of embeddings using an ESM-2 pre-trained model in a federated learning setting, followed by training a Multi-Layer Perceptron (MLP) classifier to predict subcellular location.

## Interactive Notebook

**👉 Start here: [task_fitting.ipynb](./task_fitting.ipynb)**

The notebook provides an interactive walkthrough including data preparation, embedding extraction, and MLP training.

## Workflow

1. **Embedding Extraction** (`job_inference/`): Extract protein embeddings using ESM2 model
2. **MLP Training** (`job_fedavg/`): Train a classifier on embeddings using FedAvg

## Quick Start

```bash
# 1. Prepare data (see task_fitting.ipynb)
# 2. Extract embeddings
cd job_inference && python job.py

# 3. Train MLP classifier
cd ../job_fedavg && python job.py
```

## Structure

```
task_fitting/
├── job_inference/       # ESM2 embedding extraction job
├── job_fedavg/          # MLP classifier training job  
├── task_fitting.ipynb   # Interactive notebook version
└── split_data.py        # Data splitting utility
```

See individual job folders for detailed documentation.

## Reference

Based on the NVIDIA BioNeMo Service [task-fitting example](https://github.com/NVIDIA/digital-biology-examples/blob/api/examples/service/notebooks/task-fitting-predictor.ipynb), adapted to run locally with NVIDIA FLARE.
