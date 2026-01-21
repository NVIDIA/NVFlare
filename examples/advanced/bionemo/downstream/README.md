# Downstream: Federated Fine-tuning of ESM2 Models

This example demonstrates three different downstream tasks for fine-tuning a BioNeMo ESM2-style model using federated learning with NVIDIA FLARE.

## Interactive Notebook

**👉 Start here: [downstream_nvflare.ipynb](./downstream_nvflare.ipynb)**

The notebook provides step-by-step instructions for running all three downstream tasks with different configurations.

## Tasks

1. **SAbDab** (`sabdab/`): Antibody binding classification (binary: pos/neg)
2. **TAP** (`tap/`): Thermostability regression (multiple endpoints per client)
3. **SCL** (`scl/`): Subcellular location classification (10 classes)

## Quick Start

Run any task with:

```bash
cd sabdab  # or tap, or scl
python job.py --num_clients 2 --num_rounds 30 --model 8m
```

## Structure

```
downstream/
├── sabdab/              # Antibody binding task
├── tap/                 # Thermostability task
├── scl/                 # Subcellular location task
├── client.py            # Shared client training script
├── bionemo_filters.py   # Custom filters for BioNeMo models
└── downstream_nvflare.ipynb  # Interactive notebook version
```

## Features

- **Client-side data resolution**: Automatically resolves per-client data paths
- **Custom filters**: BioNeMo-specific parameter and state dict filtering
- **Flexible configuration**: Support for central, local, and federated training
- **Multiple model sizes**: 8m, 650m, 3b parameter ESM2 models

See individual task folders and `downstream_nvflare.ipynb` for detailed documentation.

## Reference

Uses NVIDIA BioNeMo Framework for model training and NVIDIA FLARE for federated orchestration.
