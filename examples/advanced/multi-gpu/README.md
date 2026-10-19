# Multi-GPU Federated Learning Examples

This directory contains examples demonstrating multi-GPU federated learning with NVFlare across different deep learning frameworks.

## Overview

Multi-GPU training allows clients to use multiple GPUs for faster local training or greater model capacity in federated learning scenarios. Each example showcases framework-specific approaches to distributed training integrated with NVFlare.

## Examples

| Framework | Strategy | Description | Directory |
|-----------|----------|-------------|-----------|
| **PyTorch** | DDP (DistributedDataParallel) | Manual DDP with NCCL backend | [`pt/`](pt/) |
| **PyTorch** | FSDP2 (Fully Sharded Data Parallel) | Full-parameter Qwen2.5-1.5B training on WikiText-2 | [`fsdp2/`](fsdp2/) |
| **PyTorch Lightning** | DDP Strategy | High-level DDP abstraction | [`lightning/`](lightning/) |

## What Each Example Demonstrates

### PyTorch DDP (`pt/`)
- PyTorch's DistributedDataParallel with manual process group management
- Per-site configuration (different master ports for different sites)
- Explicit rank coordination with NVFlare server
- Uses `torch.distributed.run` for process spawning

### PyTorch FSDP2 (`fsdp2/`)
- Fine-tunes all Qwen2.5-1.5B parameters on disjoint WikiText-2 training shards
- Shards parameters, gradients, and AdamW state across two GPUs per client
- Uses rank-zero Client API with collective full-state loading and gathering
- Runs two simulated clients on four GPUs; see its README for memory requirements

### PyTorch Lightning (`lightning/`)
- High-level Lightning DDP integration with NVFlare
- Uses `nvflare.client.lightning` API with `flare.patch(trainer)`
- Automatic handling of distributed training boilerplate
- Built-in validation, testing, and checkpointing flows

## Getting Started

### Prerequisites

**Hardware:**
- CUDA-capable GPUs; see each example for its required GPU count
- NCCL support (for PyTorch examples)

**Software:**
- Python version supported by the selected example and its framework
- CUDA toolkit
- Framework-specific dependencies (see individual READMEs)

### Quick Start

The DDP examples follow this pattern. For FSDP2, follow the [FSDP2 quick start](fsdp2/README.md#quick-start),
which uses `python prepare_data.py` to cache Qwen2.5-1.5B and prepare WikiText-2 before training:

```bash
# Navigate to example directory
cd pt/  # or lightning/

# Install dependencies
pip install -r requirements.txt

# Download dataset
bash prepare_data.sh

# Run federated learning
python job.py
```

## Troubleshooting

**Port Conflicts (PyTorch DDP)**
- Use different `--master_port` for each site

**GPU Memory Issues**
- Reduce batch size or number of GPUs

**Check GPU Availability**
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.device_count())"
```

## Additional Resources

### Documentation
- [NVFlare Documentation](https://nvflare.readthedocs.io/)
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [PyTorch Lightning DDP](https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html)
