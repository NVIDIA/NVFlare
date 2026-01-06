# Multi-GPU Federated Learning Examples

This directory contains examples demonstrating multi-GPU federated learning with NVFlare across different deep learning frameworks.

## Overview

Multi-GPU training allows clients to leverage multiple GPUs for faster local training in federated learning scenarios. Each example showcases framework-specific approaches to distributed training integrated with NVFlare.

## Examples

| Framework | Strategy | Description | Directory |
|-----------|----------|-------------|-----------|
| **PyTorch** | DDP (DistributedDataParallel) | Manual DDP with NCCL backend | [`pt/`](pt/) |
| **TensorFlow** | MirroredStrategy | Automatic multi-GPU distribution | [`tf/`](tf/) |
| **PyTorch Lightning** | DDP Strategy | High-level DDP abstraction | [`lightning/`](lightning/) |

## What Each Example Demonstrates

### PyTorch DDP (`pt/`)
- PyTorch's DistributedDataParallel with manual process group management
- Per-site configuration (different master ports for different sites)
- Explicit rank coordination with NVFlare server
- Uses `torch.distributed.run` for process spawning

### TensorFlow MirroredStrategy (`tf/`)
- TensorFlow's MirroredStrategy for synchronous multi-GPU training
- Automatic GPU detection and distribution
- Variables created within `strategy.scope()`
- Single-process multi-device training

### PyTorch Lightning (`lightning/`)
- High-level Lightning DDP integration with NVFlare
- Uses `nvflare.client.lightning` API with `flare.patch(trainer)`
- Automatic handling of distributed training boilerplate
- Built-in validation, testing, and checkpointing flows

## Getting Started

### Prerequisites

**Hardware:**
- 2+ CUDA-capable GPUs
- NCCL support (for PyTorch examples)

**Software:**
- Python 3.8+
- CUDA toolkit
- Framework-specific dependencies (see individual READMEs)

### Quick Start

Each example follows the same pattern:

```bash
# Navigate to example directory
cd pt/  # or tf/ or lightning/

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
- TensorFlow: `TF_FORCE_GPU_ALLOW_GROWTH=true python job.py`

**Check GPU Availability**
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.device_count())"
```

## Additional Resources

### Documentation
- [NVFlare Documentation](https://nvflare.readthedocs.io/)
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [TensorFlow Distributed Guide](https://www.tensorflow.org/guide/distributed_training)
- [PyTorch Lightning DDP](https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html)
