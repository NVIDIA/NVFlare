# Multi-GPU Federated Learning with PyTorch Lightning

This example demonstrates federated learning with PyTorch Lightning using DDP strategy for multi-GPU training.

## Overview

This example showcases how to use PyTorch Lightning's DDP (Distributed Data Parallel) strategy with NVFlare for federated learning across multiple GPUs. PyTorch Lightning provides a high-level abstraction that simplifies multi-GPU training.

## Key Features

- **PyTorch Lightning DDP**: High-level distributed training with minimal boilerplate
- **Automatic Distribution**: Lightning handles process spawning and synchronization
- **NVFlare Integration**: Uses `nvflare.client.lightning` for seamless integration
- **CIFAR-10 Dataset**: Trains a CNN on CIFAR-10 for image classification

## Quick Start

1. **Prepare data:**
```bash
bash ./prepare_data.sh
```

2. **Run federated learning:**
```bash
python job.py
```

## Project Structure

```
lightning/
├── job.py              # Job configuration
├── model.py            # LitNet Lightning module
├── client.py           # Lightning DDP client script
├── prepare_data.sh     # Script to download CIFAR-10
└── requirements.txt    # Python dependencies
```

## Command Line Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--n_clients` | Number of clients | 2 |
| `--num_rounds` | Number of training rounds | 5 |
| `--use_tracking` | Enable TensorBoard tracking | False |
| `--export_config` | Export job config only | False |

## Examples

```bash
# Run with default settings (2 clients, 5 rounds)
python job.py

# Run with 3 clients and 10 rounds
python job.py --n_clients 3 --num_rounds 10

# Run with TensorBoard tracking
python job.py --use_tracking

# Export config for deployment
python job.py --export_config
```

## Requirements

```bash
pip install -r requirements.txt
```

### Hardware Requirements
- 2+ GPUs (configured with `devices=2` in Trainer)
- CUDA-capable GPUs

### Software Requirements
- PyTorch with CUDA support
- PyTorch Lightning
- torchvision
- nvflare with PyTorch support

## Troubleshooting

### GPU Memory Issues
- Reduce `batch_size` in the DataModule
- Reduce number of GPUs with `devices` in Trainer
- Use gradient accumulation in Trainer

### No GPUs Detected
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.device_count())"
```

## Additional Resources

- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)
- [Lightning DDP Guide](https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html)
- [NVFlare Lightning Integration](https://nvflare.readthedocs.io/)
- [Lightning DataModule Guide](https://lightning.ai/docs/pytorch/stable/data/datamodule.html)
