# Multi-GPU Federated Learning with PyTorch DDP

This example demonstrates federated learning with PyTorch using Distributed Data Parallel (DDP) for multi-GPU training.

## Overview

This example showcases how to use PyTorch's DistributedDataParallel (DDP) with NVFlare for federated learning across multiple GPUs. Each client can utilize multiple GPUs for local training, with the trained model sent back to the server for aggregation.

## Key Features

- **Multi-GPU Training**: Uses PyTorch DDP with NCCL backend for efficient multi-GPU training
- **Per-Site Configuration**: Different sites can have different configurations (e.g., different master ports)
- **External Process Launch**: Uses `torch.distributed.run` to launch training processes
- **CIFAR-10 Dataset**: Trains a simple CNN on CIFAR-10 for image classification

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
pt/
├── job.py              # Job configuration with per-site configs
├── model.py            # Net CNN model definition
├── client.py           # PyTorch DDP client script
├── prepare_data.sh     # Script to download CIFAR-10
└── requirements.txt    # Python dependencies
```

## Command Line Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--num_rounds` | Number of training rounds | 5 |
| `--use_tracking` | Enable TensorBoard tracking | False |
| `--export_config` | Export job config only | False |

## Examples

```bash
# Run with default settings (5 rounds)
python job.py

# Run with 10 rounds
python job.py --num_rounds 10

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
- 2+ GPUs per client
- CUDA-capable GPUs with NCCL support

### Software Requirements
- PyTorch with CUDA support
- torchvision
- nvflare

## Multi-GPU Configuration

### Number of Processes
Set `--nproc_per_node` to the number of GPUs you want to use:
```bash
# Use 4 GPUs
python3 -m torch.distributed.run --nproc_per_node=4 client.py
```

### Multiple Clients on Same Machine
When running multiple clients on the same machine, use different master ports:
```python
per_site_config={
    "site-1": {
        "command": "... --master_port=7777",
    },
    "site-2": {
        "command": "... --master_port=8888",
    },
}
```

## Troubleshooting

### Port Conflicts
If you see "Address already in use" errors, ensure each site uses a unique `--master_port`.

### GPU Memory Issues
- Reduce `batch_size` in `client.py`
- Reduce number of GPUs with `--nproc_per_node`

### NCCL Errors
Ensure NCCL is properly installed and all GPUs are visible:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.device_count())"
```

## Additional Resources

- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [torch.distributed.run Documentation](https://pytorch.org/docs/stable/distributed.html#launch-utility)
- [NVFlare Documentation](https://nvflare.readthedocs.io/)

