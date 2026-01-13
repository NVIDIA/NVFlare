# Multi-GPU Federated Learning with TensorFlow

This example demonstrates federated learning with TensorFlow using MirroredStrategy for multi-GPU training.

## Overview

This example showcases how to use TensorFlow's `MirroredStrategy` with NVFlare for federated learning across multiple GPUs. Each client can utilize multiple GPUs for local training, with the trained model sent back to the server for aggregation.

## Key Features

- **Multi-GPU Training**: Uses TensorFlow's MirroredStrategy for data parallelism
- **Automatic GPU Detection**: Automatically discovers and uses all available GPUs
- **CIFAR-10 Dataset**: Trains a CNN on CIFAR-10 for image classification
- **Simple Integration**: Minimal code changes to support multi-GPU

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
tf/
├── job.py              # Job configuration
├── model.py            # TFNet CNN model definition
├── client.py           # TensorFlow client with MirroredStrategy
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
- 2+ GPUs (will use all available GPUs automatically)
- CUDA-capable GPUs

### Software Requirements
- TensorFlow with GPU support
- nvflare

We recommend using [NVIDIA TensorFlow Docker](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow) for GPU support:

```bash
# choose a recent tag from NGC, e.g., 24.09-tf2-py3
docker run --gpus=all -it --rm -v $(pwd):/workspace nvcr.io/nvidia/tensorflow:<xx.xx>-tf2-py3
cd /workspace
pip install nvflare
```

## GPU Memory Management

By default, TensorFlow attempts to allocate all available GPU memory. When running multiple clients simultaneously, enable memory growth:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true python job.py

# Or with async allocator
TF_FORCE_GPU_ALLOW_GROWTH=true TF_GPU_ALLOCATOR=cuda_malloc_async python job.py
```

## Monitoring GPU Usage

Check GPU utilization during training:

```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi
```

## Troubleshooting

### Out of Memory Errors
- Enable GPU memory growth (see above)
- Reduce batch size in training
- Reduce number of clients running simultaneously

### CUDA Errors
Ensure CUDA and cuDNN are properly installed:
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### No GPUs Detected
```bash
# Check GPU availability
nvidia-smi
python -c "import tensorflow as tf; print('GPUs:', len(tf.config.list_physical_devices('GPU')))"
```

## Additional Resources

- [TensorFlow Distributed Training Guide](https://www.tensorflow.org/guide/distributed_training)
- [MirroredStrategy Documentation](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy)
- [NVFlare Documentation](https://nvflare.readthedocs.io/)
