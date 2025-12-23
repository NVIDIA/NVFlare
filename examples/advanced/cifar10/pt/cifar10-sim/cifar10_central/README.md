# Centralized Training Baseline

This directory contains a centralized training baseline for CIFAR-10 classification using PyTorch.

## Overview

The centralized training script (`train.py`) trains a CNN model on the entire CIFAR-10 dataset without federated learning. This serves as a performance baseline to compare against federated learning approaches.

**Key Characteristics:**
- Single training process using all available data
- No federated learning orchestration
- Uses the same model architecture (`ModerateCNN`) as federated experiments
- Supports cosine annealing learning rate scheduler (enabled by default)

## Usage

### Basic Usage

Train for 25 epochs (recommended for comparison with FL experiments):

```bash
python train.py --epochs 25
```

This configuration roughly matches the total number of training iterations in the federated experiments (50 FL rounds × 4 local epochs ÷ 8 clients ≈ 25 epochs).

### Command-line Arguments

#### Training Parameters
- `--epochs` - Number of training epochs (default: `4`)
- `--lr` - Learning rate (default: `0.01`)
- `--batch_size` - Training batch size (default: `64`)
- `--num_workers` - Data loading workers (default: `2`)

#### Learning Rate Scheduler
- `--no_lr_scheduler` - Disable cosine annealing LR scheduler (enabled by default)
- `--cosine_lr_eta_min_factor` - Minimum LR as factor of initial LR (default: `0.01`)

#### Other Options
- `--train_idx_root` - Root directory for training data (default: `/tmp/cifar10_splits`)
- `--output_dir` - Directory to save model and logs (default: `/tmp/nvflare/simulation/cifar10_central`)

## Examples

### Standard Training (25 epochs)

```bash
python train.py --epochs 25
```

### Custom Learning Rate

```bash
python train.py --epochs 25 --lr 0.05
```

### Disable Learning Rate Scheduler

```bash
python train.py --epochs 25 --no_lr_scheduler
```

### Custom Output Directory

```bash
python train.py --epochs 25 --output_dir ./my_results
```

## Implementation Details

### Model Architecture

Uses the `ModerateCNN` architecture defined in `../src/model.py`:
- 2 convolutional layers with max pooling
- 3 fully connected layers
- ~122K parameters

### Training Configuration

- **Optimizer**: SGD with momentum (0.9)
- **Loss Function**: Cross-entropy loss
- **Learning Rate Scheduler**: Cosine annealing (optional, enabled by default)
  - Smoothly decays learning rate from initial value to 1% of initial value
  - Helps achieve better convergence

### Data

- **Dataset**: CIFAR-10 (60,000 32×32 color images in 10 classes)
- **Training Split**: 50,000 images
- **Test Split**: 10,000 images (used for validation)
- **Preprocessing**: Standard normalization for CIFAR-10

### Performance

With the default configuration (25 epochs, cosine annealing LR):
- Training time: ~5 minutes on NVIDIA A6000 GPU
- Expected validation accuracy: ~87.5%

## Viewing Results

After training, view the TensorBoard logs:

```bash
tensorboard --logdir=/tmp/nvflare/simulation/cifar10_central
```

Then open `http://localhost:6006` in your browser to see:
- Training loss curves
- Validation accuracy
- Learning rate schedule

## Comparison with Federated Learning

This centralized baseline helps evaluate the performance trade-offs of federated learning:

| Approach | Accuracy | Training Time | Data Distribution |
|----------|----------|---------------|-------------------|
| Centralized | ~87.5% | ~5 min | All data in one location |
| FedAvg (α=1.0) | ~89.4% | ~10 min | Uniform split across 8 clients |
| FedAvg (α=0.1) | ~80.7% | ~10 min | Highly non-IID split |

See the [main README](../README.md) for detailed federated learning experiments.

## References

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [NVFlare Documentation](https://nvflare.readthedocs.io/)

