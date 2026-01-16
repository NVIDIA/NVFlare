# Plotting Scripts

This directory contains visualization scripts for the CIFAR-10 federated learning experiments.

## plot_label_distributions.py

Visualizes label distributions across clients for different alpha values using stacked bar charts to demonstrate how the Dirichlet distribution parameter controls data heterogeneity.

### Prerequisites

Before running the visualization script, you need to download the CIFAR-10 dataset:

```bash
cd ..  # Go to cifar10-sim directory
./prepare_data.sh
```

This downloads the CIFAR-10 dataset to `/tmp/cifar10` (approximately 170MB, one-time download).

## plot_tensorboard_events.py

Plots training curves from TensorBoard event files. See the main README for usage examples.
