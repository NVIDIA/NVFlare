# SCAFFOLD with TensorFlow

This folder contains the SCAFFOLD algorithm implementation for CIFAR-10 using TensorFlow.

## Files

- `client.py`: Client-side training logic with SCAFFOLD control variates
- `job.py`: Job configuration and execution

## Usage

```bash
python cifar10_scaffold/job.py --n_clients 8 --num_rounds 50 --alpha 0.1
```

## Arguments

- `--n_clients`: Number of clients (default: 8)
- `--num_rounds`: Number of FL rounds (default: 50)
- `--batch_size`: Batch size for training (default: 64)
- `--epochs`: Number of local training epochs per round (default: 4)
- `--alpha`: Dirichlet sampling parameter for data heterogeneity (default: 0.1)
- `--name`: Optional job name (default: "")
