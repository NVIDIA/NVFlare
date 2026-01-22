# Hello PyTorch Lightning - Model Evaluation

This example demonstrates how to use NVIDIA FLARE to evaluate a pre-trained PyTorch Lightning model across multiple clients in a federated setting using the `EvalRecipe`.

## Setup

### Install NVIDIA FLARE

```bash
pip install nvflare
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Code Structure

```
hello-lightning-eval/
├── client.py            # Client evaluation script
├── model.py             # Model definition (LitNet with CIFAR-10)
├── generate_pretrain.py # Script to generate pre-trained model
├── job.py               # Job recipe using EvalRecipe
├── requirements.txt     # Dependencies
├── prepare_data.sh      # Download CIFAR-10 dataset
└── README.md            # This file
```

## Workflow

### Step 1: Download Data

The CIFAR-10 dataset needs to be downloaded before running any training or evaluation:

```bash
bash prepare_data.sh
```

### Step 2: Generate Pre-trained Model

Train a model on CIFAR-10 that will be evaluated across clients:

```bash
python generate_pretrain.py --epochs 1 --batch_size 32 --output pretrained_model.pt
```

This will:
- Train a LitNet model on CIFAR-10 for 1 epoch (quick demonstration)
- Save the trained weights to `pretrained_model.pt`
- Print validation accuracy (~23% after 1 epoch)

**Arguments:**
- `--epochs`: Number of training epochs (default: 5). Use 1 for quick demo, 5-10 for better accuracy (~50%)
- `--batch_size`: Batch size for training (default: 32)
- `--output`: Output checkpoint file path (default: `pretrained_model.pt`)

**Note:** This example uses 1 epoch for quick demonstration. For better accuracy, increase the number of epochs.

### Step 3: Evaluate Model Across Clients

Evaluate the pre-trained model using federated evaluation:

```bash
python job.py --n_clients 2 --batch_size 24 --checkpoint pretrained_model.pt
```

This will:
- Load the pre-trained model from the checkpoint
- Send it to all clients for evaluation
- Collect and display evaluation metrics from each client

**Arguments:**
- `--n_clients`: Number of simulated clients (default: 2)
- `--batch_size`: Batch size for evaluation (default: 24)
- `--checkpoint`: Path to pre-trained model checkpoint (default: `pretrained_model.pt`)

## How It Works

### EvalRecipe

The `EvalRecipe` is a simple recipe for evaluating a pre-trained model across multiple sites:

```python
from nvflare.app_opt.pt.recipes.eval import EvalRecipe
from model import LitNet

recipe = EvalRecipe(
    min_clients=n_clients,
    initial_model=LitNet(),
    eval_script="client.py",
    eval_args=f"--batch_size {batch_size}",
    source_checkpoint="pretrained_model.pt",  # Loads pre-trained weights
)
```

### Client Code

The client evaluation script (`client.py`) is simple - it just validates the global model:

```python
import nvflare.client.lightning as flare

# Patch the trainer
flare.patch(trainer)

# Evaluate the current global model
print("--- validate global model ---")
trainer.validate(model, datamodule=cifar10_dm)
```

No training occurs - just evaluation on local data.

### What Happens

1. **Server** loads the pre-trained model from `pretrained_model.pt`
2. **Server** sends the model to all clients using `GlobalModelEval` controller
3. **Each client** evaluates the model on their local CIFAR-10 test set
4. **Each client** reports metrics (accuracy, loss) back to server
5. **Server** logs all evaluation results

## Example Output

After 1 epoch of pre-training (quick demo):

```
Using pre-trained checkpoint: /path/to/pretrained_model.pt
Sending model for evaluation
Got 2 results

Validation DataLoader 0: 100%|██████████| 417/417 [00:03<00:00]
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃      Validate metric      ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│       val_acc_epoch       │    0.23170000314712524    │
│         val_loss          │     2.201637029647827     │
└───────────────────────────┴───────────────────────────┘
```


## Notes

- This example uses **1 epoch** for pre-training by default for quick demonstration (~23% accuracy)
- For better accuracy (~50%), increase epochs: `python generate_pretrain.py --epochs 10`
- The same CIFAR-10 dataset is used on all clients for simplicity
- In a real federated setting, each client would have their own local test set
- The model architecture must match between training and evaluation
- Evaluation is read-only - no model updates occur
