# ML to FL with PyTorch

This example demonstrates federated learning with PyTorch, supporting 4 training modes:

| Mode | Description | Requirements |
|------|-------------|--------------|
| `pt` | Standard PyTorch | 1 GPU |
| `pt_ddp` | PyTorch DDP (multi-GPU) | 2+ GPUs |
| `lightning` | PyTorch Lightning | 1 GPU |
| `lightning_ddp` | Lightning + DDP | 2+ GPUs |

## Quick Start

1. **Prepare data:**
```bash
bash ./prepare_data.sh
```

2. **Run federated learning:**
```bash
# Standard PyTorch
python job.py --mode pt

# PyTorch with DDP (multi-GPU)
python job.py --mode pt_ddp

# PyTorch Lightning
python job.py --mode lightning

# Lightning with DDP (multi-GPU)
python job.py --mode lightning_ddp
```

## Project Structure

```
pt/
├── job.py                  # Unified job config (all 4 modes)
├── model.py                # Net class (for pt, pt_ddp)
├── lit_model.py            # LitNet class (for lightning modes)
├── client.py               # Standard PyTorch client
├── client_ddp.py           # PyTorch DDP client
├── client_lightning.py     # Lightning client (single GPU)
├── client_lightning_ddp.py # Lightning DDP client (multi-GPU)
├── prepare_data.sh
└── requirements.txt
```

## Command Line Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Training mode: `pt`, `pt_ddp`, `lightning`, `lightning_ddp` | `pt` |
| `--n_clients` | Number of clients | 2 |
| `--num_rounds` | Number of training rounds | 5 |
| `--use_tracking` | Enable TensorBoard tracking | False |
| `--export_config` | Export job config only | False |

## Examples

```bash
# Basic PyTorch with 3 clients, 10 rounds
python job.py --mode pt --n_clients 3 --num_rounds 10

# PyTorch with TensorBoard tracking
python job.py --mode pt --use_tracking

# Lightning DDP with 2 clients
python job.py --mode lightning_ddp --n_clients 2

# Export config for deployment
python job.py --mode pt_ddp --export_config
```

## Client Scripts

### `client.py` - Standard PyTorch

Uses `nvflare.client` API:
```python
import nvflare.client as flare

flare.init()
while flare.is_running():
    input_model = flare.receive()
    net.load_state_dict(input_model.params)
    # ... train ...
    flare.send(flare.FLModel(params=net.state_dict(), ...))
```

### `client_ddp.py` - PyTorch DDP

Adds distributed training support:
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
flare.init(rank=f"{rank}")  # Pass rank to flare
ddp_model = DDP(net, device_ids=[device])
# ... only rank 0 sends/receives ...
```

### `client_lightning.py` - PyTorch Lightning

Uses `nvflare.client.lightning` API:
```python
import nvflare.client.lightning as flare

flare.patch(trainer)  # Patch the trainer
while flare.is_running():
    input_model = flare.receive()
    trainer.fit(model, datamodule)  # Trainer handles FL internally
```

### `client_lightning_ddp.py` - Lightning DDP

For DDP, we need to broadcast `is_running` to all ranks:
```python
while True:
    is_running = flare.is_running()
    is_running = trainer.strategy.broadcast(is_running, src=0)  # Sync all ranks
    if not is_running:
        break
    ...
```

## Requirements

```bash
pip install -r requirements.txt
```

For DDP examples, ensure you have multiple GPUs available.
