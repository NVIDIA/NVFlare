# Federated ESM2 Embedding Inference

This example demonstrates how to use NVIDIA FLARE with BioNeMo to extract protein embeddings using the ESM2 model in a federated setting. The embeddings are generated independently on each client without sharing raw sequence data.

## Overview

This job performs federated inference to extract protein embeddings using BioNeMo's ESM2 model. Each client:
1. Loads their local protein sequence data
2. Runs ESM2 inference to generate embeddings
3. Saves embeddings locally for downstream tasks

No model training occurs in this job - it's purely inference-based.

## Code Structure

```bash
job_inference/
├── client.py    # Client script for ESM2 inference
├── job.py       # Job configuration using FedAvgRecipe
└── README.md    # This file
```

## Prerequisites

Install BioNeMo Framework and NVFlare:

```bash
# Follow BioNeMo installation instructions
# Install NVFlare
pip install nvflare
```

## Data Preparation

Before running this job, you need to prepare and split the protein sequence data. This is typically done using the data splitting script in the parent directory:

```bash
cd ..
# Run data preparation cells from task_fitting.ipynb or use split_data.py
```

This will create data files for each client in `/tmp/data/mixed_soft/`.

## Configuration

Key parameters in `job.py`:
- `n_clients`: Number of federated clients (default: 3)
- `checkpoint_path`: Path to ESM2 model checkpoint (auto-downloaded)
- `data_root`: Root directory containing client data files
- `results_path`: Directory to save inference results
- `micro_batch_size`: Batch size for inference (default: 64)
- `precision`: Model precision (default: bf16-mixed)

## Client Code

The client code ([client.py](./client.py)) implements the inference workflow:

1. Initialize NVFlare client API with `flare.init()`
2. Get site-specific information using `flare.get_site_name()`
3. Resolve per-client data paths automatically
4. Run ESM2 inference using BioNeMo's `infer_esm2` command
5. Save embeddings to results directory

```python
import nvflare.client as flare

flare.init()
site_name = flare.get_site_name()

# Construct client-specific paths
data_path = os.path.join(args.data_root, f"data_{site_name}.csv")

# Run inference
command = ["infer_esm2", "--checkpoint-path", checkpoint_path, ...]
subprocess.run(command)
```

## Job Recipe

The job uses `FedAvgRecipe` with a single round (since this is inference only):

```python
recipe = FedAvgRecipe(
    name="esm2_embeddings",
    min_clients=n_clients,
    num_rounds=1,  # Inference only needs 1 round
    train_script="client.py",
    train_args=script_args,
    launch_external_process=True,
    command="python3",
)
```

## Run Job

From the `job_inference` directory:

```bash
python job.py
```

The job will:
1. Download the ESM2 model if not already available
2. Run inference on each client sequentially (threads=1 for GPU memory management)
3. Save embeddings to `/tmp/data/mixed_soft/results/inference_results_{site_name}/`

## Output

Each client generates:
- `predictions__rank_0.pt`: Torch file containing embeddings and other inference outputs

The embeddings can then be used for downstream tasks like training an MLP classifier (see `../job_fedavg/`).

## Notes

- This job runs clients **sequentially** (threads=1) due to GPU memory constraints
- Each client processes their local data independently
- No data or model updates are shared between clients during inference
- The same ESM2 model is used by all clients (downloaded from NGC)
