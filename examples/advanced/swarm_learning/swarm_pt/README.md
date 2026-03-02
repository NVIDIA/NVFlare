
# Swarm Learning with LoRA Fine-Tuning

This example demonstrates how to use NVIDIA FLARE's [Swarm Learning](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.ccwf.html) workflow to fine-tune a large language model using [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) in a federated setting. Each client fine-tunes a Qwen2.5 causal language model (0.5B or 1.5B, configurable) on its own local data shard using LoRA adapters — only the small adapter weights are exchanged each round, keeping communication cost low regardless of base-model size.

The complete example code can be found in the `swarm_pt` directory. It is recommended to create a virtual environment and run everything within a virtualenv.

## NVIDIA FLARE Installation

For the complete installation instructions, see [Installation](https://nvflare.readthedocs.io/en/main/installation.html)

```bash
pip install nvflare
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Code Structure

First get the example code from GitHub:

```bash
git clone https://github.com/NVIDIA/NVFlare.git
```

Then navigate to the example directory:

```bash
git switch <release branch>
cd examples/advanced/swarm_learning/swarm_pt
```

```
swarm_pt/
|
|-- client.py           # client LoRA fine-tuning script (runs as subprocess)
|-- model.py            # QwenLoRAModelWrapper — LoRA-adapted model for server persistor
|-- job.py              # job recipe using SimpleSwarmLearningRecipe
|-- download_data.py    # pre-download wikitext-2 dataset and Qwen2.5 model
|-- prepare_data.py     # split dataset among N clients
|-- requirements.txt    # dependencies
```

## Hardware Requirements

A CUDA-capable GPU is strongly recommended. The base model is loaded in `bfloat16` and placed automatically via `device_map="auto"`, so it runs on GPU when available and falls back to CPU otherwise. Running 4 clients simultaneously on CPU is very slow (~10 minutes per step per client).

## Dataset

This example uses the [wikitext-2-raw-v1](https://huggingface.co/datasets/wikitext) dataset (Apache-2.0, no license acceptance required). The training split is partitioned into disjoint shards — one per client — to simulate real-world data silos where each participant trains only on their own local data.

## Model

[Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) (Apache-2.0, default) or [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) is used as the base model, selectable via `--model_size`. LoRA adapters are attached to the query and value projection layers of every attention block:

```python
LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
           lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
```

The `QwenLoRAModelWrapper` in `model.py` overrides `state_dict()` / `load_state_dict()` to expose only the LoRA adapter parameters (~2 MB for 0.5B, ~4 MB for 1.5B) to the server persistor and aggregator, instead of the full base model weights (~1 GB / ~3 GB). The base model weights are loaded in `bfloat16` to halve memory usage.

`load_state_dict()` also detects shape mismatches between a saved checkpoint and the current model size. If you switch `--model_size` between runs without clearing the workspace, the incompatible checkpoint is silently discarded and training restarts from a fresh adapter — no manual cleanup required.

## Client Code

`client.py` implements the local LoRA fine-tuning loop. The NVFlare Client API integration follows the same three-step pattern as any NVFlare client script:

```python
import nvflare.client as flare

flare.init()                          # 1. Initialize NVFlare Client API
input_model = flare.receive()         # 2. Receive global LoRA adapter from server
apply_global_adapter(model, input_model.params)  # 3. Apply global adapter weights

adapter_diff = local_train(model, dataloader, steps)  # local fine-tuning

flare.send(flare.FLModel(             # 4. Send LoRA adapter diff back
    params_type=ParamsType.DIFF,
    params=adapter_diff,
))
```

Each client fine-tunes for a fixed number of gradient steps per round, then returns the adapter weight **diff** (after − before). The server aggregates these diffs across clients using weighted averaging.

## Server-Side Workflow

This example uses [Swarm Learning](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.ccwf.html), a decentralized federated workflow where clients act as both trainers and aggregators in a peer-to-peer fashion — there is no central aggregation server. The workflow proceeds as follows:

1. A designated starting client scatters the global LoRA adapter to all peers.
2. Each client trains locally and submits its adapter diff to a designated aggregator.
3. The aggregator accumulates diffs and produces a new global adapter.
4. The updated adapter is scattered to the next round of trainers.
5. This repeats for `num_rounds` rounds.

## Job Recipe Code

`job.py` uses `SimpleSwarmLearningRecipe` to configure the entire job with a few lines:

```python
model_path = MODEL_SIZES[args.model_size]   # e.g. "Qwen/Qwen2.5-0.5B" or "Qwen/Qwen2.5-1.5B"

recipe = SimpleSwarmLearningRecipe(
    name="ccwf_swarm_pt_lora",
    model=QwenLoRAModelWrapper(model_path=model_path),
    num_rounds=args.num_rounds,
    train_script="client.py",
    min_clients=2,
    launch_external_process=True,       # run client.py as a subprocess
    cuda_empty_cache=True,              # free GPU cache between rounds
    train_args={"script_args": script_args},
    expected_data_kind=DataKind.WEIGHT_DIFF,
    params_transfer_type=TransferType.DIFF,
    start_task_timeout=1200,
    progress_timeout=14400,
    max_status_report_interval=600,
)

env = SimEnv(num_clients=args.n_clients, workspace_root=args.workspace)
recipe.execute(env)
```

`SimpleSwarmLearningRecipe` handles all component wiring automatically:
- `PTClientAPILauncherExecutor` + `SubprocessLauncher` + `CellPipe` for subprocess execution
- `PTFileModelPersistor` for checkpoint management (stores only LoRA adapter weights)
- `InTimeAccumulateWeightedAggregator` for peer-to-peer adapter aggregation
- `SimpleModelShareableGenerator` for model ↔ shareable conversion

## Step-by-Step Instructions

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Download model and dataset

```bash
python download_data.py
```

This downloads the `wikitext-2-raw-v1` dataset and the default `Qwen/Qwen2.5-0.5B` model into the HuggingFace cache.

To download the 1.5B model instead:

```bash
python download_data.py --model_size 1.5B
```

To skip the model download (e.g. model is already cached or on a local mount):

```bash
python download_data.py --skip_model
```

To point at a local model path:

```bash
python download_data.py --model_path /model/qwen/qwen2.5-0.5B --skip_model
```

### Step 3 — Prepare per-client data splits

```bash
python prepare_data.py --n_clients 4
```

This partitions the training split into 4 disjoint shards and writes them to `/tmp/swarm_data/`:

```
/tmp/swarm_data/
    site-1/train/      # Arrow dataset — shard 0
    site-2/train/      # Arrow dataset — shard 1
    site-3/train/      # Arrow dataset — shard 2
    site-4/train/      # Arrow dataset — shard 3
    validation/        # shared validation split
```

Skip this step to use in-memory sharding instead (no disk writes, fewer data options).

### Step 4 — Run the simulation

With pre-split data (recommended):

```bash
python job.py --n_clients 4 --num_rounds 5 --data_dir /tmp/swarm_data
```

To use the larger 1.5B model:

```bash
python job.py --n_clients 4 --num_rounds 5 --model_size 1.5B --data_dir /tmp/swarm_data
```

With in-memory sharding (quick start, no prepare_data.py needed):

```bash
python job.py --n_clients 4 --num_rounds 5
```

On CPU, reduce batch size and sequence length to keep step time manageable:

```bash
python job.py --n_clients 4 --num_rounds 5 --batch_size 1 --max_seq_len 32 --data_dir /tmp/swarm_data
```

The simulation results are written to `/tmp/nvflare/simulation/ccwf_swarm_pt_lora/`.

### Step 5 — (Optional) Export job for production deployment

```bash
python job.py --export_dir /tmp/swarm_lora_job
```

This writes a standard NVFlare job folder that can be submitted to a production provisioned system.

## Configuration Reference

| Argument | Default | Description |
|---|---|---|
| `--n_clients` | 2 | Number of simulated clients |
| `--num_rounds` | 3 | Number of swarm learning rounds |
| `--model_size` | `0.5B` | Base model size: `0.5B` or `1.5B` |
| `--local_steps` | 10 | Gradient steps per client per round |
| `--batch_size` | 4 | Training batch size per client |
| `--max_seq_len` | 128 | Maximum tokenized sequence length |
| `--data_dir` | *(empty)* | Pre-split data root from `prepare_data.py`; in-memory if omitted |
| `--workspace` | `/tmp/nvflare/simulation` | Root directory for simulation output |
| `--export_dir` | *(empty)* | If set, export job folder instead of running |

## Output Summary

#### Initialization
- Global LoRA adapter initialized from `QwenLoRAModelWrapper`.
- Swarm workflow configured; starting client selected.

#### Each Round
- **Scatter**: Starting client sends global LoRA adapter to all trainers.
- **Local training**: Each client fine-tunes for `local_steps` gradient steps on its data shard, reporting loss every 5 steps.
- **Submit**: Each client sends its LoRA adapter diff to the designated aggregator.
- **Aggregate**: Aggregator accumulates diffs, produces updated global adapter.

#### Completion
- Final global LoRA adapter persisted to the simulation workspace.
- Results available at `<workspace>/ccwf_swarm_pt_lora/`.
