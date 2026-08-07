# Federated LLM Supervised Fine-Tuning with Collab

This example performs full-parameter supervised fine-tuning (SFT) of a Hugging
Face causal language model with TRL and synchronous federated averaging. It
uses Qwen2.5-0.5B-Instruct by default and exchanges complete model states as
native PyTorch tensor dictionaries through the Collab API.

> **Resource note:** This example exchanges and averages the complete model,
> not a parameter-efficient subset. Resource requirements scale with both the
> checkpoint size and client count, including RAM, GPU memory, transfer time,
> and checkpoint space.

## NVIDIA FLARE Installation

Follow the repository [installation instructions](../../../../README.md), then
start from the `examples/advanced` directory and install this example's
additional dependencies:

```bash
python -m pip install -r collab/pt_llm_sft/requirements.txt
```

## Code Structure

| File | Purpose |
|---|---|
| `pt_llm_sft.py` | Defines the stateful SFT client, synchronous server workflow, Collab recipe, and simulator entry point. |
| `prepare_data.py` | Creates deterministic synthetic or Dolly JSONL shards for each client. |
| `requirements.txt` | Lists the additional Hugging Face and PyTorch dependencies. |
| `figures/training_loss.png` | Shows the observed training-curve alignment. |

## Data

### Synthetic data (quick start)

From `examples/advanced`, create small instruction-tuning shards for four
logical clients:

```bash
python collab/pt_llm_sft/prepare_data.py \
    --data-mode synthetic \
    --num-clients 4 \
    --data-root /tmp/nvflare/collab/pt_llm_sft/data
```

This creates six training records and one validation record under each site
directory. With the default batch size of one, the six batches support five
synchronization intervals per epoch.

### Dolly

Download `databricks/databricks-dolly-15k` and split it across four clients:

```bash
python collab/pt_llm_sft/prepare_data.py \
    --data-mode dolly \
    --num-clients 4 \
    --data-root /tmp/nvflare/collab/pt_llm_sft/dolly
```

The rows are shuffled deterministically, divided into disjoint client shards,
and split into 90% training and 10% validation data within each client. Use
`--seed` to change the partition, `--validation-fraction` to change the
split, and `--cache-dir` to select the Hugging Face dataset cache.

For any prepared dataset, `--syncs-per-epoch` must not exceed the number of
local training batches.

## Model

The default Qwen2.5-0.5B-Instruct checkpoint keeps the quick start relatively
small. Select another compatible causal LM with `--model-name-or-path`. The
example trains all model parameters and uses `--precision auto` to select
BF16 on supported CUDA devices and FP32 otherwise.

Remote checkpoint code is disabled by default. If a checkpoint requires custom
code, review its repository, pin a trusted revision with `--model-revision`,
and only then pass `--trust-remote-code`. That option allows
repository-supplied Python to execute in every client process.

## Client Code

`LLMSFTClient.initialize()` is marked with `@collab.init`, so each client
loads its own prepared shard, tokenizer, model, trainer, dataloader, and
optimizer once. `SFTTrainer` prepares the SFT batches and places the model on
the process-local device selected by the simulator.

The published training method receives and returns ordinary Python objects,
including the complete `dict[str, torch.Tensor]` state:

```python
@collab.publish
def train(self, sync_number, global_weights):
    self.model.load_state_dict(global_weights, strict=True)
    # Evaluate and train the next fraction of the local epoch.
    return {
        "weights": cpu_model_state(self.model),
        "num_examples": num_examples,
        "train_loss": train_loss,
        "eval_loss": eval_loss,
    }
```

The model, optimizer, prepared dataloader, and current data position remain in
memory between calls. No application-level conversion to NumPy, `FLModel`,
`Shareable`, or checkpoint file is needed between synchronization intervals.

For multi-process launch, experiment tracking, and deployment-oriented
configuration, see the [full Hugging Face LLM example](../../llm_hf/README.md).
Those concerns are intentionally outside this Collab example.

## Server-Side Workflow

`SFTFedAvg.run()` is the single `@collab.main` entry point. It obtains the
initial state from one initialized client, fans each global state out to all
clients, and applies sample-weighted FedAvg to the returned tensors:

```python
call_results = collab.clients(timeout=self.call_timeout).train(sync_number, global_weights)
global_weights, average_loss = average_model_states(dict(call_results), self.min_clients)
```

Floating-point tensors are accumulated in FP32 on the server and converted
back to the model's original dtype. Non-floating state is copied from the first
update. The example requires every configured client to return successfully;
otherwise aggregation stops with a quorum error.

By default, one local epoch is divided into five synchronization intervals.
The server repeats client training and aggregation for every interval and
writes the final state after the last one.

## Job Recipe Code

`make_recipe()` wires the client and server objects directly into a
`CollabRecipe`. `make_env()` creates a `SimEnv` with the requested client
count, workspace, and GPU placement:

```python
recipe = CollabRecipe(
    job_name="pt_llm_sft",
    server=server,
    client=trainer,
    min_clients=args.num_clients,
)
recipe.execute(make_env(args))
```

The standard Recipe export arguments are also supported. Export builds the job
without loading the model or requiring prepared data on the machine performing
the export:

```bash
python -m collab.pt_llm_sft.pt_llm_sft \
    --export \
    --export-dir /tmp/nvflare/jobs/job_config
```

## Run Job

From `examples/advanced`, run the prepared synthetic quick start:

```bash
python -m collab.pt_llm_sft.pt_llm_sft \
    --data-root /tmp/nvflare/collab/pt_llm_sft/data \
    --num-clients 4 \
    --num-epochs 1 \
    --syncs-per-epoch 5
```

The Hugging Face model is downloaded on the first run if it is not already in
the local cache. CUDA GPUs are recommended; CPU execution is supported but is
considerably slower and stores the default model state in FP32.

Pass `--gpu-config 0,1,2,3` to place four clients on four GPUs, one client per
GPU. A bracketed group assigns multiple GPUs to one client, for example
`--gpu-config "[0,1],[2,3]"`. If `--gpu-config` is omitted, the simulator
does not apply per-client GPU placement.

Clients evaluate each received global model by default. Pass
`--skip-evaluation` when that evaluation is not needed.

## Output

The final model state is written to
`/tmp/nvflare/collab/pt_llm_sft/results/server/model_final.pt` by default.
Pass `--save-every-sync` to retain intermediate states. Each checkpoint is a
complete PyTorch model state dictionary; load the same base architecture and
apply it with `model.load_state_dict(...)`.

## Training-curve alignment

The standard and Collab simulator paths produced similar training-loss
trajectories for the compared workload:

![Qwen3-8B aggregated training loss for the Standard and Collab simulators](figures/training_loss.png)
