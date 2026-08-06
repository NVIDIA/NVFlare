# Federated LLM Supervised Fine-Tuning with Collab

This example performs full-parameter supervised fine-tuning (SFT) of a Hugging
Face causal language model with TRL and synchronous federated averaging. It
uses Qwen2.5-0.5B-Instruct by default; select any compatible checkpoint with
`--model-name-or-path`.

Its focus is the Collab programming model. Client methods accept and return
ordinary Python objects, including the complete
`dict[str, torch.Tensor]` model state. The application contains no conversion
to NumPy, `FLModel`, `Shareable`, or another transport-oriented format:

```python
client_results = collab.clients.train(sync_number, global_weights)
global_weights, average_loss = average_model_states(dict(client_results), min_clients)
```

Collab handles transport between simulated processes. Each client keeps its
model, optimizer, prepared SFT dataloader, and current data position in memory.
It applies the averaged state directly with `model.load_state_dict(...)`, trains
the next portion of the epoch, and returns the full model state directly.

By default, every local epoch contains five synchronization intervals. This
makes the example useful for studying repeated native-tensor exchange without
introducing application-level model-format transitions or checkpoint/resume
files between intervals.

> **Resource note:** This example exchanges and averages the complete model,
> not a parameter-efficient subset. Resource requirements scale with both the
> checkpoint size and client count, including RAM, GPU memory, transfer time,
> and checkpoint space.

## Install the LLM dependencies

Starting from `examples/advanced`, install the additional packages:

```bash
python -m pip install -r collab/pt_llm_sft/requirements.txt
```

The parent Collab README assumes NVFlare is already installed from this
repository, so no additional editable install is needed here.

## 1. Prepare data

### Synthetic data (quick start)

Create small synthetic instruction-tuning shards for four logical clients:

```bash
python collab/pt_llm_sft/prepare_data.py \
    --data-mode synthetic \
    --num-clients 4 \
    --data-root /tmp/nvflare/collab/pt_llm_sft/data
```

This command creates six training records and one validation record under each
site directory. With the default batch size of one, the six batches support
five synchronization intervals per epoch.

### Dolly (real data)

Download `databricks/databricks-dolly-15k` and split it across four clients:

```bash
python collab/pt_llm_sft/prepare_data.py \
    --data-mode dolly \
    --num-clients 4 \
    --data-root /tmp/nvflare/collab/pt_llm_sft/dolly
```

The Dolly rows are shuffled deterministically, divided into four disjoint
client shards, and split into 90% training and 10% validation data within each
client. Use `--seed` to change the partition, `--validation-fraction` to change
the split, and `--cache-dir` to select the Hugging Face dataset cache. To train
on Dolly, use `/tmp/nvflare/collab/pt_llm_sft/dolly` as the `--data-root` in
the next command.

For any prepared dataset,
`--syncs-per-epoch` must not exceed the number of local training batches.

## 2. Run the Collab recipe

```bash
python -m collab.pt_llm_sft.pt_llm_sft \
    --data-root /tmp/nvflare/collab/pt_llm_sft/data \
    --num-clients 4 \
    --num-epochs 1 \
    --syncs-per-epoch 5
```

The Hugging Face model is downloaded on the first run if it is not already in
the local cache. CUDA GPUs are recommended; CPU execution is supported but is
considerably slower and stores the default model state in FP32. Use
`--model-name-or-path` to select another compatible causal-LM checkpoint.
Pass `--gpu-config 0,1,2,3` to place four clients on four GPUs, one client per
GPU. A bracketed group assigns multiple GPUs to one client, for example
`--gpu-config "[0,1],[2,3]"`. If `--gpu-config` is omitted, no per-client GPU
placement is applied by the simulator.
Remote checkpoint code is disabled by default. If a model requires custom
code, review its repository, pin a trusted revision with `--model-revision`,
and only then pass `--trust-remote-code`; this option allows repository-supplied
Python to execute in every client process.

Before each synchronization interval, clients evaluate the received global
model. Pass `--skip-evaluation` when measuring synchronization overhead.

The final model is written to
`/tmp/nvflare/collab/pt_llm_sft/results/server/model_final.pt` by default. Pass
`--save-every-sync` to retain intermediate states as well. Checkpoints are
complete PyTorch model state dictionaries. Load the same base architecture,
then apply a checkpoint with `model.load_state_dict(...)`.

## Workflow

1. Each client loads and prepares its causal LM and SFT data once.
2. The server gets the initial full state from one initialized client.
3. Clients train the next fraction of their epoch.
4. Clients return full model tensors, loss, and processed sample count.
5. The server computes sample-weighted FedAvg and immediately starts the next
   interval.
6. Steps 3–5 repeat five times per epoch by default.

This intentionally omits the production-oriented features in
[the full Hugging Face LLM example](../../llm_hf/README.md), such as
multi-GPU launch, experiment tracking, and deployment configuration. It keeps
the LLM code close to an ordinary SFT program so direct, tensor-native Collab
calls remain easy to see.

## Training-curve alignment

The standard and Collab simulator paths produce similar training-loss
trajectories, confirming that they execute comparable learning workloads:

![Qwen3-8B aggregated training loss for the Standard and Collab simulators](figures/training_loss.png)
