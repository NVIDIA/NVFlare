# Multi-GPU Federated Learning with PyTorch FSDP2

This example fine-tunes all parameters of [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)
on [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext) with NVFlare FedAvg.
Two clients each use two GPUs. PyTorch Fully Sharded Data Parallel (FSDP2) shards parameters,
gradients, and AdamW state within each client; the server exchanges and averages full model states.
Each GPU runs one process, called a rank. The training loop uses native PyTorch and the NVFlare Client API.

## Quick Start

### Requirements

- Linux, four BF16-capable CUDA GPUs, and NCCL support.
- Python 3.10 or later and CUDA-enabled PyTorch 2.6 or later.
- Newer PyTorch versions using Triton need a C compiler and matching Python development headers.
- Substantial host RAM for the server and full-state exchange; plan for at least 128 GB for the simulation.

The default job was validated on four NVIDIA A10G 24 GB GPUs with PyTorch 2.6 and CUDA 12.6.
That host required `NCCL_P2P_DISABLE=1 python job.py` because its NCCL peer-to-peer transport stalled.
Memory use depends on sequence length and batch size.

From this directory:

```bash
pip install -r requirements.txt
python prepare_data.py
python job.py
```

Preparation caches the model and tokenizer, splits WikiText-2 training rows between site-1 and site-2,
and packs text into 256-token blocks with EOS between nonempty rows. It saves each site's training
blocks and shared validation blocks under `/tmp/nvflare/fsdp2/data`.
The [dataset card](https://huggingface.co/datasets/Salesforce/wikitext) describes the corpus and its licenses.

The default job assigns GPUs 0–1 to site-1 and GPUs 2–3 to site-2 for three federated rounds.
Each rank evaluates up to five held-out batches, then takes ten AdamW steps with batch size one.
These are real language-model updates on public text; the short run demonstrates federation,
not convergence. The job prints its result location, where the server saves `FL_global_model.pt`.

## Project Structure

```text
fsdp2/
├── prepare_data.py     # Cache the model and prepare tokenized site datasets
├── job.py              # FedAvg recipe and simulation GPU groups
├── model.py            # Shared pretrained Qwen model definition
├── client.py           # FSDP2 evaluation, training, and full-state exchange
└── requirements.txt    # Python dependencies
```

## Command Line Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_name` | Qwen checkpoint or local model directory; preparation and job | `Qwen/Qwen2.5-1.5B` |
| `--data_dir` | Prepared dataset root; preparation and job | `/tmp/nvflare/fsdp2/data` |
| `--seq_length` | Tokens per block; preparation only | 256 |
| `--num_rounds` | Federated rounds | 3 |
| `--local_steps` | AdamW steps per rank and round | 10 |
| `--eval_steps` | Maximum validation batches per rank and round | 5 |
| `--batch_size` | Token blocks per rank and batch | 1 |
| `--gpu_config` | Two separate two-GPU groups for simulation | `[0,1],[2,3]` |

Use matching `--model_name` and `--data_dir` values for preparation and the job.
Regenerate the data when changing the tokenizer or sequence length.

```bash
python job.py --num_rounds 5 --local_steps 20
python job.py --gpu_config "[2,3],[4,5]"

# Export for deployment instead of running the simulation
python job.py --export --export-dir /tmp/nvflare/jobs
```

Exporting does not load or download the model. Model files and prepared datasets are not bundled
in the exported job: make the model available on the server and clients, and ensure `--data_dir`
exists at each client with its `<site-name>/train` and shared `validation` datasets.
Each deployed client needs two visible GPUs; `--gpu_config` applies only to simulation.

## How It Works

`job.py` uses the standard PyTorch `FedAvgRecipe` and tensor exchange. The server loads pretrained
FP32 weights when the job starts. Clients construct the same architecture, move it to their GPU,
and shard each decoder block and the root with `fully_shard()`, keeping tied embeddings together.
This initialization briefly holds a complete model on each GPU before sharding.

FSDP2 retains FP32 parameter shards and uses BF16 computation with FP32 gradient reduction.
Activation checkpointing reduces activation memory. AdamW is created after sharding and reset each
round, with learning rate `2e-5` and `foreach=False` to reduce temporary allocations.
Each site's `DistributedSampler` distributes its training blocks across ranks without overlap.

For each round:

1. Rank zero receives the global model. Both ranks call `set_model_state_dict()` with
   `full_state_dict=True` and `broadcast_from_rank0=True` to load their shards.
2. Both ranks evaluate the received global model on held-out blocks, then train on local data.
3. Both ranks call `get_model_state_dict()` with `full_state_dict=True` and `cpu_offload=True`.
   Rank zero sends the full CPU state to NVFlare, then releases it.
4. Rank zero shares stop signals and Client API errors so both ranks leave the loop together.

Each site reports `train_loss`, `val_loss`, and `peak_gpu_memory_gib`. Losses are averaged across
its ranks; validation scores the received model before local training. Peak memory is the maximum
PyTorch allocation across ranks during evaluation, training, and gathering; it excludes startup.
The server saves the last aggregate; validation does not select a best checkpoint.

`NUM_STEPS_CURRENT_ROUND` supplies the FedAvg weight. Both clients use the same batch size, rank
count, and step count. Optimizer state stays local. FSDP2 reduces GPU training memory, while rank
zero still needs CPU RAM for a full FP32 state, each transfer carries that state, and the server
needs memory for aggregation and persistence.

## Additional Resources

- [PyTorch FSDP2 tutorial and state-dict APIs](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html#state-dict-with-dcp-apis)
- [PyTorch DDP example](../pt/)
- [NVFlare Client API documentation](https://nvflare.readthedocs.io/en/main/programming_guide/execution_api_type/client_api.html)
