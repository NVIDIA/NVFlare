# FedUMM: Federated Learning for Unified Multimodal Models

> **Paper:** [FedUMM: A General Framework for Federated Learning with Unified Multimodal Models](https://arxiv.org/abs/2601.15390)
>
> Built on [NVIDIA FLARE](https://github.com/NVIDIA/NVFlare) · LoRA adapter-only aggregation · Dirichlet non-IID data partitioning

## What Is This?

This is the code for **FedUMM** — a federated learning framework that lets multiple
clients collaboratively fine-tune vision-language models (VLMs) **without sharing
their private data**. Only lightweight LoRA adapter weights (~1-2 MB) are exchanged
each round, reducing communication by **99.7%** compared to full model fine-tuning.

```
                 Server (FedAvg)
            aggregate LoRA deltas only
                /       |       \
         Client 1   Client 2   Client K
         LoRA on    LoRA on    LoRA on
          VLM        VLM        VLM
         (data      (data      (data
          stays      stays      stays
          local)     local)     local)
```

## Supported Models

| Backend key | Model | LoRA target | Trainable params |
|---|---|---|---|
| `blip_vqa` | Salesforce/blip-vqa-base | text_encoder + text_decoder (q/k/v) | ~1.2 M |
| `januspro` | deepseek-ai/Janus-Pro-1B | language_model (q_proj/k_proj/v_proj) | ~2.4 M |

## Project Structure

```
fedumm/
├── src/
│   ├── __init__.py              # Auto-registers backends
│   ├── model_registry.py        # register / get backend
│   ├── common.py                # Seeding, Dirichlet partition, train loop
│   ├── blip_backend.py          # BLIP VQA backend
│   ├── januspro_backend.py      # JanusPro backend
│   ├── fl_client.py             # NVFlare FL client (all models)
│   └── local_train.py           # Centralized baseline (no FL)
├── envs/
│   ├── env_blip.yml             # Conda env for BLIP
│   └── env_januspro.yml         # Conda env for JanusPro
├── scripts/
│   ├── setup_envs.sh            # Create both conda envs
│   ├── launch_blip.sh           # Wrapper: activate env -> run python
│   ├── launch_januspro.sh       # Wrapper: activate env -> run python
│   └── slurm_run.sh             # HPC batch job template
├── job.py                       # FedJob config (FedAvg + ScriptRunner)
├── requirements.txt             # Minimal deps for single-env usage
└── README.md
```

---

## Quick Start (3 steps)

### Step 1: Install

```bash
# Option A: pip (simplest)
pip install nvflare torch transformers peft datasets Pillow numpy

# Option B: conda (recommended for HPC)
conda create -n fedumm python=3.10 -y
conda activate fedumm
pip install nvflare torch transformers peft datasets Pillow numpy
```

### Step 2: Run Centralized Baseline

Verify everything works before starting federated training:

```bash
python src/local_train.py \
    --model_backend blip_vqa \
    --batch_size 8 \
    --num_epochs 1 \
    --max_train_samples 500 \
    --max_eval_samples 100
```

Expected output:
```
Train: 500, Eval: 100
trainable: 1,179,648 / 252,562,178 (0.4672%)
Epoch 1/1  loss=2.3456  acc=0.3200
```

### Step 3: Run Federated Learning

```bash
python job.py \
    --model_backend blip_vqa \
    --num_clients 2 \
    --num_rounds 3 \
    --dirichlet_alpha 0.5 \
    --max_train_samples 500 \
    --max_eval_samples 100 \
    --simulator
```

That's it! The simulator runs everything on one machine.

---

## Reproducing Paper Experiments

### Table 1: VQA v2 with varying clients and heterogeneity

The paper tests K = {2, 4, 8, 16} clients with alpha = {0.1, 0.5, 1.0}.

```bash
# alpha=0.1 (extreme non-IID), K=8 clients
python job.py --model_backend blip_vqa \
    --num_clients 8 --num_rounds 100 --local_epochs 5 \
    --dirichlet_alpha 0.1 \
    --batch_size 32 --lr 2e-5 \
    --simulator

# alpha=0.5 (moderate non-IID), K=8 clients
python job.py --model_backend blip_vqa \
    --num_clients 8 --num_rounds 100 --local_epochs 5 \
    --dirichlet_alpha 0.5 \
    --batch_size 32 --lr 2e-5 \
    --simulator

# alpha=1.0 (mild non-IID), K=16 clients
python job.py --model_backend blip_vqa \
    --num_clients 16 --num_rounds 100 --local_epochs 5 \
    --dirichlet_alpha 1.0 \
    --batch_size 32 --lr 2e-5 \
    --simulator
```

### What `--dirichlet_alpha` does

Controls how "uneven" the data distribution is across clients:

```
alpha=0.1 (extreme)          alpha=0.5 (moderate)        alpha=1.0 (mild)

Client 0: ████████ yes/no    Client 0: ████ yes/no       Client 0: ███ yes/no
          █ color                      ███ color                    ███ color
          █ count                      ███ count                    ██ count

Client 1: █ yes/no           Client 1: ███ yes/no        Client 1: ███ yes/no
          ████████ color               ████ color                   ███ color
          █ count                      ██ count                     ███ count

Each client specializes      Clients have preferences    Nearly uniform
in a few answer types        but still see variety       distribution
```

| alpha value | Meaning | When to use |
|---|---|---|
| `0` (default) | IID round-robin | Baseline / debugging |
| `0.1` | Extreme non-IID | Stress test, worst-case scenario |
| `0.5` | Moderate non-IID | Realistic hospital/enterprise setting |
| `1.0` | Mild non-IID | Near-IID with slight variation |

### Paper hyperparameters

| Parameter | Paper value | CLI flag |
|---|---|---|
| LoRA rank | 16 | `--lora_r 16` |
| LoRA alpha | 32 | `--lora_alpha 32` |
| Learning rate | 2e-5 | `--lr 2e-5` |
| Batch size | 32 | `--batch_size 32` |
| Local epochs | 5 | `--local_epochs 5` |
| Communication rounds | 100 | `--num_rounds 100` |
| Clients | 2-16 | `--num_clients K` |
| Optimizer | AdamW (wd=0.05) | hardcoded |

---

## All CLI Options

### `job.py` (Federated Learning)

```
python job.py [OPTIONS]

Model:
  --model_backend        blip_vqa | januspro              (required)
  --model_name_or_path   HuggingFace model id             (uses default if empty)

Federated Learning:
  --num_clients          Number of FL clients              (default: 2)
  --num_rounds           Communication rounds              (default: 3)
  --local_epochs         Local epochs per round            (default: 1)
  --dirichlet_alpha      Non-IID level: 0=IID, 0.1/0.5/1.0  (default: 0)

Training:
  --batch_size           Per-client batch size             (default: 8)
  --grad_accum           Gradient accumulation steps       (default: 8)
  --lr                   Learning rate                     (default: 5e-5)
  --lora_r               LoRA rank                         (default: 16)
  --lora_alpha           LoRA scaling factor               (default: 32)

Data:
  --max_train_samples    Limit training data (-1 = all)    (default: -1)
  --max_eval_samples     Limit eval data (-1 = all)        (default: -1)
  --data_path            HuggingFace cache directory        (default: "")
  --seed                 Random seed                       (default: 42)

Execution:
  --simulator            Run with NVFlare simulator (single machine)
  --export_dir           Export job config for production deployment
  --use_env_script       Use conda env isolation via bash wrappers
```

### `src/local_train.py` (Centralized Baseline)

```
python src/local_train.py [OPTIONS]

  --model_backend        blip_vqa | januspro
  --num_epochs           Training epochs                   (default: 1)
  --output_dir           Save model here                   (default: ./workspace_centralized)
  (... same training/data options as job.py)
```

---

## HPC / Slurm Usage

### Interactive testing

```bash
srun --gres=gpu:1 --mem=32G --time=02:00:00 --pty bash
conda activate fedumm
python src/local_train.py --model_backend blip_vqa \
    --max_train_samples 200 --max_eval_samples 50 \
    --data_path /scratch/$USER/hf_cache
```

### Batch job

```bash
mkdir -p logs
sbatch scripts/slurm_run.sh blip_vqa
```

Edit `scripts/slurm_run.sh` to match your cluster (partition name, GPU type,
module names).

### Pre-downloading data (for compute nodes without internet)

```bash
# On login node (has internet):
python -c "
from datasets import load_dataset
ds = load_dataset('HuggingFaceM4/VQAv2', split='train',
                  cache_dir='/scratch/\$USER/hf_cache')
print(f'Downloaded {len(ds)} samples')
"

# Then on compute node:
python job.py --data_path /scratch/$USER/hf_cache ...
```

---

## Environment Isolation (for JanusPro)

JanusPro requires the `janus` package from GitHub which may conflict with
other dependencies. Use conda env isolation:

```bash
# 1. Create isolated environments
bash scripts/setup_envs.sh

# 2. Run with env isolation
python job.py --model_backend januspro --use_env_script --simulator
```

How it works: NVFlare's `ScriptRunner` calls a bash wrapper that activates
the correct conda env before running the Python training script.

```
job.py --use_env_script
  -> ScriptRunner(command="bash scripts/launch_januspro.sh")
    -> source conda.sh && conda activate nvflare_januspro
      -> exec python -u fl_client.py "$@"
```

---

## Adding a New Model

1. **Create backend** `src/my_model_backend.py`:

```python
from .model_registry import register_backend

class MyModelBackend:
    name = "my_model"

    def build_model_and_processor(self, model_name_or_path, lora_r,
                                  lora_alpha, lora_dropout, device):
        # Load model, apply LoRA, return (model, processor)
        ...

    def build_dataset(self, hf_ds, processor, max_q_len, max_a_len):
        # Wrap HF dataset into torch Dataset
        ...

    def collate_fn(self, batch):
        # Custom batching
        ...

    def train_step(self, model, batch, device):
        # Forward pass, return scalar loss (no .backward())
        ...

    def evaluate(self, model, dataloader, processor, device):
        # Return metric (e.g. VQA accuracy float)
        ...

    def hf_dataset_name(self):    return "HuggingFaceM4/VQAv2"
    def hf_train_split(self):     return "train"
    def hf_eval_split(self):      return "validation[:50%]"
    def keep_columns(self):       return ["image", "question",
                                          "multiple_choice_answer", "answers"]

register_backend("my_model", MyModelBackend())
```

2. **Register** in `src/__init__.py`:
```python
try:
    from . import my_model_backend  # noqa: F401
except ImportError:
    pass
```

3. **Use it**:
```bash
python job.py --model_backend my_model --simulator
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `Unknown backend 'blip_vqa'. Available: ` | Files not in `src/` package. Check `src/__init__.py` exists. |
| `EnvironmentFileNotFound` in `setup_envs.sh` | Run from project root: `bash scripts/setup_envs.sh` |
| CUDA out of memory | Reduce `--batch_size` to 2-4, increase `--grad_accum` |
| `trust_remote_code` blocked (JanusPro) | `export HF_HUB_TRUST_REMOTE_CODE=1` |
| No internet on compute node | Pre-download data, then use `--data_path` |
| Very slow first run | HuggingFace is downloading VQAv2 (~50 GB). Use `--max_train_samples 500` for testing. |

---

## Citation

```bibtex
@article{su2026fedumm,
    title={FedUMM: A General Framework for Federated Learning
           with Unified Multimodal Models},
    author={Su, Zhaolong and Zhao, Leheng and Wu, Xiaoying
            and Xu, Ziyue and Wang, Jindong},
    journal={arXiv preprint arXiv:2601.15390},
    year={2026}
}
```

## License

This example is licensed under the Apache License 2.0.
Model weights are subject to their respective licenses
(Salesforce for BLIP, DeepSeek Model License for JanusPro).

## Acknowledgments

This work was partially supported by the NVIDIA Academic Grant Program.
