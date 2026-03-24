# Federated MedGemma Fine-Tuning

This example adapts Google's centralized [MedGemma fine-tuning notebook](https://github.com/google-health/medgemma/blob/main/notebooks/fine_tune_with_hugging_face.ipynb) into an NVFlare federated workflow. It fine-tunes [MedGemma 4B IT](https://huggingface.co/google/medgemma-4b-it) on the [NCT-CRC-HE-100K](https://zenodo.org/records/1214456) histopathology dataset using Hugging Face `TRL`, QLoRA, and FedAvg across 3 clients.

The example is self-contained: data download/prep scripts live next to the FL client, model wrapper, and job recipe.

## Code structure

| File | Role |
|------|------|
| `data_utils.py` | Shared prompt, class-label mappings, dataset splitting helpers, and response parsing helpers. |
| `model.py` | MedGemma LoRA wrapper used by the server and clients. The server exchanges only LoRA adapter weights. |
| `client.py` | NVFlare client entry point. Loads MedGemma in 4-bit, applies the received LoRA adapters, runs local SFT with `SFTTrainer`, and sends the updated adapters back. |
| `job.py` | FedAvg recipe for 3 clients with per-site data paths. |
| `download_data.py` | Downloads and extracts `NCT-CRC-HE-100K.zip` from Zenodo. |
| `prepare_data.py` | Discovers image files, builds random site shards, and writes `train.json` / `validation.json` for each client. |
| `run_inference.py` | Runs before/after inference on prepared validation samples using either the base model, an adapter directory, or NVFlare `FL_global_model.pt`. |

## Prerequisites

- Python 3.10+
- A CUDA GPU per client that supports `bfloat16` and has enough memory for MedGemma QLoRA. The upstream notebook recommends at least 40 GB GPU memory.
- Access to [`google/medgemma-4b-it`](https://huggingface.co/google/medgemma-4b-it) on Hugging Face
- A Hugging Face login/token already configured locally for gated-model access

## 1. Install dependencies

From this directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
./install_requirements.sh
```

## 2. Download and prepare data

Download and extract the NCT-CRC-HE-100K archive:

```bash
python download_data.py
```

Then create client splits:

```bash
python prepare_data.py
```

By default, `prepare_data.py` creates 3 client shards with `3333` samples per client and `333` validation samples per client. That keeps the total dataset size close to the official MedGemma notebook's 10k-sample walkthrough while preserving a site-based FL layout.

Output:

```
  site-1: train=3000 -> ./data/site-1/train.json, validation=333 -> ./data/site-1/validation.json
  site-2: train=3000 -> ./data/site-2/train.json, validation=333 -> ./data/site-2/validation.json
  site-3: train=3000 -> ./data/site-3/train.json, validation=333 -> ./data/site-3/validation.json
```

Useful flags:

- `--samples_per_client 0` uses the full dataset evenly across all clients.
- `--validation_size_per_client 0` disables per-site validation files.
- `--dataset_dir /path/to/NCT-CRC-HE-100K` points to a different extracted dataset location.

## 3. Run the federated job

With data prepared under `./data/site-{1,2,3}`:

```bash
python job.py
```

Important notes:

- The example always uses LoRA-only FL exchange. The MedGemma base model stays frozen on every client, and FedAvg aggregates only the adapter weights.
- `job.py` defaults to 3 clients and one GPU per client. If needed, override the GPU mapping with `--gpu "[0],[1],[2]"`.
- The client uses the same prompt format, 4-bit quantization, LoRA config, and multimodal collator pattern as the official notebook.

Useful flags:

| Option | Description | Example |
|--------|-------------|---------|
| `--workspace` | Override the simulator workspace root. | `python job.py --workspace /data/nvflare/sim` |
| `--n_clients` | Number of federated clients (default: 3). Must match the number of `--gpu` groups and prepared `data/site-*` directories. | `python job.py --n_clients 5` |
| `--num_rounds` | Number of FL rounds. | `python job.py --num_rounds 5` |
| `--max_steps` | Limit local steps per round for quick tests. | `python job.py --max_steps 50` |
| `--learning_rate` | Peak learning rate for local SFT. | `python job.py --learning_rate 1e-4` |
| `--wandb` | Enable Weights & Biases tracking if `WANDB_API_KEY` is set. | `python job.py --wandb` |

## 4. Run inference

Compare the base model against the FL-trained global adapter on prepared validation samples.

Base model:

```bash
python run_inference.py --model_path google/medgemma-4b-it
```

Federated fine-tuned global model:

```bash
python run_inference.py --model_path /tmp/nvflare/simulation/medgemma/server/simulate_job/app_server/FL_global_model.pt
```

Adapter directory saved by a trainer/client:

```bash
python run_inference.py --model_path /path/to/adapter-checkpoint
```

`run_inference.py` prints the ground-truth tissue label, the parsed model prediction, and the raw generated text for each sample.

## Sources

- Official centralized baseline: [fine_tune_with_hugging_face.ipynb](https://github.com/google-health/medgemma/blob/main/notebooks/fine_tune_with_hugging_face.ipynb)
- MedGemma model: [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it)
- Training data: [NCT-CRC-HE-100K on Zenodo](https://zenodo.org/records/1214456)
