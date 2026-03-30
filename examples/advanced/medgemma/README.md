# Federated MedGemma Fine-Tuning

This example adapts Google's centralized [MedGemma fine-tuning notebook](https://github.com/google-health/medgemma/blob/main/notebooks/fine_tune_with_hugging_face.ipynb) into an NVFlare federated workflow. It fine-tunes [MedGemma 4B IT](https://huggingface.co/google/medgemma-4b-it) on the [NCT-CRC-HE-100K](https://zenodo.org/records/1214456) histopathology dataset using Hugging Face `TRL`, [QLoRA](https://arxiv.org/abs/2305.14314), and FedAvg across 3 clients.

The example is self-contained: data download/prep scripts live next to the FL client, model wrapper, and job recipe.

The fine-tuning data follows the official notebook's setup: [NCT-CRC-HE-100K](https://zenodo.org/records/1214456), a dataset of histopathology image patches from human colorectal cancer and normal tissue. The downstream task is a vision classification task framed as multimodal instruction tuning: for each image patch, the model is prompted with a multiple-choice tissue-type question and learns to generate one of nine labels (`adipose`, `background`, `debris`, `lymphocytes`, `mucus`, `smooth muscle`, `normal colon mucosa`, `cancer-associated stroma`, or `colorectal adenocarcinoma epithelium`).

## Before you start: Hugging Face gated model

Weights for [`google/medgemma-4b-it`](https://huggingface.co/google/medgemma-4b-it) are **gated** on Hugging Face. The repo is visible to everyone, but you must accept the license terms and be allowlisted before file downloads work.

1. Sign in at [huggingface.co](https://huggingface.co) and open [`google/medgemma-4b-it`](https://huggingface.co/google/medgemma-4b-it).
2. On the model page, follow **Access MedGemma on Hugging Face**: review and agree to the **Health AI Developer Foundations** terms so your account is authorized (this is often instant after you accept).
3. If you see an error such as *access is restricted and you are not in the authorized list*, you are not allowlisted yet—repeat the access step on the model page while logged into the correct account, or wait if access is pending.
4. On every machine that downloads or trains the model, use that **same** Hugging Face account: run `huggingface-cli login` or set the `HF_TOKEN` environment variable.

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
| `run_evaluation.py` | Evaluates base vs fine-tuned accuracy on `CRC-VAL-HE-7K`, following the MedGemma notebook's evaluation setup. |

## Prerequisites

- **Hugging Face:** allowlisted access and local auth for [`google/medgemma-4b-it`](https://huggingface.co/google/medgemma-4b-it), as described in [Before you start: Hugging Face gated model](#before-you-start-hugging-face-gated-model).
- Python 3.10+
- A CUDA GPU per client that supports `bfloat16` and has at least 40 GB of memory for MedGemma QLoRA.

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

To also download the `CRC-VAL-HE-7K` evaluation dataset used in the notebook's evaluation section:

```bash
python download_data.py --include_eval
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
- This keeps the full MedGemma checkpoint local to each client and exchanges only the smaller LoRA adapter updates between clients and server.
- `job.py` defaults to 3 clients and one GPU per client. If needed, override the GPU mapping with `--gpu "[0],[1],[2]"`.
- The client uses the same prompt format, 4-bit quantization, LoRA config, and multimodal collator pattern as the official notebook.
- Under heterogeneous or non-IID client data, simple FedAvg over LoRA adapters can introduce aggregation noise and dilute client-specific information, so heterogeneity-aware aggregation may perform better.

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

`run_inference.py` prints the ground-truth tissue label, the parsed model prediction, and the raw generated text for each sample.

Example output before fine-tuning:

```text
--- Sample 1 ---
Ground truth: A: adipose
Prediction:   G: normal colon mucosa
Raw output:   Based on the image, the most likely tissue type is **G: normal colon mucosa**.

Here's why:

*   The image shows a relatively uniform, pink-ish background with some

--- Sample 2 ---
Ground truth: I: colorectal adenocarcinoma epithelium
Prediction:   I: colorectal adenocarcinoma epithelium
Raw output:   Based on the image, the most likely tissue type is **(I) colorectal adenocarcinoma epithelium**.

Here's why:

*   **Epithelial cells:** The image shows a collection of
```

Global model after fine-tuning:

```text
--- Sample 1 ---
Ground truth: A: adipose
Prediction:   A: adipose
Raw output:   A: adipose

--- Sample 2 ---
Ground truth: I: colorectal adenocarcinoma epithelium
Prediction:   I: colorectal adenocarcinoma epithelium
Raw output:   I: colorectal adenocarcinoma epithelium
```

This qualitative shift suggests that fine-tuning is capturing the downstream task well. Before fine-tuning, the base model often answers in a more open-ended, explanatory style and can miss the target class label. After fine-tuning, the global model responds in the compact label format used during training and aligns more directly with the expected tissue-classification output.

## 5. Evaluate accuracy before and after fine-tuning

The MedGemma [fine-tuning notebook](https://github.com/google-health/medgemma/blob/main/notebooks/fine_tune_with_hugging_face.ipynb) evaluates on the separate [CRC-VAL-HE-7K](https://zenodo.org/records/1214456) dataset rather than on the fine-tuning split. This example includes `run_evaluation.py` to mirror that setup and compute accuracy for the base model and the fine-tuned global model on the same evaluation subset.

First, download the evaluation dataset if you have not already:

```bash
python download_data.py --include_eval
```

Then run:

```bash
python run_evaluation.py \
  --dataset_dir ./CRC-VAL-HE-7K \
  --tuned_model_path /tmp/nvflare/simulation/medgemma/server/simulate_job/app_server/FL_global_model.pt
```

By default, `run_evaluation.py`:

- uses `google/medgemma-4b-it` as the before-fine-tuning baseline,
- evaluates on a shuffled subset of **1000 samples**, matching the notebook's default,
- computes accuracy for both models and reports the delta.

> Note, this can take a while to execute but should produce the following result:

**Evaluation Result**
```
Accuracy summary
Base model:       accuracy=0.4130 (413/1000), unparsed=0
Fine-tuned model: accuracy=0.9540 (954/1000), unparsed=0
Delta:            accuracy=+0.5410
```

Useful flags:

- `--max_samples 0` evaluates the full CRC-VAL-HE-7K dataset.
- `--show_examples 2` prints a couple of qualitative prediction examples per model before the summary.
- `--base_model_path /path/to/model` evaluates a different unfine-tuned checkpoint as the baseline.

## Sources

- Official centralized baseline: [fine_tune_with_hugging_face.ipynb](https://github.com/google-health/medgemma/blob/main/notebooks/fine_tune_with_hugging_face.ipynb)
- MedGemma model: [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it)
- Training data: [NCT-CRC-HE-100K on Zenodo](https://zenodo.org/records/1214456)
- Heterogeneous LoRA reference: [Heterogeneous LoRA for Federated Fine-tuning of On-Device Foundation Models (HetLoRA)](https://research.google/pubs/heterogeneous-lora-for-federated-fine-tuning-of-on-device-foundation-models/)
