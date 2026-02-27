# Federated Multi-modal Fine-Tuning with Qwen3-VL

This example shows how to fine-tune [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) (vision-language) in a federated setting using [NVIDIA FLARE](https://nvflare.readthedocs.io/), with the [PubMedVision](https://huggingface.co/datasets/FreedomIntelligence/PubMedVision) medical VQA dataset split across 3 clients.

## Code structure

As in typical NVFlare examples (e.g. [hello-pt](../../hello-world/hello-pt/)):

| File | Role |
|------|------|
| `model.py` | Qwen3-VL wrapper used as the FL model; server can save/load `state_dict`. Model config uses HuggingFace ID (e.g. `Qwen/Qwen3-VL-2B-Instruct`). |
| `client.py` | Client entry point (launched by NVFlare via torchrun): receives global model, runs Qwen3-VL `train_qwen` in-process per round, sends updated weights back. Requires Qwen repo and `fl_site` in data_list (see below). |
| `job.py` | FedAvg recipe: 3 clients, per-site data paths; optional Weights & Biases tracking (--wandb); launches each client with a per-site torchrun command (unique `--master_port` per client). |
| `download_data.py` | Downloads PubMedVision (git clone) and unzips image archives into `PubMedVision/` for a standard layout. |
| `prepare_data.py` | Splits PubMedVision into `site-1`, `site-2`, `site-3` shards. |
| `run_inference.py` | Inference on PubMedVision-style samples; compares base vs fine-tuned (HuggingFace checkpoint or NVFlare `FL_global_model.pt`). Prints question, ground truth, model answer, and image path. |

## Prerequisites

- Python 3.10+
- CUDA-capable GPU with substantial VRAM. Training has been tested on **3× NVIDIA H100** (one GPU per client; 94GB available VRAM per GPU).
- [Git LFS](https://git-lfs.com/) (for cloning dataset repos)

## 1. Virtual environment and dependencies

Create and activate a virtual environment, then install NVFlare and this example’s dependencies (including PyTorch and `flash_attn`):

```bash
python3.12 -m venv .venv
source .venv/bin/activate

# Install NVFlare and Qwen3-VL requirements
./install_requirements.sh
```

## 2. Clone the Qwen3-VL repo (SFT scripts)

The official SFT training scripts and utilities (e.g. `qwen-vl-finetune/scripts/sft.sh`, data handling, and model configs) live in the [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) repo. Clone it next to (or inside) this example so the federated job can call into their scripts:

```bash
# From examples/advanced/qwen3-vl (or your preferred location)
git clone https://github.com/QwenLM/Qwen3-VL.git
```

Set the `QWEN3VL_ROOT` environment variable to the clone path so the example can find the SFT entrypoint and configs. E.g. if you cloned inside this example: 

```bash
export QWEN3VL_ROOT="${PWD}/Qwen3-VL"
```
(Use the absolute path to your `Qwen3-VL` clone if different.)

## 3. Data: PubMedVision

The example uses the [PubMedVision](https://huggingface.co/datasets/FreedomIntelligence/PubMedVision) medical VQA dataset. First, **download** the dataset and unzip its image archives into a local folder; then **split** it into non-overlapping shards (one per federated client) under `./data/site-1`, `site-2`, `site-3`. Both steps are automated.

From this directory (`examples/advanced/qwen3-vl`), run:

```bash
python download_data.py
python prepare_data.py
```

`download_data.py` clones the [PubMedVision](https://huggingface.co/datasets/FreedomIntelligence/PubMedVision) repo into `PubMedVision/` and unzips the image archives (requires [git](https://git-scm.com/) and [Git LFS](https://git-lfs.com/)). Unzipping can take a while. To use a different target directory: `python download_data.py --output_dir /path/to/PubMedVision`.

`prepare_data.py` defaults to `--data_file PubMedVision/PubMedVision_InstructionTuning_VQA.json` and `--output_dir ./data`, so after `download_data.py` no arguments are needed. To load from HuggingFace Hub instead of a local file: `python prepare_data.py --data_file ''`. This produces `./data/site-1`, `site-2`, `site-3` with non-overlapping shards.

**Expected output (prepare_data.py):**
```
Loading from local file: PubMedVision/PubMedVision_InstructionTuning_VQA.json
  site-1: 215586 samples -> ./data/site-1/train.json
  site-2: 215586 samples -> ./data/site-2/train.json
  site-3: 215586 samples -> ./data/site-3/train.json
Done. Client data under ./data/site-1, site-2, site-3.
```

Optional flags:

- `--subset_size 5000` — use at most 5000 samples per client (for a quick run).
- `--num_clients 3` — number of client shards (default: 3).
- `--image_root /path/to/PubMedVision` — resolve relative image paths and exclude samples whose image file(s) do not exist.

Output layout (same JSON format as the source, one file per client):

- `./data/site-1/train.json`
- `./data/site-2/train.json`
- `./data/site-3/train.json`

## 4. Weights & Biases setup (optional)

WandB is disabled by default. To enable experiment tracking, pass `--wandb` when running the job and set up logging:

**Authentication:** Set the `WANDB_API_KEY` environment variable before running the job (for local FL simulation only):

```bash
export WANDB_API_KEY=your_key_here
```

Get your API key from [wandb.ai/authorize](https://wandb.ai/authorize). Alternatively, run `wandb.login()` in Python and enter the key when prompted; it is stored for future runs.

**Configuration:** When `--wandb` is used, `job.py` configures WandB with:
- `name`: "qwen3-vl-fedavg"
- `project`: "nvflare"
- `group`: "nvidia"
- `job_type`: "training"

You can modify these in `job.py` if needed. If `WANDB_API_KEY` is not set, WandB will run in offline mode.

## 5. Run the federated job

With data under `./data/site-{1,2,3}/` (from `prepare_data.py`), run:

```bash
python job.py
```

`job.py` defaults to `--data_dir ./data`. Training uses the official [Qwen3-VL fine-tuning script](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/scripts/sft.sh) (`train_qwen`): the FL client (`client.py`) is started by NVFlare with torchrun, receives the global model, runs `train_qwen` in-process, and sends the updated weights back.

**Prerequisites for the job:** Clone Qwen3-VL and set `QWEN3VL_ROOT` (see step 2). The example's Qwen3-VL clone registers a `fl_site` dataset; the client sets `FL_SITE_DATA_DIR` to its site data dir (e.g. `./data/site-1`). If images live in a separate path, set `PUBMEDVISION_IMAGE_ROOT` to the folder containing `images/`.

**Optional arguments:**

- Single client (testing): `python job.py --n_clients 1 --max_steps 1000`
- One GPU per client: `python job.py --gpu "[0],[1],[2]"`
- WandB: `python job.py --wandb` (and set `WANDB_API_KEY` for online logging)

With 3 clients, omitting `--gpu` defaults to `[0],[1],[2]`. `--max_steps` limits steps per round (omit to train a full epoch per round).

## Checkpoints and disk space

By default, the client saves received and trained model checkpoints under the **NVFlare client workspace** (a `qwen3vl_checkpoints` subdir of the current working directory). When you run with SimEnv, that workspace is cleared every run, so you don't need to clean up manually. If you see **"No space left on device"**, the partition containing the SimEnv workspace (e.g. `/tmp/nvflare/simulation`) is full—use a different `workspace_root` in SimEnv or pass `--work_dir /path/to/large/disk` in the client's train_args to override the default.


## Timeouts and long runs

After each round the client sends the updated model weights back to the server; for large VL models this transfer can take several minutes. The executor that talks to the client script uses a **peer_read_timeout** (e.g. 300s in the framework) when sending the next round’s task: the script must return to `flare.receive()` within that time. This example avoids loading the full model after training by loading only the **state dict** from the checkpoint (`.safetensors` / `pytorch_model.bin`), so the script can send the result and get back to `receive()` sooner and reduce the chance of "failed to send 'train' ... timeout" and subsequent FOBS download errors. If you still see timeouts with very large models or slow links, you may need to increase the executor’s `peer_read_timeout` in the NVFlare codebase or wait for a configurable option.

### Client errors and model size

- **One site shows "(after error)" every round:** Training failed on that client (e.g. OOM, missing data, GPU). Check that site’s log (e.g. `.../site-1/log.txt`) for `Qwen SFT script failed:` and the exception message; the client also prints a short error hint next to `(after error)`.
- **Received vs sent model size:** The initial model is loaded in bf16 in `model.py` so the server sends the global model in bf16 (~4651 MB for 2B); otherwise `from_pretrained` can default to float32 (~9302 MB). On **success** the client sends the bf16 checkpoint (~4651 MB). On **error** the client sends the in-memory model (same dtype as received), so sent size matches received.

## Inference (before/after comparison)

Use `run_inference.py` to compare base vs fine-tuned checkpoints on PubMedVision-style samples. Run from this directory (`examples/advanced/qwen3-vl`):

By default the script uses `./data/site-1/train.json` and `./PubMedVision` (the directory that contains the `images/` folder). Override with `--data_file` and `--image_root` if needed. Output prints each question, ground-truth answer, and model answer so you can compare base vs fine-tuned runs.

### Base model (no fine-tuning)
```bash
python run_inference.py --model_path Qwen/Qwen3-VL-2B-Instruct
```
(Uses `./data/site-1/train.json` and `./PubMedVision` by default after the data workflow.)
**Expected output**
```
--- Sample 1 ---
Image: /path/to/PubMedVision/images/...
Q: <image>
What is the size of the inferior mesenteric artery aneurysm (IMAA) based on the CT angiography scan?
Ground truth: According to the measurements provided in the image, the inferior mesenteric artery aneurysm (IMAA) has a length of 4.87 cm and a width of 5.93 cm. The reference information indicates that this repres...
Model:        Based on the provided CT angiography scan, the inferior mesenteric artery aneurysm (IMAA) has a length of 4.87 cm.
```

### Fine-tuned checkpoint

**Option A — NVFlare global model** (single `.pt` file saved by the server, e.g. `FL_global_model.pt`). The script loads the architecture and processor from the default base model (Qwen/Qwen3-VL-2B-Instruct), then applies the saved weights; use `--base_model` if you used a different base in the FL job:

```bash
python run_inference.py --model_path /tmp/nvflare/simulation/qwen3-vl/server/simulate_job/app_server/FL_global_model.pt
```

**Expected output** (fine-tuned): same format as base; compare the "Model" line to the base run.
```
--- Sample 1 ---
Image: /path/to/PubMedVision/images/...
Q: <image>
What is the size of the inferior mesenteric artery aneurysm (IMAA) based on the CT angiography scan?
Ground truth: According to the measurements provided in the image, the inferior mesenteric artery aneurysm (IMAA) has a length of 4.87 cm and a width of 5.93 cm. The reference information indicates that this repres...
Model:        The inferior mesenteric artery aneurysm (IMAA) measures 4.87 cm in length.
```

**Option B — HuggingFace-style checkpoint directory** (e.g. from the FL client workspace or a saved `checkpoint-xxx` folder):

```bash
python run_inference.py --model_path ./path/to/checkpoint-xxx
```

> The example output above may look similar for base and fine-tuned checkpoints because the baseline is already strong; the intent here is to demonstrate federated fine-tuning and inference on the global checkpoint. Note that we use the same training data (e.g. `site-1/train.json`) only for illustration -- for real evaluation you should use a held-out validation or test set.

## Summary

| Step | Action |
|------|--------|
| 1 | Create venv, `pip install nvflare`, run `./install_requirements.sh` |
| 2 | Clone Qwen3-VL, set `QWEN3VL_ROOT` (example includes `fl_site` in data_dict per [Dataset config](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune#dataset-config-for-training)) |
| 3 | Data: `python download_data.py` then `python prepare_data.py` to get `./data/site-{1,2,3}/` |
| 4 | (Optional) Set `WANDB_API_KEY` and pass `--wandb` for experiment tracking |
| 5 | Run `python job.py` (optionally `--wandb`, `--max_steps N`, `--gpu "[0],[1],[2]"`) |

Standard workflow from this directory: `python download_data.py` → `python prepare_data.py` → `python job.py`.

## References

- [Qwen3-VL repo](https://github.com/QwenLM/Qwen3-VL) — SFT scripts and configs (e.g. `qwen-vl-finetune/scripts/sft.sh`)
- [Dataset config for training](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune#dataset-config-for-training) — add/register datasets in `data/__init__.py`
- [PubMedVision](https://huggingface.co/datasets/FreedomIntelligence/PubMedVision) — medical VQA dataset
- [NVIDIA FLARE](https://nvflare.readthedocs.io/) — federated learning framework
