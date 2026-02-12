# Federated Multi-modal Fine-Tuning with Qwen3-VL

This example shows how to fine-tune [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) (vision-language) in a federated setting using [NVIDIA FLARE](https://nvflare.readthedocs.io/), with the [PubMedVision](https://huggingface.co/datasets/FreedomIntelligence/PubMedVision) medical VQA dataset split across 3 clients.

## Code structure

As in typical NVFlare examples (e.g. [hello-pt](../../hello-world/hello-pt/)):

| File | Role |
|------|------|
| `model.py` | Qwen3-VL wrapper used as the FL model; server can save/load `state_dict`. Model config uses HuggingFace ID (e.g. `Qwen/Qwen2.5-VL-3B-Instruct`). |
| `client_sft_runner.py` | Client entry point: receives global model, runs the official Qwen3-VL `train_qwen.py` script as a subprocess per round, sends updated weights back. Requires Qwen repo and `fl_site` in data_list (see below). |
| `client_wrapper.sh` | Wrapper script (same pattern as [llm_hf MULTINODE](../llm_hf/MULTINODE.md)): job runs `bash custom/client_wrapper.sh` with script + args; wrapper launches `client_sft_runner.py`. |
| `job.py` | FedAvg recipe: 3 clients, per-site data paths, Weights & Biases tracking; always uses the Qwen SFT script via the runner. |
| `prepare_data.py` | Splits PubMedVision into `site-1`, `site-2`, `site-3` shards |

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended: 18GB+ VRAM for the 3B model)
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

From the [PubMedVision dataset page](https://huggingface.co/datasets/FreedomIntelligence/PubMedVision):

```bash
# Install Git LFS if needed: https://git-lfs.com/
git clone https://huggingface.co/datasets/FreedomIntelligence/PubMedVision
```

Unzip the image files (**Note** this might take a while):

```bash
cd PubMedVision
for ((i=0; i<20; i++))
do
    echo "Unzipping archive $((i+1))/20 ..."
    unzip -q -j images_$i.zip -d images/
done
cd ..
```

### Split data for 3 federated clients

Run the preparation script so each client has a non-overlapping shard under `./data/site-1`, `site-2`, `site-3`:

**Using your local JSON file (e.g. from the cloned PubMedVision repo):**

If the JSON is in the current directory:

```bash
python prepare_data.py --data_file PubMedVision/PubMedVision_InstructionTuning_VQA.json --output_dir ./data
```

**Expected output:**
```
Loading from local file: PubMedVision/PubMedVision_InstructionTuning_VQA.json
Creating json from Arrow format: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 216/216 [00:02<00:00, 82.50ba/s]
  site-1: 215586 samples -> ./data/site-1/train.json
Creating json from Arrow format: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 216/216 [00:02<00:00, 82.70ba/s]
  site-2: 215586 samples -> ./data/site-2/train.json
Creating json from Arrow format: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 216/216 [00:02<00:00, 83.10ba/s]
  site-3: 215586 samples -> ./data/site-3/train.json
Done. Client data under ./data/site-1, site-2, site-3.
```

Optional flags:

- `--subset_size 5000` — use at most 5000 samples per client (for a quick run).
- `--num_clients 3` — number of client shards (default: 3).

Output layout (same JSON format as the source, one file per client):

- `./data/site-1/train.json`
- `./data/site-2/train.json`
- `./data/site-3/train.json`

## 4. Weights & Biases setup (optional but recommended)

The example uses Weights & Biases (WandB) for experiment tracking. To enable online logging:

**Authentication:** Set the `WANDB_API_KEY` environment variable before running the job (for local FL simulation only):

```bash
export WANDB_API_KEY=your_key_here
```

Get your API key from [wandb.ai/authorize](https://wandb.ai/authorize). Alternatively, run `wandb.login()` in Python and enter the key when prompted; it is stored for future runs.

**Configuration:** The `job.py` script configures WandB with (see `job.py` around lines 95–98):
- `name`: "qwen3-vl-fedavg"
- `project`: "nvflare"
- `group`: "nvidia"
- `job_type`: "training"

You can modify these in `job.py` if needed. If `WANDB_API_KEY` is not set, WandB will run in offline mode.

## 5. Run the federated job

Training uses the official [Qwen3-VL fine-tuning script](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/scripts/sft.sh) (`train_qwen.py`) as a subprocess per round, following the same pattern as [llm_hf MULTINODE](../llm_hf/MULTINODE.md): the FL client runs `client_wrapper.sh`, which launches `client_sft_runner.py`; the runner receives the global model, runs `train_qwen.py`, and sends the updated weights back.

1. **Clone Qwen3-VL and set `QWEN3VL_ROOT`** (see step 2 above).

2. **Add a `fl_site` dataset entry** in the Qwen repo so the script can load each site’s `train.json`. The runner sets the env var `FL_SITE_DATA_DIR` to the site data dir (e.g. `./data/site-1`) before calling the script. In the Qwen3-VL repo, edit `qwen-vl-finetune/qwenvl/data/__init__.py` (or the module that defines `data_list`):

   - In the `data_list(dataset_names)` function, inside the loop over `dataset_names`, add a branch for `"fl_site"` (after `sampling_rate = parse_sampling_rate(...)` and `dataset_name = re.sub(r"%(\d+)$", "", dataset_name)`):

   ```python
   if dataset_name == "fl_site":
       base = os.environ.get("FL_SITE_DATA_DIR", "")
       if not base:
           raise ValueError("FL_SITE_DATA_DIR is not set for fl_site dataset")
       config = {
           "annotation_path": os.path.join(base, "train.json"),
           "data_path": base,
           "sampling_rate": sampling_rate,
       }
       config_list.append(config)
       continue
   ```

   Add `import os` at the top of that file if not already present.

3. **Run the job**:

   ```bash
   export QWEN3VL_ROOT=/path/to/Qwen3-VL
   export WANDB_API_KEY=your_key_here   # optional
   python job.py --data_dir ./data --max_steps 50
   ```

   `--max_steps` limits steps per round (default: 50).

## Summary

| Step | Action |
|------|--------|
| 1 | Create venv, `pip install nvflare`, run `./install_requirements.sh` |
| 2 | Clone Qwen3-VL, set `QWEN3VL_ROOT`, and add `fl_site` to `data_list` in the repo |
| 3 | Download PubMedVision (clone or Hub) and run `prepare_data.py` to get `./data/site-{1,2,3}/` |
| 4 | (Optional) Set `WANDB_API_KEY` for online experiment tracking |
| 5 | Run `python job.py --data_dir ./data` (optionally `--max_steps N`) |

## References

- [Qwen3-VL repo](https://github.com/QwenLM/Qwen3-VL) — SFT scripts and configs (e.g. `qwen-vl-finetune/scripts/sft.sh`)
- [PubMedVision](https://huggingface.co/datasets/FreedomIntelligence/PubMedVision) — medical VQA dataset
- [NVIDIA FLARE](https://nvflare.readthedocs.io/) — federated learning framework
