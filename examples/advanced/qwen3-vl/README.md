# Federated Multi-modal Fine-Tuning with Qwen3-VL

This example shows how to fine-tune [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) (vision-language) in a federated setting using [NVIDIA FLARE](https://nvflare.readthedocs.io/), with the [PubMedVision](https://huggingface.co/datasets/FreedomIntelligence/PubMedVision) medical VQA dataset split across 3 clients.

## Code structure

As in typical NVFlare examples (e.g. [hello-pt](../../hello-world/hello-pt/)):

| File | Role |
|------|------|
| `model.py` | Re-exports the Qwen2.5-VL wrapper used as the FL model (`Qwen2VLModelWrapper`) |
| `model_wrapper.py` | Defines the wrapper so the server can save/load `state_dict` |
| `client.py` | Client training script: receives global model, trains on site data, sends update |
| `job.py` | FedAvg recipe: 3 clients, per-site data paths, TensorBoard tracking |
| `prepare_data.py` | Splits PubMedVision into `site-1`, `site-2`, `site-3` shards |

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended: 18GB+ VRAM for the 3B model)
- [Git LFS](https://git-lfs.com/) (for cloning dataset repos)

## 1. Virtual environment and dependencies

Create and activate a virtual environment, then install NVFlare and this example’s dependencies (including PyTorch and `flash_attn`):

```bash
cd examples/advanced/qwen3-vl

python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS

# Install NVFlare and Qwen3-VL requirements
./install_requirements.sh
```

The script installs PyTorch first (needed to build `flash_attn`), then the rest of `requirements.txt`. If you prefer to install from the NVFlare repo root, you can pass the path to this example’s `requirements.txt` and run the same steps there.

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

### Option A: Clone via Git (recommended for full dataset)

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
    unzip -j images_$i.zip -d images/ & # wait patiently, it takes a while...
done
cd ..
```

Then use the JSON file for the Instruction Tuning VQA split, e.g.:

- `PubMedVision/PubMedVision_InstructionTuning_VQA.json`  
  (or the path where that file lives in your clone.)


### Option B: Load from HuggingFace Hub

You can skip cloning and load a subset by name in the prepare step (see below).

### Split data for 3 federated clients

Run the preparation script so each client has a non-overlapping shard under `./data/site-1`, `site-2`, `site-3`:

**Using your local JSON file (e.g. from the cloned PubMedVision repo):**

If the JSON is in the current directory:

```bash
python prepare_data.py --data_file PubMedVision_InstructionTuning_VQA.json --output_dir ./data
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

## 4. Run the federated job

With the venv activated, `QWEN3VL_ROOT` set, and data under `./data`, start the NVFlare job (exact command may depend on your `job.py` and how it invokes the Qwen3-VL SFT scripts):

```bash
export QWEN3VL_ROOT=/path/to/Qwen3-VL   # if not already set
python job.py --data_dir ./data
```

Adjust `--data_dir` (or equivalent) to point at the same `./data` directory used in the prepare step. The job will run the Qwen3-VL SFT pipeline in a federated way across the 3 sites using the split PubMedVision data.

## Summary

| Step | Action |
|------|--------|
| 1 | Create venv, `pip install nvflare`, run `./install_requirements.sh` |
| 2 | `git clone https://github.com/QwenLM/Qwen3-VL.git` and set `QWEN3VL_ROOT` |
| 3 | Download PubMedVision (clone or Hub) and run `prepare_data.py` to get `./data/site-{1,2,3}/` |
| 4 | Run `python job.py` (with `--data_dir ./data` or as configured) |

## References

- [Qwen3-VL repo](https://github.com/QwenLM/Qwen3-VL) — SFT scripts and configs (e.g. `qwen-vl-finetune/scripts/sft.sh`)
- [PubMedVision](https://huggingface.co/datasets/FreedomIntelligence/PubMedVision) — medical VQA dataset
- [NVIDIA FLARE](https://nvflare.readthedocs.io/) — federated learning framework
