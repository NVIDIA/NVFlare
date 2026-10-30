# VLA fine-tuning with native NVIDIA FLARE

This version replaces decent-vla's standalone FL simulator with FLARE's
`FedAvg` and PyTorch `FedOpt` workflows. Policy adapters, data partitioning,
local optimizer settings and evaluation remain available. There is one
federated execution path: a FLARE job, run in its simulator or submitted to a
POC/provisioned federation. `fl.mode: local` is accepted only as a migration
alias for `nvflare`; it does not invoke an internal simulator.

For author validation, start with [AUTHOR_COMPARISON.md](AUTHOR_COMPARISON.md):
comparison commands, deterministic controls, GPU/thread settings and the complete
state checkpoints needed for equivalent sequential execution.

Read [NATIVE_HANDOFF.md](NATIVE_HANDOFF.md) for the migration map, deterministic
comparison commands, validation status and intentional lifecycle differences.
This folder is a complete project: install from the directory containing this
README and `pyproject.toml`. It needs no code or resources from its parent. Unit
tests live in `tests/unit_test/`; real simulator tests live in
`tests/integration_test/fast/`. See [tests/README.md](tests/README.md) for commands.

## Setup

**Hardware.** The paper's configs use 8 clients. Two clients share a 96 GB GPU wherever they fit, so most configs
run on one node with 4 such GPUs. A π0.5 full fine-tuning client needs a GPU of its own, so those three configs need
8 GPUs. The paper ran each client on its own 96 GB GH200. The smoke configs use 2 clients on 2 GPUs. The CPU smoke run
needs no GPU.

**SmolVLA and π0.5 environment** (Python 3.12, which lerobot requires), built with [uv](https://docs.astral.sh/uv/).
Install the PyTorch build that matches the CUDA driver first, so the editable install keeps it:

```bash
cd /path/to/decent-vla-project  # directory containing this README and pyproject.toml
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install -e ".[nvflare,lerobot,libero,peft,dev]"
decent-vla doctor                  # prints the torch, CUDA and plugin versions
```

π0.5 uses the PaliGemma tokenizer, which is gated: accept its terms at
[google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224), then run `hf auth login`.

**GR00T environments** (Python 3.10, which Isaac-GR00T requires). GR00T trains and evaluates in two separate
environments, because the LIBERO simulator needs package versions that the training stack cannot share:

```bash
bash scripts/setup_groot.sh        # .venv-groot: Isaac-GR00T at the pinned commit, decent-vla, NVFlare
bash scripts/setup_groot_eval.sh   # .venv-groot-eval: LIBERO, robosuite and MuJoCo for evaluation
```

Both scripts need `git`, `uv` and a CUDA 12.8 driver, and run on a GPU node; on aarch64 they also need `git-lfs`,
for the flash-attn and torchcodec wheels that Isaac-GR00T stores with Git LFS. `setup_groot.sh` clones Isaac-GR00T
into `third_party/Isaac-GR00T` and applies `scripts/isaac_gr00t_pyav.patch`, which adds PyAV decoding for the
AV1-encoded LIBERO videos (torchcodec would need a system FFmpeg). The evaluation environment's package list is for
aarch64 machines such as GH200 nodes.

**Rendering.** LIBERO renders off-screen: set `export MUJOCO_GL=egl` before any evaluation.

## Data Preparation

- **SmolVLA and π0.5** train on [HuggingFaceVLA/libero](https://huggingface.co/datasets/HuggingFaceVLA/libero), the
  40 LIBERO tasks in LeRobot format. lerobot downloads it into the Hugging Face cache (`$HF_HOME`) on first use.
- **GR00T** trains on the four LIBERO suites in GR00T's LeRobot format:

  ```bash
  bash scripts/prep_groot_libero.sh   # about 2 GB, into data/groot-libero/
  ```

  The script downloads the `IPEC-COMMUNITY/libero_*_no_noops_1.0.0_lerobot` datasets and prepares them as
  Isaac-GR00T's LIBERO example describes.
- **GR00T weights.** The GR00T configs load the model from the Hugging Face cache only (`local_files_only: true`),
  so download it and its backbone first; the backbone is gated: accept its terms on Hugging Face and log in with
  `hf auth login`. SmolVLA and π0.5 weights download on first use.

  ```bash
  hf download nvidia/GR00T-N1.7-3B
  hf download nvidia/Cosmos-Reason2-2B
  ```
- **Client partition.** Each run computes its partition from the config (`task_round_robin`, seed 42) and writes it
  to `outputs/<run>/partition_manifest.json`. `decent-vla partition --config <config> --output-dir <dir>` writes it
  without training.

## Run Instructions

**Train.** From this project directory, in the environment of the policy:

```bash
decent-vla train --config examples/pi05_libero_lora_nvflare.yaml
```

`decent-vla` builds the NVFlare job and runs it in the simulator, with client `i` on the GPU at position `i` of
`fl.nvflare.gpu`; a GPU listed twice holds two clients. A run resumes from its latest checkpoint when started again.
The configs:

| Config | Policy | Federated | Algorithm | Rounds | GPUs |
|---|---|---|---|---|---|
| `pi05_libero_fullft_nvflare.yaml` | π0.5 | full model | FedAvg | 100 | 8 |
| `pi05_libero_lora_nvflare.yaml` | π0.5 | LoRA | FedAvg | 100 | 4 |
| `pi05_libero_fullft_fedprox_nvflare.yaml` | π0.5 | full model | FedProx | 100 | 8 |
| `pi05_libero_fullft_fedadam_nvflare.yaml` | π0.5 | full model | FedAdam | 100 | 8 |
| `groot_libero_fullft_nvflare.yaml` | GR00T N1.7 | action head | FedAvg | 140 | 4 |
| `groot_libero_lora_nvflare.yaml` | GR00T N1.7 | action head, LoRA | FedAvg | 140 | 4 |
| `smolvla_libero_fullft_nvflare.yaml` | SmolVLA | full model | FedAvg | 160 | 4 |
| `smolvla_libero_lora_nvflare.yaml` | SmolVLA | LoRA | FedAvg | 160 | 4 |

Each config has a `_smoke.yaml` version: 2 clients on 2 GPUs, 2 rounds of 20 local steps. To check the NVFlare
setup without a GPU, run `bash scripts/smoke_nvflare.sh` (mock model, 2 clients, 2 rounds).

A run writes to `outputs/<config name>/`:

| Path | Content |
|---|---|
| `config.json` | the resolved run config |
| `partition_manifest.json` | the tasks and episodes of each client |
| `checkpoints/server_round_NNNN.pt` | the fp32 global model (and FedAdam's moments) every 10 rounds; a restarted run resumes from the latest one |
| `checkpoints/server_round_final.pt` | the final fp32 global model, written when the run finishes |
| `checkpoints/round_NNNN.pt` | the final global model in the policy's checkpoint format |
| `nvflare_workspace/` | the NVFlare simulator workspace, with the server and client logs |

Checkpoints are kept, not pruned. One `server_round_NNNN.pt` takes about 16 GB for π0.5 full fine-tuning (47 GB with
FedAdam), 0.5 GB for π0.5 LoRA, 1.5 GB for SmolVLA, 6.2 GB for the GR00T action head and 1.5 GB with LoRA, so a
full π0.5 run needs about 200 GB of disk (about 500 GB with FedAdam).

**Run on a federation.** With `fl.nvflare.env: export` in the config, `decent-vla train` writes an NVFlare job
folder instead of running the simulator. Submit it to a running federation (POC or provisioned):

```bash
decent-vla train --config <config with fl.nvflare.env: export>
nvflare job submit -j outputs/<config name>/nvflare_job/<config name>
```

- The job carries its run config, the files that it names (the lerobot training config and the held-out task spec,
  `data.held_out`), and the client script.
  Every site needs this folder installed (`pip install -e .`, see [Setup](#setup)) and its own copy of the data and
  model weights. Download them before starting the site, not inside the first run.
- Name the sites `c0` … `cN-1`, or set `fl.nvflare.site_prefix: site-` for sites `site-1` … `site-N`.
- A relative `fl.output_dir` is created inside each site's job folder. When the job ends, the FLARE server archives
  its job folder into its job store: get the checkpoints with `nvflare job download <job id>`, or set an absolute
  `fl.output_dir` to keep them on the server's disk (better for large models). A resubmitted job gets a new folder,
  so only a run with an absolute `fl.output_dir` resumes from its checkpoints.
- Relative data paths, such as the GR00T `dataset_paths`, resolve against the folder that each site sets in the
  `DECENT_VLA_WORKDIR` environment variable.
- Set `fl.round_timeout_s`. A round needs `fl.min_clients` updates (default: every selected client), and NVFlare
  stops the job if fewer sites than that stay connected. Once that many clients have answered, the server waits
  `fl.nvflare.wait_time_after_min_received` seconds (default 10) for the others. FLARE owns round dispatch,
  responses and timeouts. An unmet quorum aborts the run. Use all clients and fresh output directories for
  deterministic training comparisons; see [NATIVE_HANDOFF.md](NATIVE_HANDOFF.md) for resume limitations.
- `fl.nvflare.launch_external_process: true` runs each client's training in a subprocess.

**Evaluate SmolVLA and π0.5.** `scripts/eval_libero.sh` exports a checkpoint to a lerobot policy folder and runs
`lerobot-eval` on LIBERO with the paper's protocol (all four suites, 10 episodes per task, replanning every 10 steps):

```bash
export MUJOCO_GL=egl
bash scripts/eval_libero.sh examples/pi05_libero_lora_nvflare.yaml \
    outputs/pi05_libero_lora_nvflare/checkpoints/server_round_0100.pt
```

The success rates land in `outputs/<run>/eval/<checkpoint>/`. `SUITES` and `N_EPISODES` select a smaller
evaluation, for example `SUITES=libero_spatial N_EPISODES=1`.

**Evaluate GR00T.** In the GR00T evaluation environment:

```bash
source .venv-groot-eval/bin/activate
export MUJOCO_GL=egl PYTHONPATH=$PWD/third_party/Isaac-GR00T
decent-vla eval --config examples/groot_libero_lora_nvflare.yaml \
    --ckpt outputs/groot_libero_lora_nvflare/checkpoints/server_round_0140.pt
```

The success rates land in `outputs/<run>/eval/results.json`, which the next evaluation of the same config replaces:
copy it before evaluating another checkpoint.

To reproduce the paper's protocol, evaluate the checkpoints of rounds 20, 40, 60 and so on, and take the best.

## Expected Results

Results from our paper ([arXiv:2609.22973](https://arxiv.org/abs/2609.22973), Tables I, II and IV). LIBERO
success rate (%) over 400 episodes; each federated result is the best of the evaluations every 20 rounds. The configs
named below carry the paper's settings; we have not run them at full length on NVFlare.

**Federated versus centralized fine-tuning** (federated: FedAvg, full fine-tuning; GR00T: action head):

| Policy | Training | Config | Spatial | Object | Goal | Long | Overall |
|---|---|---|---|---|---|---|---|
| π0.5 | Centralized | | 97.0 | 99.0 | 97.0 | 95.0 | 97.00 |
| π0.5 | Federated | `pi05_libero_fullft_nvflare` | 92.0 | 100.0 | 99.0 | 99.0 | 97.50 |
| GR00T N1.7 | Centralized | | 97.0 | 97.0 | 99.0 | 94.0 | 96.75 |
| GR00T N1.7 | Federated | `groot_libero_fullft_nvflare` | 91.0 | 99.0 | 95.0 | 79.0 | 91.00 |
| SmolVLA | Centralized | | 89.0 | 97.0 | 95.0 | 79.0 | 90.00 |
| SmolVLA | Federated | `smolvla_libero_fullft_nvflare` | 75.0 | 70.0 | 81.0 | 75.0 | 75.25 |

The centralized references fine-tune on all 40 tasks pooled: SmolVLA with decent-vla and one client, GR00T with
Isaac-GR00T's fine-tuning script; for π0.5, the reference is the public LIBERO checkpoint, evaluated with 50
episodes per task.

**Full fine-tuning versus LoRA** (FedAvg; payload is the update each client sends per round):

| Policy | Federated | Config | Parameters | Payload | Success |
|---|---|---|---|---|---|
| π0.5 | full model | `pi05_libero_fullft_nvflare` | 3.62B | 7.24 GB | 97.50 |
| π0.5 | LoRA | `pi05_libero_lora_nvflare` | 0.12B | 0.24 GB | 97.50 |
| GR00T N1.7 | action head | `groot_libero_fullft_nvflare` | 1.62B | 3.24 GB | 91.00 |
| GR00T N1.7 | action head, LoRA | `groot_libero_lora_nvflare` | 0.39B | 0.77 GB | 87.00 |
| SmolVLA | full model | `smolvla_libero_fullft_nvflare` | 0.40B | 0.81 GB | 75.25 |
| SmolVLA | LoRA | `smolvla_libero_lora_nvflare` | 0.04B | 0.08 GB | 55.25 |

**Aggregation algorithms** (π0.5, full fine-tuning):

| Algorithm | Config | Spatial | Object | Goal | Long | Overall |
|---|---|---|---|---|---|---|
| FedAvg | `pi05_libero_fullft_nvflare` | 92.0 | 100.0 | 99.0 | 99.0 | 97.50 |
| FedAdam | `pi05_libero_fullft_fedadam_nvflare` | 91.0 | 100.0 | 99.0 | 95.0 | 96.25 |
| FedProx | `pi05_libero_fullft_fedprox_nvflare` | 89.0 | 99.0 | 95.0 | 94.0 | 94.25 |

## License

The code in this folder is released under the Apache License 2.0. The third-party code, model weights and data it
uses carry their own terms:

| Component | Used for | License |
|---|---|---|
| [lerobot](https://github.com/huggingface/lerobot) | SmolVLA and π0.5 policies, datasets, `lerobot-eval` | Apache-2.0 |
| [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | GR00T N1.7 model code; `scripts/isaac_gr00t_pyav.patch` modifies one of its files | Apache-2.0 |
| [lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base) | SmolVLA weights | Apache-2.0 |
| [lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base) | π0.5 weights | Gemma Terms of Use |
| [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) | π0.5 tokenizer | Gemma Terms of Use (gated) |
| [nvidia/GR00T-N1.7-3B](https://huggingface.co/nvidia/GR00T-N1.7-3B) | GR00T N1.7 weights | NVIDIA Open Model License |
| [nvidia/Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) | GR00T N1.7 vision-language backbone | NVIDIA Open Model License (gated) |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | simulation benchmark | MIT |
| [HuggingFaceVLA/libero](https://huggingface.co/datasets/HuggingFaceVLA/libero) | SmolVLA and π0.5 training data | Apache-2.0 |
| [IPEC-COMMUNITY LIBERO datasets](https://huggingface.co/IPEC-COMMUNITY) | GR00T training data | not stated on Hugging Face; derived from LIBERO |

## Requirements

- NVFlare 2.8.1 (`nvflare==2.8.1`, pinned in `pyproject.toml`).
- Python 3.12 for SmolVLA and π0.5; Python 3.10 for GR00T.
- PyTorch 2.7 to 2.10 with a CUDA 12 build.
- lerobot at commit `8515d456` and Isaac-GR00T at commit `65cc4a19`, both pinned.
- NVIDIA GPUs with 96 GB: 4 for most configs, 8 for π0.5 full fine-tuning (see [Setup](#setup)).

## Citation

```bibtex
@misc{duan2026decentvla,
  title         = {An Empirical Study and Open Testbed for Federated Fine-Tuning of Vision-Language-Action Models},
  author        = {Duan, Zhekai and Xie, Kevin Ziyang and Tan, Xinyu and Geng, Shikai and Zhou, Chengxu and
                   Kompella, Ramana and Liu, Gaowen and Lu, Chris Xiaoxuan},
  year          = {2026},
  eprint        = {2609.22973},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}
```
