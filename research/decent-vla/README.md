# decent-vla: Federated Fine-Tuning of Vision-Language-Action Robot Policies

## Abstract

Vision-language-action (VLA) policies, such as SmolVLA, π0.5 and GR00T N1.7, map camera images and a language
instruction to robot actions. Fine-tuning them usually means pooling demonstrations from many robots, labs or
institutions in one place. decent-vla fine-tunes VLA policies with federated learning instead: each site trains on
its own tasks and data, and only model updates leave the site. This project hosts decent-vla with its NVFlare
runtime, and run configs that federate SmolVLA, π0.5 and GR00T N1.7 on the LIBERO benchmark with FedAvg, FedProx
and FedAdam, using the settings of our paper's experiments. In our paper, federated full fine-tuning of π0.5 across
8 clients that each hold different tasks reaches 97.50% LIBERO success, against 97.00% for centralized fine-tuning.

## Papers and Links

- Paper: Zhekai Duan\*, Kevin Ziyang Xie, Xinyu Tan, Shikai Geng, Chengxu Zhou, Ramana Kompella, Gaowen Liu and
  Chris Xiaoxuan Lu, "An Empirical Study and Open Testbed for Federated Fine-Tuning of Vision-Language-Action
  Models", [arXiv:2609.22973](https://arxiv.org/abs/2609.22973), 2026. (\* Corresponding author.)
- Tracking issue: [NVIDIA/NVFlare#5487](https://github.com/NVIDIA/NVFlare/issues/5487)
- Benchmark: [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
- Policies: [lerobot](https://github.com/huggingface/lerobot) (SmolVLA, π0.5) and
  [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) (GR00T N1.7)

## Objective

NVFlare has no example of federated fine-tuning for VLA policies. This project provides one on LIBERO, for three
VLA policies and three aggregation algorithms, with the settings of our paper, so readers can run its federated
experiments with NVFlare and compare new methods against its results. A later contribution adds a method for
sites whose robots have different action spaces.

## Method Summary

- **Setting:** horizontal FL with task-heterogeneous clients. LIBERO's 40 tasks, in four suites (Spatial, Object,
  Goal and Long), are split across 8 clients by a deterministic round-robin, so each client trains on 5 tasks that
  no other client sees.
- **Policies:** SmolVLA (0.45B parameters) and π0.5 (3.2B) through lerobot, and GR00T N1.7 (3.1B) through
  Isaac-GR00T, each behind decent-vla's `Backend` interface.
- **What is federated:** full fine-tuning or LoRA (rank 64, alpha 128) of the whole model for SmolVLA and π0.5; for
  GR00T, the action head, trained in full or with LoRA, while the vision-language backbone stays frozen.
- **Aggregation:** FedAvg, FedProx (mu = 0.01) and FedAdam (server learning rate 1e-3), as decent-vla strategies.
- **Training budget:** 250 local AdamW steps per round at each policy's published fine-tuning hyperparameters;
  100 rounds for π0.5, 140 for GR00T and 160 for SmolVLA.
- **Evaluation:** LIBERO simulation of all 40 tasks, 10 episodes each (400 episodes). SmolVLA and π0.5 replan every
  10 steps, GR00T every 16. The paper evaluates the global model every 20 rounds and reports the best score.
- **NVFlare integration:** a custom controller, `DecentVLAController` (a `ModelController`), runs the rounds and
  applies the aggregation strategy on the server. Each client runs a Client API script under `ScriptRunner`; the
  script keeps its policy, optimizer state and data loader in memory across rounds, trains for the configured local
  steps and returns its update. `decent-vla` builds the job with `FedJob` and runs it in the NVFlare simulator, each
  client on the GPU that the config assigns it.

## Repository Layout

```text
decent-vla/
|-- README.md
|-- pyproject.toml                 # package, extras (nvflare, lerobot, libero, peft) and plugin entry points
|-- src/decent_vla/
|   |-- core/                      # Backend, Strategy and Communicator protocols, LoRA, partitioning, results
|   |-- strategies/                # FedAvg, FedProx, FedAdam
|   |-- adapters/
|   |   |-- lerobot/               # SmolVLA, pi0, pi0.5 and ACT through lerobot; checkpoint export for lerobot-eval
|   |   |-- groot/                 # NVIDIA GR00T N1.7 through Isaac-GR00T
|   |   `-- _mock/                 # tiny CPU model for tests and smoke runs
|   |-- benchmarks/                # LIBERO simulation benchmark, mock benchmark
|   |-- runtime/                   # shared client and server round loop, checkpoints, local runner
|   |   `-- nvflare/               # NVFlare runtime: controller, Client API script, FedJob, codec
|   |-- configs/                   # run-config schema and loader
|   `-- cli/                       # `decent-vla` command line
|-- examples/                      # run configs: the paper's runs on NVFlare, their smoke versions, mock runs
|-- configs/libero_heldout_8.json  # held-out LIBERO task split
|-- scripts/                       # environment setup, data preparation, LIBERO evaluation, CPU smoke run
`-- tests/                         # unit tests, run in the SmolVLA and π0.5 environment; NVFlare is faked
```

- `examples/<policy>_libero_<recipe>[_<algorithm>]_nvflare.yaml` is one run of the paper: policy, what is federated,
  aggregation algorithm, data partition, budget and NVFlare placement in one file. Each has a `_smoke.yaml`
  version with 2 clients, 2 rounds and 20 local steps. The `*_train_pipeline*.yaml` files are the lerobot training
  settings that the SmolVLA and π0.5 configs point to.
- `decent-vla train --config <config>` runs a config; `scripts/eval_libero.sh` and `decent-vla eval` evaluate its
  checkpoints (see [Run Instructions](#run-instructions)).
- `scripts/setup_groot.sh` and `scripts/setup_groot_eval.sh` build the GR00T training and evaluation environments;
  `scripts/prep_groot_libero.sh` prepares GR00T's LIBERO data; `scripts/smoke_nvflare.sh` runs a 2-round federation
  of the mock model on CPU.

## Setup

**Hardware.** The paper's configs use 8 clients. Two clients share a 96 GB GPU wherever they fit, so most configs
run on one node with 4 such GPUs. A π0.5 full fine-tuning client needs a GPU of its own, so those three configs need
8 GPUs. The paper ran each client on its own 96 GB GH200. The smoke configs use 2 clients on 2 GPUs. The CPU smoke run
needs no GPU.

**SmolVLA and π0.5 environment** (Python 3.12, which lerobot requires), built with [uv](https://docs.astral.sh/uv/).
Install the PyTorch build that matches the CUDA driver first, so the editable install keeps it:

```bash
cd research/decent-vla
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

**Train.** From `research/decent-vla`, in the environment of the policy:

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

## Validation performed

Short runs of every configuration, following this README from environment setup to LIBERO evaluation: federated
training through the NVFlare simulator, aggregation, checkpointing and evaluation, for each policy, recipe and
algorithm.

- **Revision:** the commit that adds this folder. Two late changes came after some of the checks: `run_eval` now
  refuses a checkpoint path that does not exist, and unused packages were removed from the GR00T evaluation package
  list. Federated training and the SmolVLA and π0.5 evaluations ran before both changes and use neither; the unit
  tests and the mock federation ran after the first and do not use the second; the GR00T evaluation ran after both.
- **Environment:** Isambard-AI, NVIDIA GH200 120GB nodes (aarch64), driver 580.178.04, one node per run.
  - SmolVLA and π0.5 environment: Python 3.12.13, torch 2.7.1+cu128, lerobot 0.5.2 (commit `8515d456`),
    NVFlare 2.8.1, transformers 5.5.4, peft 0.21.2.
  - GR00T training environment: Python 3.10.20, torch 2.7.1+cu128, Isaac-GR00T `65cc4a19` with
    `scripts/isaac_gr00t_pyav.patch`, NVFlare 2.8.1, transformers 4.57.3, peft 0.17.1.
  - GR00T evaluation environment: Python 3.10.20, torch 2.7.1+cu128, robosuite 1.4.0, MuJoCo 3.8.1.
- **Commands and results:**

| Check | Command | Result |
|---|---|---|
| Unit tests | `python -m pytest tests --basetemp=/tmp/dvla-pytest` (SmolVLA and π0.5 environment) | 391 passed; one cache test needs sub-second file times, so its temporary files go on local disk, not Lustre |
| Mock federation on CPU | `bash scripts/smoke_nvflare.sh` | passed: 2 clients, 2 rounds, checkpoint written and evaluated |
| Federated training | `decent-vla train --config examples/<config>_smoke.yaml` | all 8 configs completed; see below |
| LIBERO evaluation, SmolVLA and π0.5 | `SUITES=libero_spatial N_EPISODES=1 bash scripts/eval_libero.sh <config> <checkpoint>` | ran end to end: 10 episodes each, 0% success (SmolVLA 208 s, π0.5 453 s) |
| LIBERO evaluation, GR00T | `decent-vla eval --config examples/groot_libero_lora_nvflare_smoke.yaml --ckpt <checkpoint>` | ran end to end: 10 episodes, 0% success (242 s) |
| Missing checkpoint, GR00T | the same command with a `--ckpt` file that does not exist | refused with `FileNotFoundError`; nothing written |

Each smoke run federates 2 clients (20 LIBERO tasks each) for 2 rounds of 20 local steps. In every run both clients
returned an update in both rounds, and the server wrote `server_round_0001.pt`, `server_round_0002.pt`,
`server_round_final.pt` and the exported `round_0002.pt`:

| Config | Wall time (s) | Mean client loss, round 1 | Mean client loss, round 2 |
|---|---|---|---|
| `smolvla_libero_fullft_nvflare_smoke` | 131 | 3.1272 | 2.8850 |
| `smolvla_libero_lora_nvflare_smoke` | 128 | 3.0318 | 2.9020 |
| `pi05_libero_fullft_nvflare_smoke` | 782 | 0.8291 | 0.8186 |
| `pi05_libero_lora_nvflare_smoke` | 607 | 0.8104 | 0.7785 |
| `pi05_libero_fullft_fedprox_nvflare_smoke` | 835 | 0.8350 | 0.8463 |
| `pi05_libero_fullft_fedadam_nvflare_smoke` | 832 | 0.8237 | 0.8278 |
| `groot_libero_fullft_nvflare_smoke` | 760 | 1.2556 | 1.2611 |
| `groot_libero_lora_nvflare_smoke` | 1106 | 1.2583 | 1.2630 |

The evaluations ran `libero_spatial` with one episode per task (10 episodes) on the LoRA smoke checkpoints of
round 2; after 40 local steps per client the policies do not yet solve the tasks, so these runs check that export,
policy loading, simulation and scoring work, not the success rate.

- **Limitations:** the runs exercise the NVFlare simulator only; the code has not been tested outside it. They use
  2 clients and 2 rounds; the full-scale numbers in [Expected Results](#expected-results) are from our paper.

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
