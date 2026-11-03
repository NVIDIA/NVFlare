# VLA fine-tuning with NVIDIA FLARE

Federated fine-tuning of SmolVLA, π0.5 and GR00T N1.7 on LIBERO using native
FLARE `FedAvg` / PyTorch `FedOpt` workflows and `PTFedProxLoss`. Clients retain
their local optimizer, scheduler and data iterator across rounds. FLARE owns
dispatch, aggregation and persistence; there is no internal FL simulator.

Like the other research examples, `job.py` selects a policy, generates its
configuration and exports a standard FLARE job. Generated YAML is editable;
the repository does not store a file for each algorithm/tuning/smoke combination.

## Project structure

The FLARE-facing files follow the layout used by the neighboring research examples:

```text
decent-vla/
├── job.py                     # compose and export the FLARE job
├── client.py                  # FLARE Client API training entry point
├── evaluate.py
└── src/
    ├── workflows.py           # native FedAvg and FedOpt workflow adapters
    ├── persistor.py           # FLARE LearnablePersistor
    ├── adapters/              # policy-specific implementations
    ├── configs/               # run configuration
    └── support/               # partitioning, training, checkpoints and evaluation support
```

## Setup and data

Run from the directory containing this README. The package requires FLARE
2.8.1 and Torch >=2.4,<2.11. For SmolVLA/π0.5, use Python 3.12 and install the
Torch build appropriate for your CUDA driver before the policy dependencies:

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install -e '.[lerobot,libero,peft]'
```

The LeRobot revision is pinned in `pyproject.toml`. Data comes from
[HuggingFaceVLA/libero](https://huggingface.co/datasets/HuggingFaceVLA/libero).
For π0.5, accept the gated [PaliGemma tokenizer](https://huggingface.co/google/paligemma-3b-pt-224)
terms and run `hf auth login`. Each site reconstructs its own disjoint share
of the cached dataset; the partition is recorded in `partition_manifest.json`.

GR00T uses a separate Python 3.10 environment and the pinned Isaac-GR00T
revision, with PyAV support for LIBERO's AV1 videos:

```bash
bash scripts/setup_groot.sh
bash scripts/setup_groot_eval.sh
bash scripts/prep_groot_libero.sh
hf download nvidia/GR00T-N1.7-3B
hf download nvidia/Cosmos-Reason2-2B
```

The scripts need `git`, `uv`, a CUDA 12.8 driver, and `git-lfs` on aarch64.
Accept the gated Cosmos backbone terms before downloading. Evaluation uses
`.venv-groot-eval`; training uses `.venv-groot`. Download/cache weights and data
on each site before launching a federation.

## Export and run with FLARE

`job.py` composes the experiment and exports it; execution uses FLARE commands:

```bash
python job.py
nvflare simulator jobs/mock_fl_nvflare -w outputs/mock_workspace -c c0,c1 -t 2

# Real policy, two clients, two rounds; cache weights/data first.
export DECENT_VLA_WORKDIR="$PWD"
python job.py --policy pi05 --tuning lora --smoke
nvflare simulator jobs/pi05_libero_lora_nvflare_smoke \
  -w outputs/pi05_workspace -c c0,c1 -t 2 -gpu 0,1

# Other full recipes; all export without loading the policy.
python job.py --policy pi05 --algorithm fedadam
python job.py --policy smolvla --tuning lora --num-clients 8
python job.py --policy groot --tuning fullft
```

The script writes `outputs/<name>/run.yaml`, LeRobot's `train_pipeline.yaml`
where needed, and `jobs/<name>/`. Use `--output-dir` for the run folder,
`--job-dir` for the export location, and `python job.py --help` for training
overrides. After editing a generated YAML, re-export it with
`python job.py --config outputs/<name>/run.yaml`.

| Policy | Full run | Batch size, full / LoRA | Reference GPU count, full / LoRA |
| --- | --- | --- | --- |
| π0.5 | 100 rounds × 250 steps | 32 / 32 | 8 / 4 |
| SmolVLA | 160 rounds × 250 steps | 32 / 64 | 4 / 4 |
| GR00T action head | 140 rounds × 250 steps | 16 / 32 | 4 / 4 |

Full recipes use eight clients. `--smoke` uses two clients, two rounds and
20 local steps while preserving optimizer/preprocessing settings. The mock
defaults to a small CPU job. FedProx uses `mu=0.01` (sum for VLA recipes);
FedAdam uses server LR/epsilon `0.001` and betas `(0.9, 0.999)`.

Set simulator workspaces, client names, threads and GPUs through
`nvflare simulator`. Match the logical client count and keep
each client resident, so it keeps its optimizer, scheduler and data iterator
between rounds. Eight clients on two GPUs use:

```bash
nvflare simulator jobs/smolvla_libero_lora_nvflare \
  -w outputs/smolvla_workspace -c c0,c1,c2,c3,c4,c5,c6,c7 -t 8 \
  -gpu 0,1,0,1,0,1,0,1
```

`-gpu` takes a group map, not a pool of GPUs. With several groups, FLARE 2.8.1
runs one worker per group and overrides `-t`; with one group, `-t` sets the
number of workers, so always set it.

| FLARE 2.8.1 options, eight clients | Effect |
| --- | --- |
| `-t 8`, no GPU map | Eight resident workers; set `device: cpu` for CPU execution |
| `-t 8 -gpu 0` | Eight resident clients on one GPU, if memory fits |
| `-gpu 0,1,0,1,0,1,0,1` | Eight workers, four per GPU |
| `-gpu 0,1` | Two workers; clients rotate through them and lose warm state |
| `-gpu '[0,1]'` | One group exposing two GPUs; no model parallelism |

Each GPU must fit the models, optimizer state and buffers of all its assigned
clients, and the host needs RAM for aggregation and the final export.
`fl.min_clients` sets the updates required per round, not concurrency; do not
lower it to fit GPU memory. For more than 100 clients, use `-m N` with a
matching `-c` list.

With fewer workers than clients, the simulator swaps client processes out
between turns. `fl.persist_client_state` saves weights, optimizer and
scheduler, but not RNG or data-iterator position, so swapped clients do not
train as resident ones do. Disk-backed client state through FLARE's
`SWAP_OUT`/`SWAP_IN` events is not implemented. A one-client, one-round job
with the desired local-step budget provides the centralized baseline through
the same FLARE workflow.

Exported jobs include the native `src/` tree. Install the policy dependencies
and NVFlare on every site, then submit the same job:

```bash
nvflare job submit -j jobs/<name>
```

Sites are `c0`…`cN-1`, or set `fl.nvflare.site_prefix: site-` for provisioned
`site-1`…`site-N`. Jobs include their configs and client script. Relative data
paths resolve against each site's `DECENT_VLA_WORKDIR` when set. Relative output
paths live under each site's job run folder. For the default CPU simulator run,
server artifacts are under `outputs/mock_workspace/server/simulate_job/outputs/mock_fl_nvflare/`.
Use an absolute `fl.output_dir` to choose a fixed site-local output folder.
Download deployed results with `nvflare job download <job-id>`.

A completed run writes `config.json`, data/scope manifests, `training_curve.json`,
periodic `checkpoints/server_round_NNNN.pt` and final `server_round_final.pt`.
Check the final artifact and logs: FLARE
2.8.1 can return zero after a workflow error. Reusing outputs auto-resumes from
periodic FLARE checkpoints unless `DECENT_VLA_NO_RESUME=1` is set. Set
`fl.round_timeout_s` for deployment; an unmet quorum aborts through FLARE.

## Explicit repository assignment

For LeRobot datasets already organized by site, use `explicit_repos` in the
generated run YAML instead of splitting one dataset into tasks or episodes:

```yaml
backend:
  name: lerobot
  # Keep the generated model, optimizer and train_config_path settings.
data:
  partitioning: explicit_repos
  num_clients: 2
  client_repo_ids:
    - [org/client0-task-a, org/client0-task-b]  # c0 trains on both datasets
    - [org/client1-task-c]                    # c1 trains on this dataset
eval:
  personalized: false
  # Keep the generated LIBERO benchmark settings.
```

Use actual LeRobot dataset repository IDs in place of the example IDs. Each
entry in `client_repo_ids` must be a non-empty list of non-empty strings;
entry `i` belongs to client `ci`. `num_clients` must equal the number of lists,
or omit it to derive the count. Re-export the edited YAML with `job.py --config`.

Each client trains on all eligible frames from its assigned repositories,
concatenated locally. Assign non-overlapping data and exclude held-out tasks
from those repositories yourself: this mode does not split episodes, apply
the held-out task filter, or detect overlap between repositories. All sites
need access to the repository metadata used for shared normalization statistics.
Aggregation weights use each client's eligible frame count.

This mode requires `backend.name: lerobot` and `eval.personalized: false`.
It supports global checkpoint evaluation. Personalized evaluation requires
client task names, which the repository-only manifest does not record; use
`task_round_robin` or `explicit_tasks` partitioning if you need that feature.

## Evaluate

```bash
export MUJOCO_GL=egl
bash scripts/eval_libero.sh outputs/pi05_libero_lora_nvflare/run.yaml \
  /absolute/path/to/run/checkpoints/server_round_0100.pt

# In .venv-groot-eval, with Isaac-GR00T on PYTHONPATH:
python evaluate.py --config outputs/groot_libero_lora_nvflare/run.yaml \
  --ckpt /absolute/path/to/run/checkpoints/server_round_0140.pt
```

LeRobot evaluation uses all four LIBERO suites, ten episodes/task and replanning
every ten steps; `SUITES` and `N_EPISODES` select smaller checks. Results live
under the run's `eval/`. GR00T evaluation replaces `eval/results.json`; copy it
before evaluating another checkpoint.

The paper reports full-tuning FedAvg overall success of 97.50% (π0.5), 75.25%
(SmolVLA) and 91.00% (GR00T), selecting the best checkpoint evaluated every
20 rounds. These numbers are from our paper; this example has not been run at
full length (see [Validation performed](#validation-performed)). See the
[paper](https://arxiv.org/abs/2609.22973) for the complete result tables.

## Validation performed

Short runs of every policy, tuning mode and algorithm in the NVFlare simulator, from job export to LIBERO
evaluation, all on the code in this folder.

- **Environment:** Isambard-AI, NVIDIA GH200 120GB nodes (aarch64), driver 580.178.04; two GPUs per training run,
  one per evaluation.
  - SmolVLA and π0.5: Python 3.12.13, torch 2.7.1+cu128, lerobot 0.5.2 (commit `8515d456`), NVFlare 2.8.1,
    transformers 5.5.4, peft 0.21.2.
  - GR00T training: Python 3.10.20, torch 2.7.1+cu128, Isaac-GR00T `65cc4a19` with `scripts/isaac_gr00t_pyav.patch`,
    NVFlare 2.8.1, transformers 4.57.3, peft 0.17.1.
  - GR00T evaluation: Python 3.10.20, torch 2.7.1+cu128, robosuite 1.4.0, MuJoCo 3.8.1.
- **Commands and results:**

| Check | Command | Result |
| --- | --- | --- |
| CPU mock | `python job.py` (and `--algorithm fedprox`), then `nvflare simulator jobs/<name> -w <workspace> -c c0,c1 -t 2` | FedAvg and FedProx completed: 2 clients, 2 rounds, final checkpoint written |
| Federated training | `python job.py --policy <policy> --tuning <tuning> --algorithm <algorithm> --smoke`, then `nvflare simulator jobs/<name> -w <workspace> -c c0,c1 -t 2 -gpu 0,1` | all 8 completed; see below |
| LIBERO evaluation, SmolVLA and π0.5 | `SUITES=libero_spatial N_EPISODES=1 bash scripts/eval_libero.sh <run.yaml> <checkpoints/server_round_0002.pt>` | ran end to end: 10 episodes each, 0% success (SmolVLA 192 s, π0.5 452 s) |
| LIBERO evaluation, GR00T | `python evaluate.py --config <run.yaml> --ckpt <checkpoints/server_round_0002.pt>` | ran end to end: 10 episodes, 0% success (228 s) |

Each smoke run federates 2 clients for 2 rounds of 20 local steps. In every run both clients returned an update in
both rounds, and the server wrote `server_round_0001.pt`, `server_round_0002.pt` and `server_round_final.pt`:

| Policy | Tuning | Algorithm | Wall time (s) | Mean client loss, round 1 | Mean client loss, round 2 |
| --- | --- | --- | --- | --- | --- |
| SmolVLA | full | FedAvg | 103 | 3.0374 | 2.7010 |
| SmolVLA | LoRA | FedAvg | 127 | 3.0561 | 2.8543 |
| π0.5 | full | FedAvg | 643 | 0.8404 | 0.8053 |
| π0.5 | LoRA | FedAvg | 469 | 0.8372 | 0.8081 |
| π0.5 | full | FedProx | 700 | 0.8404 | 0.8055 |
| π0.5 | full | FedAdam | 688 | 0.8404 | 0.8054 |
| GR00T action head | full | FedAvg | 698 | 1.2588 | 1.2493 |
| GR00T action head | LoRA | FedAvg | 987 | 1.2636 | 1.2601 |

The three π0.5 full runs share round 1 because the seeds are fixed. The evaluations ran `libero_spatial` with one
episode per task on the round-2 LoRA checkpoints; after 40 local steps per client the policies do not yet solve the
tasks, so these runs check export, policy loading, simulation and scoring, not the success rate.

- **Limitations:** the runs use 2 clients, 2 rounds and 20 local steps, in the NVFlare simulator only. The full
  recipes have not been run with this example; the success rates under [Evaluate](#evaluate) are from our paper.

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
