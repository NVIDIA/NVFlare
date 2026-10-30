# VLA fine-tuning with NVIDIA FLARE

Federated fine-tuning of SmolVLA, π0.5 and GR00T N1.7 on LIBERO using native
FLARE `FedAvg` / PyTorch `FedOpt` workflows and `PTFedProxLoss`. Clients retain
their local optimizer, scheduler and data iterator across rounds. FLARE owns
dispatch, aggregation and persistence; there is no internal FL simulator.

Like the other research examples, `job.py` selects a policy, generates its
configuration and exports a standard FLARE job. Generated YAML is editable;
the repository does not store a file for each algorithm/tuning/smoke combination.
See [AUTHOR_COMPARISON.md](AUTHOR_COMPARISON.md) for deterministic comparison
with the original implementation and GPU/sequential execution constraints.

## Project structure

The FLARE-facing files follow the layout used by the neighboring research examples:

```text
flare_native/
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
uv pip install -e '.[lerobot,libero,peft,dev]'
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
each client resident for curve alignment. Eight clients on two GPUs use:

```bash
nvflare simulator jobs/smolvla_libero_lora_nvflare \
  -w outputs/smolvla_workspace -c c0,c1,c2,c3,c4,c5,c6,c7 -t 8 \
  -gpu 0,1,0,1,0,1,0,1
```

Each GPU must fit all its assigned clients. Lowering threads or shortening the
GPU map can rotate client processes and lose warm training state. See
[AUTHOR_COMPARISON.md](AUTHOR_COMPARISON.md#clients-threads-and-gpus) before
reducing concurrency. A one-client, one-round job with the desired local-step
budget provides the centralized baseline through the same FLARE workflow.

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
20 rounds. These are paper results, not verified native-migration results;
GPU/full-length alignment remains for the authors. See the
[paper](https://arxiv.org/abs/2609.22973) for the complete result tables.

## Tests and contribution

Focused checks cover FLARE adapter numerics, persistence, data separation,
generated recipes, comparison failures and portable real simulator jobs:

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/unit_test -q
RUN_NATIVE_SIMULATOR_TESTS=1 python -m pytest tests/integration_test/fast -q
```

See [tests/README.md](tests/README.md). This folder is independent of the parent
implementation. After the authors confirm alignment, follow the
[final PR cleanup checklist](AUTHOR_COMPARISON.md#final-pr-cleanup-after-alignment)
to remove migration scaffolding and the original implementation, then move the
remaining native contents (including dotfiles) up into `research/decent-vla/`.
Regenerate configs and jobs in their final location.

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
