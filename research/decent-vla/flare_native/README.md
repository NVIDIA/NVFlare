# VLA fine-tuning with NVIDIA FLARE

Federated fine-tuning of SmolVLA, π0.5 and GR00T N1.7 on LIBERO using native
FLARE `FedAvg` / PyTorch `FedOpt` workflows and `PTFedProxLoss`. Clients retain
their local optimizer, scheduler and data iterator across rounds. FLARE owns
dispatch, aggregation and persistence; there is no internal FL simulator.

Like the other research examples, `job.py` selects a policy and generates its
configuration before running or exporting the job. Generated YAML is editable;
the repository does not store a file for each algorithm/tuning/smoke combination.
See [AUTHOR_COMPARISON.md](AUTHOR_COMPARISON.md) for deterministic comparison
with the original implementation and GPU/sequential execution constraints.

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

## Run or export

```bash
python job.py                                      # CPU mock, 2 clients, 2 rounds
python job.py --policy pi05 --tuning lora --smoke    # real policy, short run
python job.py --policy pi05 --algorithm fedadam     # full paper recipe
python job.py --policy smolvla --tuning lora --num-clients 8 --gpus 0,1
python job.py --policy groot --tuning fullft --generate-only
python job.py --policy pi05 --tuning lora --export
```

`--generate-only` writes `outputs/<name>/run.yaml` and, for LeRobot policies,
`train_pipeline.yaml`. `--export` writes a standard FLARE job under
`outputs/<name>/nvflare_job/<name>`. Otherwise the job runs in the simulator.
Use `--output-dir` for a different run folder and `python job.py --help` for
client/round/step/batch/worker overrides. Edit generated YAML for further
backend or algorithm options, then use `decent-vla train --config <run.yaml>`.

| Policy | Full run | Batch size, full / LoRA | Default GPUs, full / LoRA |
| --- | --- | --- | --- |
| π0.5 | 100 rounds × 250 steps | 32 / 32 | 8 / 4 |
| SmolVLA | 160 rounds × 250 steps | 32 / 64 | 4 / 4 |
| GR00T action head | 140 rounds × 250 steps | 16 / 32 | 4 / 4 |

Full recipes use eight clients. `--smoke` uses two clients, two rounds and
20 local steps while preserving policy-specific optimizer/preprocessing settings.
The mock always defaults to a small CPU run. FedProx uses `mu=0.01` (sum for
VLA recipes); FedAdam uses server LR/epsilon `0.001` and betas `(0.9, 0.999)`.
GPUs must have enough memory for every assigned resident client. `--gpus 0,1`
expands the device pool to one map entry per client; it does not limit the run
to two concurrent clients. See the comparison guide before reducing concurrency.

A run records its resolved config, data/scope manifests, `training_curve.json`,
periodic `checkpoints/server_round_NNNN.pt`, final `server_round_final.pt`,
the backend export `round_NNNN.pt`, and the simulator workspace. A repeated run
auto-resumes from periodic checkpoints unless `DECENT_VLA_NO_RESUME=1` is set.
Checkpoints can be tens of GB, especially FedAdam weights plus moments.

For deployment, install this package on every site and submit the exported job:

```bash
nvflare job submit -j outputs/<name>/nvflare_job/<name>
```

Sites are `c0`…`cN-1`, or use `fl.nvflare.site_prefix: site-` for `site-1`…`site-N`.
The job includes its configs and client script. Relative data paths resolve
against each site's `DECENT_VLA_WORKDIR`. Relative output paths live inside
the site's job folder; download results with `nvflare job download <job-id>`.
Use an absolute server output path for resume across submissions. Set
`fl.round_timeout_s` appropriately; an unmet quorum aborts through FLARE.

## Evaluate

```bash
export MUJOCO_GL=egl
bash scripts/eval_libero.sh outputs/pi05_libero_lora_nvflare/run.yaml \
  outputs/pi05_libero_lora_nvflare/checkpoints/server_round_0100.pt

# In .venv-groot-eval, with Isaac-GR00T on PYTHONPATH:
decent-vla eval --config outputs/groot_libero_lora_nvflare/run.yaml \
  --ckpt outputs/groot_libero_lora_nvflare/checkpoints/server_round_0140.pt
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
implementation. After the authors confirm alignment, retain its entire contents
(including dotfiles), remove the other original files and move these contents
up into `research/decent-vla/`. No source edits are needed; regenerate configs
in their final location.

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
