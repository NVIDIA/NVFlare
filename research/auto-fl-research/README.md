# Auto-FL-Research with NVFlare

This bundle is a practical starting point for an **autoresearch-style** Auto-FL loop on top of NVFlare.

## Example progress

The plot below is an example result from using this harness with Claude Code, model: Opus 4.7 (1M context),
effort level: max.

![Example Auto-FL progress plot](assets/example_progress.png)

It is designed to combine:
- NVFlare's **Client API + Recipe API** patterns for `client.py`, `FedAvgRecipe`, `SimEnv`, TensorBoard tracking, and optional cross-site evaluation;
- the CIFAR-10 simulation path for **diff-based uploads** and **custom aggregators**; and
- the **`program.md`-centered control-plane** style popularized by [karpathy/autoresearch](https://github.com/karpathy/autoresearch), where the human primarily evolves the research instructions and the agent iterates on a bounded code surface.

## What is included

- `program.md` — the main agent control plane
- `AGENTS.md`, `CLAUDE.md` — thin repository guardrails that point back to `program.md`
- `job.py` — merged NVFlare baseline recipe
- `client.py` — merged client with DIFF updates and `flare.is_evaluate()` support
- `custom_aggregators.py` — FedAvg, FedOpt-style, SCAFFOLD, and median aggregators
- `model.py`, `train_utils.py`, `data/` — minimal supporting code
- `mutation_schema.yaml` — bounded mutation surface for the current harness
- `scripts/init_run.sh` — creates an autoresearch branch and initializes `results.tsv`
- `scripts/run_iteration.sh` — runs one candidate mutation with log redirection and score extraction
- `scripts/finalize_batch_status.py` — promotes reviewed candidates to `keep` or demotes them to `discard`
- `scripts/extract_score.py` — extracts a comparable score from cross-site validation JSON
- `scripts/validate_contract.py` — static contract checks
- `templates/` — result logging templates
- `skills/autofl-nvflare-report/` — post-run reporting skill for stopped experiment branches
- `ACKNOWLEDGEMENTS.md` — provenance and attribution notes

## How this uses the autoresearch approach properly

The [autoresearch](https://github.com/karpathy/autoresearch) repo keeps the setup intentionally small and treats `program.md` as the agent-facing control plane. The core repo only has a few files that matter, with one main editable target and a fixed evaluation harness. This starter follows that spirit, but adapts it to NVFlare:

- **Primary control plane:** `program.md` is the first file the agent should read.
- **Bounded edit surface:** mutations should mostly target `client.py`, then `custom_aggregators.py`, then `job.py`; registered, parameter-capped architecture variants may also touch `model.py`.
- **Fixed evaluation budget:** compare candidates under the same recipe budget (`n_clients`, `num_rounds`, `aggregation_epochs`, `batch_size`, `eval_batch_size`, `alpha`, `seed`, `model_arch`, `max_model_params`).
- **Comparable metric extraction:** recommended runs enable cross-site evaluation and extract a single score from `cross_val_results.json`.
- **Run keep / discard loop:** on one local H100, the agent can launch several same-budget candidates concurrently when the 80 GB memory budget allows, then rank the completed batch against the ledger and keep, narrow, or discard.
- **Autonomous continuation:** after setup and baseline, the agent keeps running same-budget candidates until manually interrupted.
- **Literature-grounded recovery:** when progress stalls, the agent should use the Camyla-inspired literature loop in `program.md` to generate diverse paper queries, extract challenge cards, score compatible proposals, fill `templates/literature_loop.md`, and select the next contract-safe idea.
- **Tracked experiment ledger:** `results.tsv` is committed on experiment branches so the branch carries its run provenance.

> Note: This is not a literal clone of [karpathy/autoresearch](https://github.com/karpathy/autoresearch); it is an NVFlare-specific adaptation of the same operating model.

## QWBE-style literature proposals

QWBE is currently implemented as an **instruction and artifact workflow**, not as imported [Camyla](https://yifangao112.github.io/camyla-page) code or a separate tree-search scheduler. When progress stalls, `program.md` directs the agent to use the Camyla-inspired literature loop and fill `templates/literature_loop.md`.

The current flow is:

1. Generate source-backed proposal cards from recent `results.tsv` symptoms and relevant papers.
2. Filter out duplicates, known null/worse ideas, and proposals that violate the current contract.
3. Score each remaining proposal from 1-5 on expected gain, contract safety, simplicity, evidence, novelty, and runtime cost.
4. Compute:

   ```text
   2*expected_gain + 2*contract_safety + simplicity + evidence + novelty - runtime_cost
   ```

5. Rank the next compatible proposals with the scoring rubric and select a small batch of top candidates, up to the current `PARALLEL_CANDIDATES` width.
6. Launch the selected candidates with the normal `scripts/run_iteration.sh` mechanism, using unique `RUN_LOG` and `--name` values for each concurrent run on the same H100.
7. Wait for the batch to finish or time out, rank the completed runs, then finalize reviewed ledger rows so completed `candidate` rows become `keep` or `discard`.

This keeps the Camyla/QWBE idea inside the existing harness contract: no new dependencies, no evaluation changes, and no server-client protocol changes except explicitly labeled modes such as `--aggregator scaffold`. Architecture changes are allowed only as registered `--model_arch` variants under the active `--max_model_params` budget.

## Assumed usage

This starter is intended to live inside an NVFlare working tree or another repo where `nvflare`, `torch`, and `torchvision` are installed.
Install the dependencies before handing the repository to an agent; the agent should use that prepared Python environment rather than creating virtual environments or installing packages during the research loop.

A simple layout is:

```text
NVFlare/
  examples/
    advanced/
      cifar10/
        pt/
          autofl_nvflare/
            <this bundle>
```

## Quick start

Human preflight:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHON="$(python -c 'import sys; print(sys.executable)')"
"$PYTHON" --version
```

Pass the prepared interpreter to validation and runs:

```bash
PYTHON="$PYTHON" make validate
TAG="h100-baseline-$(date +%Y%m%d)" make init-run
PYTHON="$PYTHON" make smoke
```

Then give the folder to a coding agent and tell it to start with `program.md`. Include the exact prepared `PYTHON` path in the initial prompt so the agent can reuse the environment without spending tokens on dependency setup or brittle interpreter discovery.

## Recommended agent runtime

For long autonomous runs, prefer launching the coding agent inside a devcontainer rather than directly on the host. The recommended starting point is Trail of Bits' [`claude-code-devcontainer`](https://github.com/trailofbits/claude-code-devcontainer), which is designed to run Claude Code with broad command permissions inside a filesystem-isolated container.

Suggested flow:

```bash
# On the host, one-time setup from the Trail of Bits devcontainer README.
npm install -g @devcontainers/cli
git clone https://github.com/trailofbits/claude-code-devcontainer ~/.claude-devcontainer
~/.claude-devcontainer/install.sh self-install

# From this repository checkout.
devc .
```

Before starting the container shell, make sure `.devcontainer/devcontainer.json` exposes the H100 to Docker by including `--gpus=all` in `runArgs`. If `runArgs` already exists, append the value and keep the existing entries:

```json
{
  "runArgs": ["--gpus=all"]
}
```

Then start the container shell:

```bash
devc shell
```

Inside the container, install this harness' Python requirements once, export the prepared interpreter, and run preflight before handing control to the agent:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHON="$(python -c 'import sys; print(sys.executable)')"
PYTHON="$PYTHON" make validate
PYTHON="$PYTHON" make smoke
```

On the H100 node, verify that the container can see the GPU before starting an overnight campaign:

```bash
nvidia-smi
```

For Claude Code, start the agent from the devcontainer shell with:

```bash
claude --permission-mode auto
```

For Codex CLI, install it inside the same devcontainer if needed, start Codex from the repository root, then use `/permissions` to set permissions inside the Codex session to `Auto-review`

Keep the devcontainer boundary meaningful: mount only the repository and deliberate scratch/drop paths, avoid mounting broad host directories or secrets, and remember that the container may still have outbound network access, git identity, and forwarded SSH-agent access depending on how it is configured. For overnight runs, use `tmux` or the devcontainer's persistent shell workflow so the agent can keep running after you disconnect.

## Recommended agent entrypoint

For Codex or Claude Code, the first instruction should be close to:

```text
Use the autofl-nvflare skill.

Start in this directory and read `program.md` first. Treat it as the complete research control plane and follow its setup, mutation, budget, ledger, literature-loop, and continuation instructions.

Start a fresh autoresearch campaign for the local single-H100 node. If no campaign is initialized yet, derive a descriptive run tag at runtime using `<node>-<campaign-topic>-$(date +%Y%m%d)`, initialize the ledger, and establish the baseline first. Do not use date-only names or copy stale example dates.

Use this prepared Python environment for all validation and runs:
PYTHON=<absolute path to the environment's python>
Treat that PYTHON value as authoritative. First verify it with `test -x "$PYTHON"` and `"$PYTHON" -c "import sys; print(sys.executable)"`, then use that exact interpreter for validation, smoke tests, candidate runs, plotting, summaries, and reports.
Do not create a virtual environment, install dependencies, or search for alternate Python interpreters unless I explicitly ask you to. If PYTHON is missing or invalid, stop and ask me for the prepared interpreter path.

Set PARALLEL_CANDIDATES=4 unless I override it. Use one local 80 GB H100; if multiple GPUs are visible, pin candidate runs to CUDA_VISIBLE_DEVICES=0 rather than spreading candidates across devices. Lower the candidate width only if CUDA memory, host memory, or I/O contention appears.

Once setup and baseline are complete, do not ask whether to keep going or whether this is a good stopping point. Continue the experiment loop until manually interrupted.
```

## Scoring recommendation

For automatic comparison, use **cross-site evaluation** and compare the server global model score extracted from:

```text
<result_dir>/server/simulate_job/cross_site_val/cross_val_results.json
```

The helper script `scripts/extract_score.py` reads metrics for the final server/global model key, `SRV_FL_global_model.pt`, and ignores `SRV_best_FL_global_model.pt`.
It tries common metric keys such as `accuracy`, `val_accuracy`, and `test_accuracy`.
Because the current CIFAR-10 validation loader is the same full test set on every simulated client, the harness defaults to final global-model evaluation on `site-1` only. This keeps the same score definition while avoiding redundant client evaluations. Use `--final_eval_clients all` for a full audit run or after making validation site-specific.

The default single-H100 candidate budget is:

```bash
--n_clients 8 --num_rounds 10 --aggregation_epochs 4 --batch_size 64 --eval_batch_size 1024 --alpha 0.5 --seed 0 --model_arch moderate_cnn --max_model_params 5000000 --aggregator weighted --final_eval_clients site-1
```

Candidate runs default to a 600-second timeout through `scripts/run_iteration.sh`, and each appended `results.tsv` row includes `runtime_seconds` for experimental cost accounting.
Training splits are reused across candidates with the same `n_clients`, `alpha`, and `seed` under `/tmp/cifar10_splits/autofl_cifar10_*`. A lock guards split creation, so candidate `--name` values do not create duplicate data-partition directories.
Client training is deterministic by default for a fixed `--seed`: each site derives a stable per-site RNG seed, PyTorch deterministic algorithms are enabled, cuDNN benchmarking is disabled, and DataLoader shuffling/workers are seeded. Use `--no_deterministic_training` only for a deliberately faster but noisier subcampaign, and do not compare those scores directly with deterministic runs.

## Bounded architecture search

The harness now permits architecture search through a registered model selector instead of free-form model replacement:

- `--model_arch moderate_cnn` keeps the original baseline architecture.
- `--model_arch moderate_cnn_norm` adds buffer-free normalization layers while staying near the baseline size.
- `--model_arch moderate_cnn_small_head` keeps the convolutional trunk and uses a smaller classifier head.
- `--max_model_params 5000000` is the default hard parameter cap.

`job.py` and `client.py` both instantiate the same registered architecture and fail before training if the parameter count exceeds the cap. Treat `model_arch` and `max_model_params` as fixed budget fields inside a campaign. Rank architecture candidates primarily by score, use `runtime_seconds` as a coarse secondary signal, and rerun finalists before trusting small score gaps.

## Baseline algorithm knobs

The current harness covers the NVFlare CIFAR-10 simulation benchmark modes that fit the existing DIFF-upload contract:

- FedAvg: `--aggregator weighted`, `--aggregator fedavg`, or NVFlare's built-in `--aggregator default`.
- FedProx: keep a FedAvg-style aggregator and set `--fedproxloss_mu <mu>`.
- FedOpt: use `--aggregator fedavgm` or `--aggregator fedopt` for server SGD momentum over aggregated DIFFs, or `--aggregator fedadam` for a server Adam variant. Tune with `--server_lr`, `--server_momentum`, `--fedopt_beta1`, `--fedopt_beta2`, and `--fedopt_tau`.
- SCAFFOLD: use `--aggregator scaffold`. This automatically passes `--scaffold` to `client.py`, sends client control deltas in `FLModel.meta["scaffold_c_diff"]`, and sends server global controls back in `FLModel.meta["scaffold_c_global"]`.

SCAFFOLD is an explicit opt-in protocol mode. It still preserves the core model contract (`ParamsType.DIFF`, same model keys, `NUM_STEPS_CURRENT_ROUND`) but it does add SCAFFOLD-specific metadata, so keep SCAFFOLD comparisons labeled separately from strict no-extra-meta baselines.

The first post-baseline calibration should run these algorithm families before broader tuning. Run them in batches of up to `PARALLEL_CANDIDATES` concurrent candidates on the same H100 under the same fixed budget:

| Step | Description | Extra args |
| --- | --- | --- |
| 0 | built-in FedAvg audit | `--aggregator default` |
| 1 | explicit FedAvg audit | `--aggregator fedavg` |
| 2 | FedProx light | `--aggregator weighted --fedproxloss_mu 1e-5` |
| 3 | FedProx medium | `--aggregator weighted --fedproxloss_mu 1e-4` |
| 4 | FedAvgM NVFlare-style | `--aggregator fedavgm --server_lr 1.0 --server_momentum 0.6` |
| 5 | FedAvgM accelerated | `--aggregator fedavgm --server_lr 2.0 --server_momentum 0.4` |
| 6 | FedAdam | `--aggregator fedadam --server_lr 1.0 --fedopt_beta1 0.9 --fedopt_beta2 0.99 --fedopt_tau 1e-3` |
| 7 | SCAFFOLD metadata mode | `--aggregator scaffold` |

Keep the step order when assembling calibration batches and schedule any unfinished calibration steps before moving to unrelated sweeps. After the sequence is complete, use the ledger summary to pick the family for narrower follow-up runs.

For single-H100 sweeps, use unique `RUN_LOG` and `--name` values for each concurrent candidate, then rank the ledger:

```bash
"${PYTHON:-python3}" scripts/summarize_results.py results.tsv --status candidate --top "${PARALLEL_CANDIDATES:-4}"
```

After reviewing the table, finalize the completed batch status:

```bash
"${PYTHON:-python3}" scripts/finalize_batch_status.py results.tsv --last "${PARALLEL_CANDIDATES:-4}" --keep-best --discard-others
```

The progress plot reads the `status` column directly. If all successful rows remain `candidate`, the plot will correctly show no kept runs.

To generate an autoresearch-style progress image from the ledger:

```bash
"${PYTHON:-python3}" scripts/plot_progress.py results.tsv --output progress.png
```

## Post-run campaign report

After manually stopping an autoresearch experiment, leave the agent on the experiment branch that contains `results.tsv` and prompt it to run the reporting skill. The skill should not launch more experiments; it only refreshes the plot, writes the report, and commits those reporting artifacts.

Copy-paste prompt:

```text
Use the autofl-nvflare-report skill.

The autoresearch run has been manually stopped. Do not launch new experiments.
Use the current branch and its results.tsv as the source of truth.
Use the prepared PYTHON interpreter if I provided one; do not create a virtual environment or install dependencies.
Refresh progress.png.
Generate reports/<branch>-autoresearch-report.md.
Commit both reports/<branch>-autoresearch-report.md and progress.png to this experiment branch.
If I pasted agent model, effort, or cost output below, include it in the report.
If no model/effort/cost output is pasted, state that agent telemetry was unavailable.

Agent model output, optional:
<paste Claude Code /model output here, or leave empty>

Agent effort output, optional:
<paste Claude Code /effort output here, or leave empty>

Agent cost output, optional:
<paste Claude Code /cost output here, or leave empty>
```

For Claude Code, run `/model`, `/effort`, and `/cost` manually before sending the reporting prompt if you want model, reasoning effort, and token/tooling cost captured. These slash commands are interactive, so the reporting agent cannot invoke them from a tool call.

The skill refreshes `progress.png`, embeds it in `reports/<branch>-autoresearch-report.md`, and commits both the report markdown and `progress.png`. The report covers baseline and best score, absolute and relative lift, final stack, runtime cost, pasted agent model/effort/cost context when available, crash notes, literature-derived ideas with source refs, null/worse ideas, and recommended reproduction or next-step experiments.

## Acknowledgements

See `ACKNOWLEDGEMENTS.md` for provenance. In short:
- the **control-plane idea** and `program.md` workflow are adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch);
- the **literature-loop / QWBE-style proposal workflow** is inspired by the public [Camyla project](https://yifangao112.github.io/camyla-page) and adapted only at the instruction/artifact level;
- the **FL execution substrate** is adapted from existing NVFlare examples and utilities, mainly [CIFAR-10 in PyTorch](../../examples/advanced/cifar10/pt/cifar10-sim).
