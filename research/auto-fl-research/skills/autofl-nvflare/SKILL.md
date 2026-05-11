---
name: autofl-nvflare
description: Help coding agents work on an NVFlare-based Auto-FL harness that follows an autoresearch-style loop. Use when the user wants to create, edit, debug, or extend program.md, task folders such as tasks/cifar10/ and tasks/vlm_med/, task-local job.py, client.py, model.py, shared custom_aggregators.py, mutation policies, results.tsv logging, or coding-agent prompts for a bounded federated-learning research loop. This skill is specifically for NVFlare harness work where the Client API loop, DIFF upload contract, and NUM_STEPS_CURRENT_ROUND metadata must stay intact unless the user explicitly asks for a protocol upgrade.
---

# autofl-nvflare

Use this skill to keep edits to the Auto-FL NVFlare starter coherent, safe, and aligned with an **autoresearch-style** operating model.

## Entry point

When the target repo includes `program.md`, read it first and treat it as the general control plane. Then read the active task profile; use `tasks/cifar10/profile.md` when the human does not specify another profile.

Use the active task's `mutation_schema.yaml` for bounded mutation details only when `program.md` or the active task profile points you there, or when choosing a mutation axis. Use `AGENTS.md` / `CLAUDE.md` only as thin local guardrails.

## Core rules

Preserve these invariants unless the user explicitly asks for a protocol change:
- `flare.init()`
- `while flare.is_running():`
- `input_model = flare.receive()`
- `flare.send(output_model)`
- `model.load_state_dict(input_model.params, strict=True)`
- `compute_model_diff(model, global_model)`
- `output_model.params_type == ParamsType.DIFF`
- `output_model.meta["NUM_STEPS_CURRENT_ROUND"]`
- the optional `flare.is_evaluate()` branch
- the same selected `model_arch` on server and clients for a run
- the active `max_model_params` cap for architecture campaigns

## Preferred mutation order

1. Client-local changes in the active task's `client.py`
   - optimizer family
   - scheduler settings
   - local epochs, fixed local training steps, batch size, workers
   - weight decay
   - gradient clipping
   - label smoothing
   - FedProx local loss
   - extra scalar metrics
2. Aggregation changes in shared `tasks/shared/custom_aggregators.py`
   - weighted aggregation refinements
   - FedAvg/FedOpt-style DIFF aggregation that stays inside the existing FLModel contract
   - explicit SCAFFOLD control-variate metadata when the user has opted into that protocol mode
   - clipping / robust aggregation
   - median or trimmed-mean style logic
3. Recipe changes in the active task's `job.py`
   - rounds
   - clients
   - `cross_site_eval`
   - `launch_external_process`
   - `client_memory_gc_rounds`
4. Registered architecture changes in the active task's `model.py`
   - named `model_arch` variants
   - parameter-count checks through `max_model_params`
   - no new dependencies

Do not change model architecture outside registered `model_arch` variants or the active `max_model_params` budget. Do not add server-coupled protocol fields outside an explicitly requested protocol mode.
FedProx is compatible as a client-local loss term. FedOpt is compatible only when it is implemented inside the custom aggregator over already-received DIFFs. SCAFFOLD is available only as an explicit opt-in mode that uses `FLModel.meta` for `scaffold_c_diff` and `scaffold_c_global`.

## Required workflow

After making edits:
1. use the interpreter and dependency rules from the active task profile. For the default CIFAR-10 profile, set and use `PYTHON=.venv/bin/python` by default, unless the human explicitly provides a different `PYTHON` value; treat the selected value as authoritative, verify it with `test -x "$PYTHON"` and `"$PYTHON" -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version; print(sys.executable)"`, and do not search for alternate interpreters with glob or discovery commands such as `ls /usr/bin/python*`, `ls /workspace/.venv*/bin/python*`, or `which python`
2. do not create virtual environments or install dependencies unless the user explicitly asks; if the active profile's interpreter is missing or invalid and no override was provided, tell the user to rerun that profile's preflight instead of guessing
3. when initializing a campaign, use a descriptive branch tag with the pattern `<node>-<campaign-topic>-YYYYMMDD`, such as `h100-fedavgm-20260430` or `h100-archsearch-20260430`; run `bash scripts/init_run.sh <tag>` before validation, baseline, or candidates; verify `git branch --show-current` starts with `autoresearch/`; never run experiments on `main`, `upstream/main`, the starter branch, or a shared feature branch; never use date-only branch names
4. run static checks and syntax validation
5. run the client contract validator if present
6. run the smoke test if the prepared environment has `nvflare`
7. follow the active task profile's local hardware and candidate-width rules. For the default CIFAR-10/H100 profile, launch up to `PARALLEL_CANDIDATES=4` same-budget candidates concurrently on one local H100 when memory allows, and reduce the width if candidates hit CUDA OOM or host contention
8. use the active task profile's default candidate budget unless told otherwise. For the default CIFAR-10/H100 profile, that budget is 8 clients, 20 communication rounds, 4 local epochs, `local_train_steps=0`, training batch size 64, eval batch size 1024, alpha 0.5, seed 0, `model_arch=moderate_cnn`, `max_model_params=5000000`, weighted aggregation, deterministic client training, final global evaluation on site-1, and a 1200-second timeout; local epochs or `local_train_steps` may be swept under that runtime cap, but do not vary both in the same narrow sweep
9. use unique `RUN_LOG` and job `--name` values for each candidate; if the active profile requires one local GPU, pin each run with `CUDA_VISIBLE_DEVICES=0` instead of spreading candidates across devices
10. record the outcome in `results.tsv`; `run_iteration.sh` initializes the header before launching logged runs, and successful runs are appended as `candidate`, which means unreviewed, not kept
11. after every completed batch, update reviewed `results.tsv` statuses before launching the next batch: promote the selected survivor to `keep`, mark reviewed non-survivors as `discard`, leave crashes as `crash`, and leave only unresolved active rows as `candidate`; prefer `scripts/finalize_batch_status.py --last "${PARALLEL_CANDIDATES:-4}"`
12. commit that ledger on the active `autoresearch/` branch after baseline and completed runs/checkpoints, and commit surviving code changes as soon as they are kept rather than carrying them uncommitted into the next batch
13. if a candidate implements a paper-derived method, include a compact source ref in the `results.tsv` description field and fuller citation details in `templates/mutation_report.md`
14. rank the completed batch against the ledger before deciding whether to keep, narrow, or revert; rank primarily by score, use runtime as a coarse secondary signal, and prefer the faster/simpler candidate when scores are within noise
15. after setup and baseline, continue launching same-budget candidate batches until manually interrupted; do not ask whether to keep going
16. after every finalized batch, run `scripts/plateau_watchdog.py results.tsv`; if it prints `recommendation=literature`, stop local jitter sweeps and run the Camyla-inspired literature loop from `program.md`: time it with `scripts/log_literature_review.py --start` / `--finish`, generate diverse queries, triage primary papers, extract challenge cards, score contract-safe proposals in `templates/literature_loop.md`, record the `literature` event row in `results.tsv`, and launch the top compatible candidate batch next; if it prints `recommendation=continue`, do not log another literature row for a routine missed batch, and keep iterating locally unless repeated crashes share one root cause or no non-duplicate safe axis remains
17. report the mutation hypothesis, changed files, commands run, observed outcome, literature basis, run analysis, and next mutation

## References

Read these when relevant:
- `references/mutation-schema.md` for the allowed mutation surface
- `references/runbook.md` for the recommended iteration loop and reporting format
- `references/provenance.md` for acknowledgement and attribution guidance
