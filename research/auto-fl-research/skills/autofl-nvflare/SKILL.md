---
name: autofl-nvflare
description: Help coding agents work on an NVFlare-based Auto-FL harness that follows an autoresearch-style loop. Use when the user wants to create, edit, debug, or extend program.md, task folders such as tasks/cifar10/ and tasks/vlm_med/, task-local job.py, client.py, model.py, shared custom_aggregators.py, mutation policies, results.tsv logging, or coding-agent prompts for a guardrailed federated-learning research loop. This skill is specifically for NVFlare harness work where the Client API loop, DIFF upload contract, and NUM_STEPS_CURRENT_ROUND metadata must stay intact unless the user explicitly asks for a protocol upgrade.
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
4. run the active task profile's static checks and syntax validation, with `TASK_DIR` set to the active task
5. run the client contract validator against the active task's `client.py`, not a stale root-level path
6. run the active task profile's smoke command if the prepared environment has `nvflare`; for non-CIFAR tasks, pass the task-specific `SMOKE_ARGS` or use `scripts/run_iteration.sh` with the active task budget
7. follow the active task profile's local hardware and candidate-width rules. For the default CIFAR-10/H100 profile, launch up to `PARALLEL_CANDIDATES=4` same-budget candidates concurrently on one local H100 when memory allows, and reduce the width if candidates hit CUDA OOM or host contention
8. use the active task profile's default candidate budget unless told otherwise. For the default CIFAR-10/H100 profile, that budget is 8 clients, 20 communication rounds, 4 local epochs, `local_train_steps=0`, training batch size 64, eval batch size 1024, alpha 0.5, seed 0, `model_arch=moderate_cnn`, `max_model_params=5000000`, weighted aggregation, deterministic client training, final global evaluation on site-1, and a 1200-second timeout; local epochs or `local_train_steps` may be swept under that runtime cap, but do not vary both in the same narrow sweep
9. use unique `RUN_LOG` and job `--name` values for each candidate; if the active profile requires one local GPU, pin each run with `CUDA_VISIBLE_DEVICES=0` instead of spreading candidates across devices
10. record the outcome in `results.tsv`; `run_iteration.sh` initializes the header before launching logged runs, and successful runs are appended as `candidate`, which means unreviewed, not kept
11. after every completed batch, update reviewed `results.tsv` statuses before launching the next batch: promote the selected survivor to `keep`, mark reviewed non-survivors as `discard`, leave crashes as `crash`, and leave only unresolved active rows as `candidate`; prefer `scripts/finalize_batch_status.py --last "${PARALLEL_CANDIDATES:-4}"`
12. commit that ledger on the active `autoresearch/` branch after baseline and completed runs/checkpoints, and commit surviving code changes as soon as they are kept rather than carrying them uncommitted into the next batch
13. if a candidate implements a paper-derived method, include a compact source ref in the `results.tsv` description field and fuller citation details in `templates/mutation_report.md`
14. rank the completed batch against the ledger before deciding whether to keep, narrow, or revert; rank primarily by score, use runtime as a coarse secondary signal, and prefer the faster/simpler candidate when scores are within noise
15. if the user gives an explicit `N`-candidate budget, count up to `N` comparable candidate attempts after the baseline. Do not count deterministic import, validation, smoke, plotting, reporting, the baseline, or infrastructure-only retries caused by sandbox/socket/runtime setup. Count a real candidate crash once the candidate run starts under the intended execution environment
16. if the user does not give an explicit candidate cap, run the original autoresearch loop: after setup and baseline, continue launching same-budget candidate batches until manually interrupted; do not invent a default stopping point, do not ask whether to keep going, and do not send a final response while safe comparable candidates remain. Progress updates in uncapped mode must be status observations, not "should I continue?" questions; continue unless the human explicitly interrupts or the campaign guard allows finalization
17. after every finalized batch, run `scripts/plateau_watchdog.py results.tsv`; treat plateau as a decision checkpoint, not an automatic stop. If it prints `recommendation=literature`, run the Camyla-inspired literature loop from `program.md`, record the `literature` event row in `results.tsv`, refresh `progress.png`, and launch the top compatible candidate batch next unless the user asked to stop. If it prints `recommendation=continue`, refresh `progress.png` and keep iterating locally unless repeated crashes share one root cause or no non-duplicate safe axis remains
18. after every finalized batch, report checkpoint, refreshed plot, local commit, encoded default verification, cap check, or possible stop point, run `"${PYTHON:-python3}" scripts/campaign_guard.py results.tsv --state .autoresearch/campaign_state.json --format json`
19. treat `.autoresearch/campaign_state.json` as authoritative campaign state. If `final_response_allowed` is `false`, do not produce a final answer; execute the returned `next_action` immediately. If `next_action` is `finalize_pending_candidates`, finalize the reviewed rows and rerun the guard. If it is `run_literature_loop`, run the literature loop and launch source-backed candidates. If it is `launch_next_candidate_batch`, choose the next same-budget axis and launch it
20. never treat a kept improvement, refreshed report, refreshed `progress.png`, encoded `job.py` defaults, or local commit as a stopping condition in an uncapped campaign; these are checkpoints before the next same-budget batch unless `campaign_guard.py` says `final_response_allowed=true`
21. report the mutation hypothesis, changed files, commands run, observed outcome, literature basis, run analysis, and next mutation; then launch the next candidate batch when the guard says to continue
22. refresh the progress plot after every finalized batch, cap exhaustion, manual stop, plateau checkpoint, or hard-blocker checkpoint with `"${PYTHON:-python3}" scripts/plot_progress.py results.tsv --output progress.png`; if matplotlib cache paths fail, retry with writable `MPLCONFIGDIR` and `XDG_CACHE_HOME`
23. if an active NVFLARE simulator candidate stalls on a hard child-process connection failure such as `Failed to create connection to the child process in SimulatorClientRunner`, or on a configured simulator no-progress watchdog after a round is dispatched but server/client progress markers stop advancing, recover inside the same campaign: mark only that candidate as `crash`, terminate the stuck `job.py` child if needed, refresh `results.tsv`, `progress.png`, and `.autoresearch/campaign_state.json`, then continue with the next same-budget candidate. Do not start a new campaign directory, new branch, new baseline, new objective, or final report unless the human explicitly asked to reset
24. do not treat a quiet NVFLARE server log as a stall by itself. After a round is dispatched, inspect `/tmp/nvflare/simulation/<run>/site-*/log.txt` or `site-*/log_fl.txt` for client epoch, finished-training, download, or task-completion progress before interrupting. If any client log or server aggregation marker advances within the expected candidate runtime, keep waiting on the same candidate; never stop the runner, final-answer, or start a new campaign for that pattern
25. only produce a final answer when `scripts/campaign_guard.py` reports `final_response_allowed=true`; then end with reproducible artifacts: finalized `results.tsv`, refreshed `progress.png`, `.autoresearch/campaign_state.json`, and a concise report or `templates/mutation_report.md` entry covering the baseline, best score, artifacts, command provenance, failures, product friction, and next mutation. The final answer must include absolute paths to `autofl.yaml` when present, `results.tsv`, `progress.png`, `.autoresearch/campaign_state.json`, and any report artifact, plus the best candidate, metric improvement, and whether the guard stopped because of a cap, manual interruption, or hard blocker

## References

Read these when relevant:
- `references/mutation-schema.md` for the allowed mutation surface
- `references/runbook.md` for the recommended iteration loop and reporting format
- `references/provenance.md` for acknowledgement and attribution guidance
