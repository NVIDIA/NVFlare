---
name: autofl-nvflare
description: help chatgpt work on an nvflare-based auto-fl harness that follows an autoresearch-style loop. use when the user wants to create, edit, debug, or extend program.md, job.py, client.py, custom_aggregators.py, model.py, mutation policies, results.tsv logging, or coding-agent prompts for a bounded federated-learning research loop. this skill is specifically for nvflare harness work where the client api loop, diff upload contract, and num_steps_current_round metadata must stay intact unless the user explicitly asks for a protocol upgrade.
---

# autofl-nvflare

Use this skill to keep edits to the Auto-FL NVFlare starter coherent, safe, and aligned with an **autoresearch-style** operating model.

## Entry point

When the target repo includes `program.md`, read it first and treat it as the single control plane.

Use `mutation_schema.yaml` for bounded mutation details only when `program.md` points you there or when choosing a mutation axis. Use `AGENTS.md` / `CLAUDE.md` only as thin local guardrails.

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

1. Client-local changes in `client.py`
   - optimizer family
   - scheduler settings
   - local epochs, batch size, workers
   - weight decay
   - gradient clipping
   - label smoothing
   - FedProx local loss
   - extra scalar metrics
2. Aggregation changes in `custom_aggregators.py`
   - weighted aggregation refinements
   - FedAvg/FedOpt-style DIFF aggregation that stays inside the existing FLModel contract
   - explicit SCAFFOLD control-variate metadata when the user has opted into that protocol mode
   - clipping / robust aggregation
   - median or trimmed-mean style logic
3. Recipe changes in `job.py`
   - rounds
   - clients
   - `cross_site_eval`
   - `launch_external_process`
   - `client_memory_gc_rounds`
4. Registered architecture changes in `model.py`
   - named `model_arch` variants
   - parameter-count checks through `max_model_params`
   - no new dependencies

Do not change model architecture outside registered `model_arch` variants or the active `max_model_params` budget. Do not add server-coupled protocol fields outside an explicitly requested protocol mode.
FedProx is v0-safe as a client-local loss term. FedOpt is v0-safe only when it is implemented inside the custom aggregator over already-received DIFFs. SCAFFOLD is available only as an explicit opt-in mode that uses `FLModel.meta` for `scaffold_c_diff` and `scaffold_c_global`.

## Required workflow

After making edits:
1. use the human-provided `PYTHON` interpreter when one is specified; treat it as authoritative, verify it with `test -x "$PYTHON"` and `"$PYTHON" -c "import sys; print(sys.executable)"`, and do not search for alternate interpreters with glob or discovery commands such as `ls /usr/bin/python*`, `ls /workspace/.venv*/bin/python*`, or `which python`
2. do not create virtual environments or install dependencies unless the user explicitly asks; if `PYTHON` is missing or invalid, ask the user for the prepared interpreter path instead of guessing
3. when initializing a campaign, use a descriptive branch tag with the pattern `<node>-<campaign-topic>-YYYYMMDD`, such as `h100-fedavgm-20260430` or `h100-archsearch-20260430`; never use date-only branch names
4. run static checks and syntax validation
5. run the client contract validator if present
6. run the smoke test if the prepared environment has `nvflare`
7. on a 4 x H100 node, prefer a `PARALLEL_CANDIDATES` same-budget sweep; default to `PARALLEL_CANDIDATES=4` unless the initial human instruction sets another value
8. use the default H100 candidate budget unless told otherwise: 8 clients, 10 rounds, 4 local epochs, training batch size 64, eval batch size 1024, alpha 0.5, seed 0, `model_arch=moderate_cnn`, `max_model_params=5000000`, weighted aggregation, deterministic client training, final global evaluation on site-1, 600-second timeout
9. use assigned `CUDA_VISIBLE_DEVICES` values plus unique `RUN_LOG` and job `--name` values for each parallel candidate; if `PARALLEL_CANDIDATES > 4`, map lane `i` to GPU `i % 4` unless `GPU_IDS` is provided
10. record the outcome in `results.tsv`; successful runs are appended as `candidate`, which means unreviewed, not kept
11. after every completed batch, update reviewed `results.tsv` statuses before launching the next batch: promote the selected survivor to `keep`, mark reviewed non-survivors as `discard`, leave crashes as `crash`, and leave only unresolved active rows as `candidate`; prefer `scripts/finalize_batch_status.py`
12. commit that ledger on experiment branches after baseline and completed batches/checkpoints
13. if a candidate implements a paper-derived method, include a compact source ref in the `results.tsv` description field and fuller citation details in `templates/mutation_report.md`
14. rank the batch before deciding whether to keep, narrow, or revert; rank primarily by score, use runtime as a coarse secondary signal, and prefer the faster/simpler candidate when scores are within noise
15. after setup and baseline, continue launching same-budget batches until manually interrupted; do not ask whether to keep going
16. if progress stalls, run the Camyla-inspired literature loop from `program.md`: generate diverse queries, triage primary papers, extract challenge cards, score contract-safe proposals in `templates/literature_loop.md`, and launch the top `PARALLEL_CANDIDATES` compatible candidates
17. report the mutation hypothesis, changed files, commands run, observed outcome, literature basis, batch analysis, and next mutation

## References

Read these when relevant:
- `references/mutation-schema.md` for the allowed mutation surface
- `references/runbook.md` for the recommended iteration loop and reporting format
- `references/provenance.md` for acknowledgement and attribution guidance
