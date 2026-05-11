# Runbook

## Recommended loop
1. Read `program.md` first when present, then read the active task profile. Use `tasks/cifar10/profile.md` by default.
2. Use the interpreter and dependency rules from the active task profile. For the default CIFAR-10 profile, set and use `PYTHON=.venv/bin/python` by default, unless the human explicitly provides a different `PYTHON` value. Treat the selected value as authoritative, verify it with `test -x "$PYTHON"` and `"$PYTHON" -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version; print(sys.executable)"`, and do not search for alternate interpreters with glob or discovery commands such as `ls /usr/bin/python*`, `ls /workspace/.venv*/bin/python*`, or `which python`.
3. Do not create virtual environments or install dependencies unless the user explicitly asks. If the active profile's interpreter is missing or invalid and no override was provided, tell the user to rerun that profile's preflight instead of guessing.
4. When initializing a campaign, use a descriptive branch tag with the pattern `<node>-<campaign-topic>-YYYYMMDD`, such as `h100-fedavgm-20260430` or `h100-archsearch-20260430`; never use date-only branch names.
5. Before validation, smoke tests, baseline, or candidates, run `bash scripts/init_run.sh <tag>` and verify `git branch --show-current` starts with `autoresearch/`. Do not run experiments on `main`, `upstream/main`, the starter branch, or a shared feature branch.
6. Propose one small mutation or a small same-budget candidate batch.
7. Edit the smallest possible set of files.
8. Run validation.
9. Run a smoke test.
10. Follow the active task profile's local hardware and candidate-width rules. For the default CIFAR-10/H100 profile, launch up to `PARALLEL_CANDIDATES=4` same-budget candidates concurrently on one local H100 when memory allows, and reduce the width if candidates hit CUDA OOM or host contention.
11. Use the active task profile's default candidate budget unless told otherwise. For the default CIFAR-10/H100 profile, that budget is 8 clients, 20 communication rounds, 4 local epochs, `local_train_steps=0`, training batch size 64, eval batch size 1024, alpha 0.5, seed 0, `model_arch=moderate_cnn`, `max_model_params=5000000`, weighted aggregation, deterministic client training, final global evaluation on site-1, and a 1200-second timeout. Local epochs or `local_train_steps` may be swept under that runtime cap, but do not vary both in the same narrow sweep.
12. Use unique `RUN_LOG` and `--name` values for every candidate. If the active profile requires one local GPU, pin each run with `CUDA_VISIBLE_DEVICES=0` instead of spreading candidates across devices.
13. Record each result in `results.tsv`. `run_iteration.sh` initializes the header before launching logged runs; successful runs are appended as `candidate`, which means unreviewed.
14. Rank the completed batch with `"${PYTHON}" scripts/summarize_results.py results.tsv --status candidate --top "${PARALLEL_CANDIDATES:-4}"`.
15. Decide whether to keep, narrow, or revert after the batch finishes. Rank primarily by score; use runtime as a coarse secondary signal and prefer the faster/simpler candidate when scores are within noise.
16. Finalize reviewed statuses before starting the next batch: promote the selected survivor to `keep`, mark reviewed non-survivors as `discard`, leave crashes as `crash`, and leave only unresolved active rows as `candidate`. Prefer `"${PYTHON}" scripts/finalize_batch_status.py results.tsv --last "${PARALLEL_CANDIDATES:-4}" --keep-best --discard-others`.
17. If a candidate implements a paper-derived method, include a compact source ref in the `results.tsv` description field and fuller citation details in `templates/mutation_report.md`.
18. Run `"${PYTHON}" scripts/plateau_watchdog.py results.tsv` after every finalized batch. If it prints `recommendation=literature`, stop local jitter sweeps and run a literature-grounded proposal loop before launching more candidates. If it prints `recommendation=continue`, keep iterating locally unless repeated crashes share one root cause or no non-duplicate safe axis remains.
19. Commit `results.tsv` on the active `autoresearch/` branch after baseline and each completed run/checkpoint. Commit surviving code changes as soon as they are kept; do not carry kept changes uncommitted into the next batch.
20. Continue with the next same-budget candidate batch until manually interrupted; do not ask whether to keep going after setup and baseline.
21. When the watchdog fires, or when repeated crashes share one root cause and a source-backed fix is needed before more runs are useful, run the literature loop: start timing with `"${PYTHON}" scripts/log_literature_review.py --start --description "plateau after <rows>: <symptom>"`, search papers, extract challenges, score contract-safe ideas, append the `literature` event with `--finish`, and launch the top compatible candidate batch next.
22. Summarize the result when interrupted or when reporting a checkpoint.

## Single-H100 mode
For the default CIFAR-10/H100 profile, run same-budget candidate batches via `PYTHON=.venv/bin/python TASK_DIR=tasks/cifar10 bash scripts/run_iteration.sh`, with unique `RUN_LOG` and `--name` values for each concurrent candidate. Default to `PARALLEL_CANDIDATES=4`, and reduce the width if CUDA memory or host contention appears. For other profiles, use that profile's hardware, environment, and candidate-width rules.

## Report format
- Hypothesis
- Files changed
- Commands run
- Observed outcome
- Literature basis
- Run analysis
- Contract check
- Rollback risk
- Next mutation
