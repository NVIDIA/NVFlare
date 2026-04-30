# Runbook

## Recommended loop
1. Read `program.md` first when present.
2. Use the human-provided `PYTHON` interpreter when one is specified. Treat it as authoritative, verify it with `test -x "$PYTHON"` and `"$PYTHON" -c "import sys; print(sys.executable)"`, and do not search for alternate interpreters with glob or discovery commands such as `ls /usr/bin/python*`, `ls /workspace/.venv*/bin/python*`, or `which python`.
3. Do not create virtual environments or install dependencies unless the user explicitly asks. If `PYTHON` is missing or invalid, ask the user for the prepared interpreter path instead of guessing.
4. When initializing a campaign, use a descriptive branch tag with the pattern `<node>-<campaign-topic>-YYYYMMDD`, such as `h100-fedavgm-20260430` or `h100-archsearch-20260430`; never use date-only branch names.
5. Propose one small mutation.
6. Edit the smallest possible set of files.
7. Run validation.
8. Run a smoke test.
9. On a 4 x H100 node, prefer a `PARALLEL_CANDIDATES` same-budget sweep; default to `PARALLEL_CANDIDATES=4` unless the initial human instruction sets another value.
10. Use the default H100 candidate budget unless told otherwise: 8 clients, 10 rounds, 4 local epochs, training batch size 64, eval batch size 1024, alpha 0.5, seed 0, `model_arch=moderate_cnn`, `max_model_params=5000000`, weighted aggregation, deterministic client training, final global evaluation on site-1, 600-second timeout.
11. Use assigned `CUDA_VISIBLE_DEVICES` values plus unique `RUN_LOG` and `--name` values for every parallel candidate; if `PARALLEL_CANDIDATES > 4`, map lane `i` to GPU `i % 4` unless `GPU_IDS` is provided.
12. Record each result in `results.tsv`. Successful runs are appended as `candidate`, which means unreviewed.
13. Rank the batch with `"${PYTHON}" scripts/summarize_results.py results.tsv --status candidate --top "${PARALLEL_CANDIDATES:-4}"`.
14. Decide whether to keep, narrow, or revert after all candidates finish. Rank primarily by score; use runtime as a coarse secondary signal and prefer the faster/simpler candidate when scores are within noise.
15. Finalize reviewed statuses before starting the next batch: promote the selected survivor to `keep`, mark reviewed non-survivors as `discard`, leave crashes as `crash`, and leave only unresolved active rows as `candidate`. Prefer `"${PYTHON}" scripts/finalize_batch_status.py results.tsv --last "${PARALLEL_CANDIDATES:-4}" --keep-best --discard-others`.
16. If a candidate implements a paper-derived method, include a compact source ref in the `results.tsv` description field and fuller citation details in `templates/mutation_report.md`.
17. Commit `results.tsv` on the experiment branch after baseline and each completed batch/checkpoint.
18. Continue with the next same-budget batch until manually interrupted; do not ask whether to keep going after setup and baseline.
19. If two batches fail to improve or the next axis is unclear, run a literature-grounded proposal loop: search papers, extract challenges, score contract-safe ideas, and launch the top `PARALLEL_CANDIDATES` compatible candidates.
20. Summarize the result when interrupted or when reporting a checkpoint.

## Serial fallback
If only one GPU or one candidate is available, run a single redirected iteration via `PYTHON=<path> bash scripts/run_iteration.sh`.

## Report format
- Hypothesis
- Files changed
- Commands run
- Observed outcome
- Literature basis
- Batch analysis
- Contract check
- Rollback risk
- Next mutation
