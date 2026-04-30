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
9. Assume one local H100; run one same-budget candidate at a time and default to `PARALLEL_CANDIDATES=1` unless the human explicitly changes the compute assumption.
10. Use the default H100 candidate budget unless told otherwise: 8 clients, 10 rounds, 4 local epochs, training batch size 64, eval batch size 1024, alpha 0.5, seed 0, `model_arch=moderate_cnn`, `max_model_params=5000000`, weighted aggregation, deterministic client training, final global evaluation on site-1, 600-second timeout.
11. Use unique `RUN_LOG` and `--name` values for every candidate. If the environment exposes multiple GPUs but this campaign should use the local H100 only, pin the run with `CUDA_VISIBLE_DEVICES=0`.
12. Record each result in `results.tsv`. Successful runs are appended as `candidate`, which means unreviewed.
13. Rank the completed run with `"${PYTHON}" scripts/summarize_results.py results.tsv --status candidate --top 1`.
14. Decide whether to keep, narrow, or revert after the candidate finishes. Rank primarily by score; use runtime as a coarse secondary signal and prefer the faster/simpler candidate when scores are within noise.
15. Finalize reviewed statuses before starting the next run: promote the selected survivor to `keep`, mark reviewed non-survivors as `discard`, leave crashes as `crash`, and leave only unresolved active rows as `candidate`. Prefer `"${PYTHON}" scripts/finalize_batch_status.py results.tsv --last 1 --keep-best --discard-others`.
16. If a candidate implements a paper-derived method, include a compact source ref in the `results.tsv` description field and fuller citation details in `templates/mutation_report.md`.
17. Commit `results.tsv` on the experiment branch after baseline and each completed run/checkpoint.
18. Continue with the next same-budget candidate until manually interrupted; do not ask whether to keep going after setup and baseline.
19. If two candidates fail to improve or the next axis is unclear, run a literature-grounded proposal loop: search papers, extract challenges, score contract-safe ideas, and launch the top compatible candidate next.
20. Summarize the result when interrupted or when reporting a checkpoint.

## Single-H100 mode
Run a single redirected iteration via `PYTHON=<path> bash scripts/run_iteration.sh`.

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
