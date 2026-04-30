---
name: autofl-nvflare-report
description: Generate and commit a markdown report after an Auto-FL NVFlare autoresearch experiment has been manually stopped. Use when the user asks to summarize a stopped campaign, report achieved improvements, explain implemented literature-derived ideas and sources, refresh progress plots, capture pasted agent model/effort/cost context when available, or commit the final report and progress plot to the current experiment branch.
---

# Auto-FL NVFlare Report

Use this skill after the human has manually stopped an autoresearch campaign.

Do **not** launch new experiments. The job is to summarize the completed branch from `results.tsv`, existing artifacts, source notes, and any available pasted agent model/effort/cost context, then commit the report markdown and refreshed progress plot.

## Workflow

1. Verify context:
   - Confirm the current branch with `git status -sb`.
   - If the branch is `main`, do not auto-commit unless the user explicitly asked to commit to `main`.
   - Confirm `results.tsv` exists and has rows.
   - Use the human-provided `PYTHON` interpreter when one is specified; do not create virtual environments or install dependencies unless the user explicitly asks.
   - Preserve unrelated dirty files; stage only the generated report and refreshed progress plot unless the user asks otherwise.

2. Refresh the progress plot:

   ```bash
   "${PYTHON:-python3}" scripts/plot_progress.py results.tsv --output progress.png
   ```

   If plotting fails because of local font/cache paths, retry with writable cache env vars, for example:

   ```bash
   MPLCONFIGDIR=/tmp/mpl-cache XDG_CACHE_HOME=/tmp/xdg-cache \
     "${PYTHON:-python3}" scripts/plot_progress.py results.tsv --output progress.png
   ```

3. Capture agent model/effort/cost context when available:
   - Claude Code: `/model`, `/effort`, and `/cost` are interactive slash commands and cannot be invoked from a shell or tool call. If the human pasted those outputs into the prompt or saved them to text files, pass the exact text via `--agent-settings`, `--agent-settings-file`, `--agent-cost`, or `--agent-cost-file`.
   - Claude Code without pasted model/effort output: pass `--agent-settings "Agent model/effort telemetry unavailable in this Claude Code runtime; /model and /effort are interactive and were not provided to the reporting agent."`
   - Claude Code without pasted cost output: pass `--agent-cost "Agent cost telemetry unavailable in this Claude Code runtime; /cost is interactive and was not provided to the reporting agent. Experiment runtime cost is reported from results.tsv."`
   - Codex: if no runtime model/effort/cost command is exposed, pass explicit unavailable notes with `--agent-settings` and `--agent-cost`.
   - Keep this separate from experiment runtime cost. Runtime cost comes from `results.tsv` and measures aggregate candidate execution time, not agent token spend.

4. Generate the report:

   ```bash
   "${PYTHON:-python3}" skills/autofl-nvflare-report/scripts/generate_report.py \
     --results results.tsv \
     --plot progress.png \
     --agent-settings-file /tmp/autofl-agent-settings.txt \
     --agent-cost-file /tmp/autofl-agent-cost.txt
   ```

   If no context files exist, use `--agent-settings "<summary>"` and `--agent-cost "<summary>"` instead.
   The script prints the report path. By default it writes to `reports/<branch>-autoresearch-report.md`.

5. Review the generated report for obvious parsing issues:
   - best score and baseline are present;
   - progress plot is embedded as a markdown image and points to the refreshed plot path;
   - final recommended stack includes exact budget/args;
   - runtime section includes total aggregate runtime and average runtime per candidate;
   - agent/tooling context section is present and either contains pasted model/effort/cost output or explains why it was unavailable;
   - literature-derived ideas include source refs from `[src: ...]` markers when present;
   - the report explicitly distinguishes candidates from reproduced/kept results.

6. Commit the report and progress plot automatically to the current experiment branch:

   ```bash
   git add <report-path> progress.png
   git commit -m "Add autoresearch campaign report"
   ```

   If a non-default plot path was used, stage that plot path instead of `progress.png`.
   Do not stage `run_logs/`, NVFlare result directories, generated local caches, or unrelated edits.

## Report Expectations

The report should lead with an executive summary, then include a technical appendix. It must cover:

- baseline score, best score, absolute and relative lift;
- embedded progress plot image;
- best run index, description, commit, budget, runtime, status, and artifact path;
- major running-best milestones and likely improvement mechanisms;
- final recommended stack and whether it preserves the current contract or uses an explicit protocol mode such as SCAFFOLD;
- total aggregate candidate runtime, average runtime per candidate, crash count, and failure notes;
- agent model/effort settings and agent/tooling cost if provided by the human, or explicit unavailable notes;
- literature-derived ideas, source refs, and whether each helped;
- null/worse/unstable ideas that should not be retried;
- recommended reproduction and next experiment directions.
