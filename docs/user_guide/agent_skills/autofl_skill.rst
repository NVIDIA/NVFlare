.. _autofl_skill:

############################
NVFlare Auto-FL Agent Skill
############################

The NVFlare Auto-FL agent skill optimizes an existing NVFlare ``job.py``
through a coding agent. It is not an ``nvflare autofl`` command and does not
add an Auto-FL command family to the NVFlare CLI.

Install the NVFlare-owned skill set from the repository root for Codex and
Claude Code:

.. code-block:: shell

   npx skills add ./skills -a codex -a claude-code

The standard Agent Skills installer places the complete skill set, including
Auto-FL's deterministic helper scripts, in each selected agent's managed skill
directory. NVFlare does not provide a separate skill installation command, and
skills are not distributed through the NVFlare Python wheel.

User Experience
===============

Select the installed skill in the coding agent and provide the optimization
intent:

.. code-block:: text

   Select: NVFlare Auto-FL skill
   Prompt: Optimize ./job.py for accuracy in sim.

An explicit candidate budget is optional. Without one, the campaign continues
until the user interrupts it or a hard safety or runtime blocker prevents
further comparable execution.

Users do not invoke scripts from the installed skill directory. The activated
coding agent resolves and runs those bundled resources internally. Bundled
scripts are private implementation details, not public NVFlare commands or
Python APIs.

Deterministic Import
====================

The skill first parses ``job.py`` without importing or executing it. Its private
importer recognizes supported NVFlare Recipe and FedJob construction patterns,
aliases, ``SimEnv`` configuration, fixed training budgets, metrics, train
scripts, and common argparse tunables.

The result is a reviewable ``autofl.yaml`` containing:

- the optimization metric, direction, environment, and candidate budget;
- the fixed comparison budget that candidates must preserve;
- ``trust_contract.allowed_edit_paths`` and allowed Python creation patterns;
- source and importer provenance;
- unresolved dynamic or unsupported fields requiring review.

When the user does not name a metric, a deterministic ``key_metric`` extracted
from ``job.py`` takes precedence. The default user experience does not require
editing ``autofl.yaml``.

Simulation Execution Permission
===============================

Before a simulation campaign starts, the coding agent asks once for approval
to run the installed Auto-FL runner outside its restricted command sandbox.
Approve only the resolved Python interpreter and runner, the ``initialize``
and ``evaluate`` actions, and the selected absolute ``job.py``. Do not approve
generic Python or shell execution, full access, another job, or POC/production
commands. The skill does not create or modify agent permission configuration.

This approval is needed because the local simulator binds sockets. It also
means that user and agent-authored training code runs with the runner's host
privileges. Use a disposable container or dedicated VM for autonomous
campaigns. Declining the approval prevents local simulation but does not relax
the sandbox or bypass normal POC/production authentication and site policy.

Candidate Lifecycle
===================

The coding agent forms a hypothesis and asks the private skill runner to create
an isolated candidate source tree. The agent may edit allowed existing files or
add Python modules, including new client algorithms and server aggregators.

For every candidate, NVFlare-owned helper code:

- recomputes changed files rather than trusting agent-written manifest paths;
- rejects edits outside the trust contract and fixed-budget drift;
- runs simulation through the configured ``SimEnv`` or prepares standard POC
  or production submission;
- records the score, metric provenance, source patch, command, artifacts, and
  failure evidence;
- retains an improved candidate or restores the previous best source.

POC and production candidates use the normal ``nvflare job submit``, ``job
wait``, and ``job download`` lifecycle with configured startup-kit policy. The
skill does not bypass authentication or site policy.

Campaign Artifacts
==================

The job directory contains the human-reviewable and reproducibility artifacts:

- ``autofl.yaml``: imported campaign and trust contract;
- ``results.tsv``: atomic candidate ledger with metric provenance;
- ``progress.png``: campaign trajectory;
- ``autofl_report.md``: current campaign summary;
- ``.nvflare/autofl/campaign_state.json``: next action and stop status;
- ``.nvflare/autofl/candidates/<id>/candidate_manifest.json``: candidate
  hypothesis, source hashes, changed files, result, and artifacts.

A manual stop takes precedence over pending execution. If a candidate is
pending, the agent abandons it safely before generating the final report.
NVFlare serializes lifecycle actions for each job workspace. If another action
is already active, the helper exits with code 2 and the agent retries after the
active action finishes; separate job workspaces remain independent.

Supported First Version
=======================

The first version supports statically recognizable NVFlare Recipe constructors
and NVFlare-distributed classes ending in ``Job``. Generic, local, and
non-NVFlare job or recipe classes remain unresolved. Ambiguous scripts and
dynamic safety-critical comparison fields block baseline execution rather than
being guessed.

Final Report After Stop
=======================

After a campaign stops cleanly, reaches its explicit cap, or ends at a hard
blocker, select the companion report skill:

.. code-block:: text

   Select: NVFlare Auto-FL Report skill
   Prompt: Generate the final report for the stopped campaign in ./job.

The report skill verifies that authoritative campaign state allows a final
response and that no ledger row or candidate manifest remains pending. For an
abrupt interruption, the human must first confirm that no campaign process is
running; this confirmation bypasses stale stop state only and never unfinished
candidate evidence.

The skill refreshes ``progress.png`` when plotting is available and generates:

- ``autofl_final_report.md`` with the trajectory, retained best candidate,
  metric provenance, lineage, literature outcomes, failures, commands, and
  comparability warnings;
- ``autofl_report_summary.json`` with the same evidence under the skill-local
  ``nvflare.autofl.report.v1`` schema.

Literature outcomes use the ``literature_event_id`` written by the campaign
runner rather than inferring relationships from ledger position. Baselines are
identified strictly by ``status=baseline``; ``best`` includes only a scored
baseline or ``keep`` row, while a better unretained ``discard`` is reported as
``best_observed``. If a valid plot cannot be produced, the Markdown and JSON
reports are still generated with an explicit plot-availability warning.

As with active Auto-FL, users invoke the skill through their coding agent and
do not run scripts from the installed skill directory themselves.
