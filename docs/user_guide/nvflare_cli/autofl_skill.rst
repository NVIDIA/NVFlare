.. _autofl_skill:

#######################
NVFlare Auto-FL Skill
#######################

The NVFlare Auto-FL skill is an agent-assisted workflow for optimizing an
existing NVFlare ``job.py``.  The user entry point is the coding agent skill:
select the NVFlare Auto-FL skill, point it at a job, and state the objective,
environment, and candidate budget.

The skill source lives in ``skills/nvflare-autofl`` with the other NVFlare-owned
agent skills.  When the general agent skill CLI is available, install it through
the standard ``nvflare agent skills`` workflow for the target coding agent.

NVFlare does not add a separate public Auto-FL command family for this workflow.
Instead, NVFlare provides the deterministic import, reviewable
``autofl.yaml`` contract, execution substrate, policy boundaries, artifacts,
and reproducibility evidence.  The agent chooses hypotheses, edits source,
implements algorithms, and runs candidates through existing NVFlare surfaces.

``autofl.yaml`` is the human-reviewable campaign configuration, not a replacement
for ``job.py`` or for exported NVFlare job folders.  It exposes the editable
Auto-FL settings, fixed-budget constraints, allowed edit paths, objective,
candidate budget, provenance, and unresolved fields.  The original ``job.py``
remains the experiment entry point the skill and agent use to run candidates.

Typical Prompt
==============

.. code-block:: text

   Use the NVFlare Auto-FL skill.
   Optimize ./job.py for validation accuracy in simulation with an
   8-candidate budget.

First Step: Deterministic Import
================================

The skill first imports the job without executing user code:

.. code-block:: shell

   python -m nvflare.app_common.autofl.job_importer ./job.py \
       --metric accuracy \
       --env sim \
       --max-candidates 8 \
       --output autofl.yaml

The importer parses supported Recipe and FedJob patterns with Python AST
inspection.  It extracts campaign-relevant settings into ``autofl.yaml`` and
marks unknown or dynamic fields as unresolved instead of guessing.

Trust Contract
==============

Before editing or running candidates, the skill should show the user three
things from ``autofl.yaml``:

- **Editable**: metric, environment, candidate budget, tunables, artifact
  locations, source hash, and importer version.
- **Unresolved**: dynamic defaults, unsupported Python semantics, missing metric
  sources, unknown data paths, or low-confidence fields.
- **Allowed**: files the agent may edit, fixed-budget fields it must preserve,
  Python modules it may add under the job root, and environment or policy
  boundaries.

This makes the workflow feel native and reproducible: NVFlare owns the truth of
the campaign settings and execution surfaces; the agent owns exploration within
explicit constraints.

Execution
=========

The bundled helper is an internal skill surface, not a public NVFlare command
family.  It first initializes the campaign and baseline:

.. code-block:: shell

   python "$CODEX_HOME/skills/nvflare-autofl/scripts/run_job_campaign.py" \
       initialize ./job.py --metric accuracy --mode max --env sim

For each attempt, the agent supplies a hypothesis and receives an isolated
candidate source directory plus ``candidate_manifest.json``:

.. code-block:: shell

   python "$CODEX_HOME/skills/nvflare-autofl/scripts/run_job_campaign.py" \
       prepare ./job.py --name fedprox-variant \
       --hypothesis "stabilize heterogeneous client updates"

The agent edits that candidate source, including new Python algorithm modules
when useful, and asks the helper to evaluate it:

.. code-block:: shell

   python "$CODEX_HOME/skills/nvflare-autofl/scripts/run_job_campaign.py" \
       evaluate ./job.py --manifest <candidate_manifest.json>

NVFlare computes the source diff and hash, checks allowed paths and detectable
fixed-budget drift, executes the candidate, updates ``results.tsv`` and
``progress.png``, and either retains the new best source or restores the prior
best.  Built-in tunable candidates are available through the helper's
``suggest`` action only as optional seeds; the agent remains free to implement
new algorithms.

The workflow then uses existing NVFlare execution surfaces:

- Simulation jobs run through the job's configured ``SimEnv``.
- POC and production jobs use the standard startup-kit and ``nvflare job``
  submission, wait, download, and inspection commands.  The skill records the
  resulting job ID, artifacts, and metric against the candidate manifest.
- Production execution is allowed when the user requests it, but the skill must
  not bypass normal startup-kit authentication, site policy, or job submission.

Supported First Version
=======================

The first version is intentionally narrow:

- Supported job surfaces: NVFlare Recipe constructors and FedJob-style scripts.
- Supported import fields: objective metric, fixed budget fields, environment,
  train script, allowed edit paths, and common argparse tunables.
- Unsupported or ambiguous custom Python is preserved as unresolved review
  fields.

The default user experience should not require editing ``autofl.yaml``.  Users
review it only when the importer reports unresolved fields or when they want to
override the campaign configuration.

Final Report After Stop
=======================

After a campaign is manually stopped, reaches its explicit cap, or ends at a
hard policy/runtime boundary, select the companion NVFlare Auto-FL Report skill.
It turns the recorded campaign evidence into a reviewable final report without
requiring Git or rerunning candidates:

.. code-block:: text

   Use the NVFlare Auto-FL Report skill.
   Generate the final report for the stopped campaign in ./job.

The skill verifies ``.nvflare/autofl/campaign_state.json``, ``results.tsv``, and
available candidate manifests before finalizing.  A pending candidate must be
finalized or abandoned through the active Auto-FL skill first.  The report
helper attempts to refresh ``progress.png`` and produces:

- ``autofl_final_report.md`` for human review;
- ``autofl_report_summary.json`` for tools and downstream agents;
- a synthesis of every literature checkpoint and the candidates evaluated
  after it;
- best-candidate lineage, inherited code changes, manifests, patch hashes,
  exact commands, artifacts, failures, and reproducibility warnings.

Plotting is optional evidence.  If plotting dependencies are unavailable or
the existing artifact is not a valid PNG, Markdown and JSON are still written,
the plot is omitted from Markdown, and
``artifacts.progress_plot_available=false`` records the degraded state.

The deterministic helper can also be invoked directly by an agent:

.. code-block:: shell

   python "$CODEX_HOME/skills/nvflare-autofl-report/scripts/generate_report.py" \
       <job-dir>

If a process was abruptly interrupted and campaign state still appears active,
the agent must first independently confirm that execution has stopped.  It may
then add ``--confirm-interrupted``.  This records the reporting assertion but
does not mutate campaign state.  It bypasses only stale stop state; it never
bypasses pending state, ``candidate`` ledger rows, or prepared candidate
manifests.

The report distinguishes the imported budget in ``autofl.yaml`` from the exact
arguments that ran.  It warns when the selected candidate changed training
compute or when multiple candidates were selected against a test-like metric.
This keeps the final result useful without overstating the evidence.
The JSON ``best`` field always means a retained baseline or ``keep`` result;
an unretained scored ``discard`` is exposed separately as ``best_observed``.
