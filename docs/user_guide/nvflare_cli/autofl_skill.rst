.. _autofl_skill:

#######################
NVFlare Auto-FL Skill
#######################

The NVFlare Auto-FL skill is an agent-assisted workflow for optimizing an
existing NVFlare ``job.py``.  The user entry point is the coding agent skill:
select the NVFlare Auto-FL skill, point it at a job, and state the objective,
environment, and candidate budget.

NVFlare does not add a separate public Auto-FL command family for this workflow.
Instead, NVFlare provides the deterministic import, reviewable
``autofl.yaml`` contract, execution substrate, policy boundaries, artifacts,
and reproducibility evidence.  The agent plans candidate edits and runs them
through existing NVFlare surfaces.

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
  and environment or policy boundaries.

This makes the workflow feel native and reproducible: NVFlare owns the truth of
the campaign settings and execution surfaces; the agent owns exploration within
explicit constraints.

Execution
=========

The skill uses existing NVFlare surfaces after import:

- Simulation jobs run through the job's configured ``SimEnv``.
- POC and production jobs use the standard startup-kit and ``nvflare job``
  submission, wait, download, and inspection commands.
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
