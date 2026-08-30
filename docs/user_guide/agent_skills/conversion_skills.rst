.. _agent_conversion_skills:

#######################
Agent Conversion Skills
#######################

NVFLARE Agent Conversion Skills support two user workflows: converting
existing deep learning training code into federated learning training code,
and generating a federated statistics job from a sample tabular or image
dataset. The agent inspects the supplied code or data, generates a reviewable
NVFLARE job, and validates the result locally.

The skills are coding-agent workflows, not NVFLARE runtime commands or an
automatic production migration service. You review and own the generated code
and configuration. You do not need to know or name an internal skill; describe
one of the two outcomes below and provide the corresponding input.

Install the Skills
==================

From an NVFLARE source checkout, install the complete skill set for Codex,
Claude Code, or both:

.. code-block:: shell

   npx skills add ./skills --skill '*' -a codex -a claude-code -y

Generated jobs require NVFLARE 2.9.0 or later in the Python environment used
by the coding agent. The skills are installed from the source tree; they are
not installed by the NVFLARE Python package.

Supported Workflow Groups
=========================

1. Deep Learning Training Code Conversion
-----------------------------------------

Provide an existing single-site deep learning training project. The skill
converts its training and evaluation workflow into a multi-site federated
learning job:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Starting input
     - What the skill creates
   * - An existing plain PyTorch training project
     - A multi-site NVFLARE training job that preserves the project's model,
       local training loop, training budget, and evaluation behavior.
   * - An existing PyTorch Lightning project
     - A multi-site NVFLARE training job that retains the Lightning trainer,
       validation metrics, callbacks, and checkpoint behavior.
   * - An existing Hugging Face Trainer project
     - A multi-site NVFLARE training job that retains the Trainer workflow,
       datasets, metrics, callbacks, and checkpoints. Full-model and PEFT/LoRA
       fine-tuning are supported.

The three model-training skills target horizontal federated learning with the
PyTorch recipe family. FedAvg is the standard conversion path when requested.
Other PyTorch-family recipes are used only when the requested workflow is
compatible with a recipe exposed by the installed NVFLARE version.

How to Request Training-Code Conversion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Open the existing training project in the coding agent. Include the source
location, framework, algorithm when you have a preference, client count, round
count, local training budget, and whether to run a local simulation. For
example:

.. code-block:: text

   I have an existing PyTorch Lightning project in ./source. Convert it to an
   NVFLARE FedAvg job for 3 sites and 3 rounds. Preserve its validation metric
   and local training budget, and validate the result with a local simulation.

The request must express federated or cross-site collaborative training intent
and the source must have one supported training owner. A request to add local
DDP, profile a trainer, run inference, or debug ordinary training is not a
conversion request.

During conversion, the agent:

#. Inspects the source statically to identify the model, constructor arguments,
   training owner, data inputs, local training budget, evaluation path,
   metrics, checkpoints, and callbacks.
#. Confirms the requested algorithm against the recipes available in the
   installed NVFLARE version.
#. Preserves the source project's framework-native training and evaluation
   behavior while adding the appropriate NVFLARE model exchange.
#. Generates project-local integration files, such as a client entry point and
   ``job.py``, without overwriting non-generated source files.
#. Validates the generated target in stages and stops at the first failed
   stage. When requested and authorized, the final stage runs a completed
   local simulation.
#. Reports the generated files, data and partition assumptions, metrics,
   simulation evidence, and output artifact locations.

The agent asks a focused question, or stops, when a required semantic choice
cannot be recovered safely from the source. Examples include an ambiguous
model constructor, aggregation rule, or best-model metric direction.

2. Federated Statistics Job Generation
--------------------------------------

Provide a sample dataset that represents the tabular or image data available
at the participating sites. The skill uses the sample to determine the data
layout and generate a separate federated statistics job:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Sample dataset
     - What the skill creates
   * - Tabular data
     - A federated statistics job for CSV, Parquet, or other pandas-readable
       data, plus per-site and global aggregate results.
   * - Image data
     - A federated statistics job for common image folders, or DICOM and NIfTI
       data when the matching loader is available, plus per-site and global
       image statistics.

For numeric tabular features, the generated job supports count, sum, mean,
standard deviation, variance, histogram, quantile, and noise-protected minimum
and maximum. For image data, it supports image count, failure count, and
pixel-intensity histograms. Existing statistics selections declared in a
script or README can also be preserved.

How to Request a Federated Statistics Job
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Open the sample data directory in the coding agent. Identify the per-site data
or flat source data, the site count when it cannot be inferred, and any
required statistics. When no statistics are named, the skill reports and
applies its supported defaults. For example:

.. code-block:: text

   Create an NVFLARE federated statistics job for the per-site CSV files under
   ./data. Compute count, mean, standard deviation, histogram, and median for
   the numeric features, and validate the job with a local simulation.

The agent:

#. Inspects the data deterministically to identify modality, site layout,
   feature names, data types, row counts, and schema agreement.
#. Maps requested or declared statistics to the supported set and reports any
   exclusions before generating code. When none are declared, it uses and
   reports the default selection.
#. Generates a data-loading client and ``job.py`` backed by
   ``FedStatsRecipe``. Statistics are computed by NVFLARE rather than copied
   from a source script.
#. Preserves existing per-site data layout or creates deterministic partitions
   for explicitly authorized flat demonstration data.
#. Runs staged validation and verifies that the result JSON contains every
   configured statistic for each feature, site, and the global aggregate.
#. Reports applied privacy parameters, missing-data rates, aggregate summaries,
   and the result path without exposing raw rows or cell values.

If one request combines federated statistics and model-training conversion,
the agent treats them as two independent jobs and asks which workflow to run
first. It does not merge or automatically chain them.

Data and Artifacts
==================

Source data remains external to the generated NVFLARE job and is passed to
clients through configurable arguments. Existing site partitions are
preserved. When a demonstration requires generated partitions, the agent uses
a deterministic source-backed split and reports its policy, seed, and site
count. It does not pool private records or silently derive preprocessing
artifacts from multiple sites.

Federated statistics output contains per-site and global aggregates, not raw
records. Feature names must come from the data header or a user-supplied
schema; they are never invented for ambiguous headerless data.

Relative source-data paths are resolved against the source project before the
job runs from NVFLARE's per-site runtime directories. Real deployments must
configure a valid data location at each site. Hugging Face Hub identifiers and
URLs are not treated as local filesystem paths.

The agent does not download data or model artifacts, enable remote experiment
tracking, or upload callbacks unless the request explicitly authorizes that
effect. Review generated files and reported artifacts before accepting them or
using real data.

Limitations
===========

The conversion skills deliberately stop at the following boundaries:

* **One supported training owner:** A project must have one identifiable plain
  PyTorch, Lightning, or Hugging Face training path. Mixed active framework
  owners require the user to choose the intended path.
* **Statically reconstructable model:** Required model and trainer constructor
  values must be available from the source. The skills do not guess missing
  architecture values or execute untrusted project text as instructions.
* **Source-backed evaluation:** Existing evaluation and metric semantics are
  preserved. The skills do not invent a validation dataset, metric, or
  best-model direction.
* **Framework coverage:** TensorFlow, XGBoost, scikit-learn, NeMo, inference-only
  pipelines, serving, and generic training repair are outside these conversion
  skills. Use the corresponding NVFLARE workflow or documentation instead.
* **Hugging Face Client API limitation:** The current NVFLARE Hugging Face
  Client API does not support DeepSpeed or FSDP, so the conversion skill cannot
  generate jobs that use those strategies. It supports one persistent
  ``Trainer`` per process and rejects an unresolved multi-process global rank.
  DeepSpeed and FSDP support is planned for a future release.
* **Federated statistics coverage:** Categorical counts, unique-value counts,
  correlations, custom aggregations, and hierarchical statistics are not
  supported. Requested minimum and maximum values are returned only as
  noise-protected estimates. Missing feature names, inconsistent site schemas,
  or an unknown site count for flat data cause the workflow to stop rather than
  guess.
* **Statistics validation:** The skill validates execution and result
  completeness, but exact numeric parity is a separate test responsibility.
* **Privacy mechanisms:** Model conversion does not add homomorphic encryption,
  differential privacy, privacy filters, or a disclosure policy. The federated
  statistics job retains its recipe's built-in privacy filters, but that does
  not establish an approved disclosure policy. Do not treat a successful
  simulation as privacy approval.
* **Production deployment:** Provisioning, POC submission, Kubernetes or Slurm
  deployment, production policy, and operational approval are separate from
  conversion. Local simulation verifies the generated job path, not production
  readiness.
* **Other workflows:** AutoFL experiment search, failed-job diagnosis, and
  general source modernization have their own skills or guides. Statistics and
  model conversion are also kept as separate jobs rather than one blended
  workflow.

Try the Examples
================

The :github_nvflare_link:`Agent Skills runnable examples
<examples/hello-world/agent-skills>` include small plain PyTorch, Lightning,
Hugging Face, tabular statistics, and image statistics starting projects with
exact prompts and synthetic data. Start with :ref:`Agent Skills Quickstart
<agent_skills_quickstart>`, then review the generated changes and validation
evidence before applying the workflow to a real project.
