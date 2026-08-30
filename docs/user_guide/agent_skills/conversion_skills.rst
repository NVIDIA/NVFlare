.. _agent_conversion_skills:

#######################
Agent Conversion Skills
#######################

NVFLARE Agent Conversion Skills help a coding agent adapt an existing
single-site training project into a reviewable, multi-site NVFLARE job. The
agent inspects the project's existing training and evaluation code, chooses a
compatible NVFLARE recipe, generates the integration files, and validates the
result locally.

The skills are coding-agent workflows, not NVFLARE runtime commands or an
automatic production migration service. You review and own the generated code
and configuration.

Supported Training Frameworks
=============================

The conversion skills support three PyTorch-family training styles:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Starting project
     - Skill
     - Supported integration
   * - Plain PyTorch
     - ``nvflare-convert-pytorch``
     - Manual training loops built with ``nn.Module``, optimizers,
       ``DataLoader``, and ``state_dict`` exchange. The generated client uses
       the NVFLARE Client API and preserves source-backed training and
       evaluation behavior.
   * - PyTorch Lightning
     - ``nvflare-convert-lightning``
     - A Lightning-owned workflow with ``LightningModule`` and ``Trainer``.
       The generated client patches the existing ``Trainer`` and keeps
       evaluation, metrics, callbacks, and checkpoint behavior in Lightning.
   * - Hugging Face
     - ``nvflare-convert-huggingface``
     - ``Trainer``, ``Seq2SeqTrainer``, TRL ``SFTTrainer``, or a compatible
       ``Trainer`` subclass. Full-model and PEFT/LoRA fine-tuning are
       supported, including single-process and replicated distributed
       training when the process rank is unambiguous.

All three skills target horizontal federated learning with the PyTorch recipe
family. FedAvg is the standard conversion path when requested. Other
PyTorch-family recipes are used only when the requested workflow is compatible
with a recipe exposed by the installed NVFLARE version.

Install the Skills
==================

From an NVFLARE source checkout, install the complete skill set for Codex,
Claude Code, or both:

.. code-block:: shell

   npx skills add ./skills --skill '*' -a codex -a claude-code -y

Generated jobs require NVFLARE 2.9.0 or later in the Python environment used
by the coding agent. The skills are installed from the source tree; they are
not installed by the NVFLARE Python package.

Request a Conversion
====================

Open the existing training project in the coding agent and describe the
federated outcome. Include the source location, framework, algorithm when you
have a preference, client count, round count, local training budget, and
whether to run a local simulation. For example:

.. code-block:: text

   I have an existing PyTorch Lightning project in ./source. Convert it to an
   NVFLARE FedAvg job for 3 sites and 3 rounds. Preserve its validation metric
   and local training budget, and validate the result with a local simulation.

You do not need to name a skill. The agent selects a converter only when the
request expresses federated or cross-site collaborative training intent and
the source has one supported training owner. A request to add local DDP,
profile a trainer, run inference, or debug ordinary training is not a
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

Data and Artifacts
==================

Training data remains external to the generated NVFLARE job and is passed to
clients through configurable arguments. Existing site partitions are
preserved. When a demonstration requires generated partitions, the agent uses
a deterministic source-backed split and reports its policy, seed, and site
count. It does not pool private records or silently derive preprocessing
artifacts from multiple sites.

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
* **Hugging Face distributed strategies:** DeepSpeed and FSDP are outside the
  Hugging Face converter's current scope. It supports one persistent
  ``Trainer`` per process and rejects an unresolved multi-process global rank.
* **Privacy mechanisms:** A conversion does not add homomorphic encryption,
  differential privacy, privacy filters, or a disclosure policy. Ask for those
  requirements as a separate security and privacy design step; do not treat a
  successful simulation as privacy approval.
* **Production deployment:** Provisioning, POC submission, Kubernetes or Slurm
  deployment, production policy, and operational approval are separate from
  conversion. Local simulation verifies the generated job path, not production
  readiness.
* **Other workflows:** Federated statistics, AutoFL experiment search, failed-job
  diagnosis, and general source modernization have their own skills or guides.
  A combined request is handled as separate workflows rather than one blended
  job.

Try the Examples
================

The :github_nvflare_link:`Agent Skills runnable examples
<examples/hello-world/agent-skills>` include small plain PyTorch, Lightning, and
Hugging Face starting projects with exact prompts and synthetic data. Start
with :ref:`Agent Skills Quickstart <agent_skills_quickstart>`, then review the
generated changes and validation evidence before applying the workflow to a
real project.
