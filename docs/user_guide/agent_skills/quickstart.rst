.. _agent_skills_quickstart:

#######################
Agent Skills Quickstart
#######################

NVFLARE Agent Skills provide guided workflows for understanding NVFLARE,
creating jobs, optimizing an existing job, and reporting AutoFL results. You
describe the goal in natural language; the coding agent selects the appropriate
installed skill from the request and available source files.

.. _agent_skills_install:

Install the Skills
==================

Install all skills from an NVFLARE checkout:

.. code-block:: shell

   npx skills add ./skills --skill '*' -a codex -a claude-code -y

The generated jobs also require NVFLARE 2.9.0 or later in the Python
environment used by the coding agent. Install the package from PyPI or use an
editable NVFLARE checkout.

Choose a Workflow
=================

You do not need to know or name the internal skill. Choose the goal that
matches your task and provide the relevant project, data, job, or evidence.

Understand Where to Start
-------------------------

Use the orientation workflow when you are new to NVFLARE, are unsure which API
or workflow fits a project, or have source code with more than one training
framework or entry point. It inspects the supplied project read-only and
recommends one next workflow without editing files or starting a job.

Create a Federated Job
----------------------

Use the job-creation workflows for either of these goals:

* Convert existing plain PyTorch, PyTorch Lightning, or Hugging Face Trainer
  code into a federated learning training job.
* Given a sample tabular or image dataset, generate a federated statistics job.

See :ref:`Agent Conversion Skills <agent_conversion_skills>` for supported
inputs, example prompts, generated results, validation behavior, and
limitations.

Optimize an Existing Job
------------------------

Use AutoFL when you already have a runnable NVFLARE ``job.py`` and want to
improve a measured objective through reproducible candidate experiments. State
the metric, execution environment (simulation, POC, or production), and any
candidate limit. AutoFL optimizes an existing job; it does not convert
standalone training code.

See :ref:`NVFlare Auto-FL Agent Skill <autofl_skill>` for campaign behavior,
permissions, comparison rules, and outputs.

Report a Completed AutoFL Campaign
----------------------------------

Use AutoFL reporting after a campaign has stopped, reached its candidate cap,
or been explicitly interrupted. It turns the recorded campaign evidence into
a reproducible Markdown report, JSON summary, and progress plot without
changing campaign results.

Try the Runnable Examples
=========================

The :github_nvflare_link:`Agent Skills runnable examples
<examples/hello-world/agent-skills>` provide five small, standalone projects
for trying NVFLARE Agent Skills with Codex or Claude Code. Each includes
starting source or synthetic data, an exact natural-language prompt, and a
local validation path.

Open one example directory in your coding agent and paste its prompt. The
examples cover plain PyTorch, PyTorch Lightning, Hugging Face Trainer, tabular
federated statistics, and image federated statistics. Review the proposed
changes before running a simulation or applying a workflow to real data.
