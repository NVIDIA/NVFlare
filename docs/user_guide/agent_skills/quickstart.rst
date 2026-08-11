#######################
Agent Skills Quickstart
#######################

The :github_nvflare_link:`Agent Skills runnable examples
<examples/hello-world/agent-skills>` provide five small, standalone projects for trying
NVFLARE Agent Skills with Codex or Claude Code. Each one includes the starting
source or synthetic data, an exact prompt naming the appropriate skill, the
expected generated artifacts, and a local validation path.

Install all skills from an NVFLARE checkout:

.. code-block:: shell

   npx skills add ./skills --skill '*' -a codex -a claude-code -y

The generated jobs also require NVFLARE 2.9.0 or later in the Python
environment used by the coding agent. Install the package from PyPI or use an
editable NVFLARE checkout.

Then open one example directory in your coding agent and paste its prompt. The
examples cover plain PyTorch, PyTorch Lightning, Hugging Face Trainer, tabular
federated statistics, and image federated statistics. Review the generated
``client.py`` and ``job.py`` before running a simulation or applying a workflow
to real data.
