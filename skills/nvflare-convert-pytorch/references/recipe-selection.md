# PyTorch Recipe Selection

PyTorch identifies the training framework; it does not determine the federated
workflow. Choose the recipe from the user's FL intent and aggregation,
state-exchange, privacy, and site-role requirements.

## Recipe Discovery, Algorithms, And Selection

Recipe discovery, the algorithm guide (FedAvg, FedProx, FedOpt, SCAFFOLD,
Cyclic, Swarm, FedEval), catalog-based selection rules, the HE-not-supported
rule, and non-FedAvg recipe rules are shared across the PyTorch recipe family.
Follow
`../../nvflare-shared/references/pytorch-family-recipe-selection.md` for all of
them before constructing `job.py`. This file covers only the plain-PyTorch
`job.py` construction details.

## Plain-PyTorch Job API Fallback

The shared construction reference owns the standard FedAvg constructor,
required model source, capability-gated options, and execution pattern. Use
these portable imports only when the selected workflow requires custom Job API
code:

```python
from nvflare.job_config.api import FedJob
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.recipe.sim_env import SimEnv
```

Do not inspect large NVFLARE modules to recover these imports unless validation
shows that the installed version differs.

Do not infer that `per_site_config` is required only because recipe metadata
mentions it. For standard FedAvg with the same `client.py`, `model.py`, and
training arguments on all clients, leave `per_site_config` unset and let the
recipe deploy the executor to all clients. Use `per_site_config` only when at
least one site needs a different `train_script`, `train_args`, command,
external-process setting, framework/exchange setting, or launch behavior.

For non-FedAvg workflows, use the matching recipe from the catalog (see the
shared reference above) and keep the PyTorch Client API exchange aligned with
that recipe's expected task names, metadata, and parameter format.

## Recipe Capabilities And Execution Mode

Follow
`../../nvflare-shared/references/pytorch-family-recipe-construction.md` for
every selected recipe. It owns the capability checks for tensor settings,
decomposer registration, best-model naming, and process-based execution mode;
do not copy the shared FedAvg constructor to a non-FedAvg recipe.

## Export Behavior

Export handling is shared across algorithms and frameworks. Follow
`../../nvflare-shared/references/conversion-workflow.md` for `--export`, `--export-dir`, and
local command-line parser behavior.
