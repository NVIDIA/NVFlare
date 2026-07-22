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

## Standard FedAvg Fast Path

For a normal PyTorch-to-FedAvg conversion, keep the `job.py` recipe construction
small and portable:

```python
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.client.config import ExchangeFormat
from nvflare.recipe.sim_env import SimEnv

model_args = {"input_size": input_size, "num_classes": num_classes}
recipe_model = {"class_path": "model.ModelClass", "args": model_args}

recipe = FedAvgRecipe(
    name=job_name,
    min_clients=num_clients,
    num_rounds=num_rounds,
    model=recipe_model,
    train_script="client.py",
    train_args=train_args,
    server_expected_format=ExchangeFormat.PYTORCH,
    enable_tensor_disk_offload=True,
)

env = SimEnv(num_clients=num_clients, workspace_root=workspace_root)
run = recipe.execute(env)
```

Prefer a recipe model dict with the same constructor arguments used by the
client-side model:

```python
model={"class_path": "model.ModelClass", "args": model_args}
```

Prefer `class_path` at recipe construction time; `path` is the normalized key
used in exported job config, and recipes accept it as an alias. Set
`enable_tensor_disk_offload=True` when the selected recipe exposes it, paired
with `server_expected_format=ExchangeFormat.PYTORCH`, per
`../../nvflare-shared/references/conversion-workflow.md` ("Conversion Defaults") and
`../../nvflare-shared/references/pytorch-model-exchange.md`.

The server-side recipe model and the client-side training model must construct
the same architecture. If the model constructor needs dimensions, class counts,
dropout settings, embedding sizes, or other architecture arguments, pass the
same values on both sides. Prefer a small shared constant, JSON/config file, or
explicit `train_args` values over hard-coded divergent defaults.

Use these portable imports when writing custom Job API code:

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

## Execution Mode (In-Process vs External)

Match the recipe's execution mode to the source project's process model:

- Single-process training — CPU or a single GPU with no distributed launch —
  runs in-process; leave `launch_external_process` unset so the recipe applies
  its own default (the in-process Client API). Do not force it on for
  single-process training.
- Multi-process / multi-GPU evidence — `torch.distributed` / DDP, `torchrun` or
  `torch.distributed.run`, `DistributedDataParallel`, or an explicit user request
  for multi-GPU — needs the external-process executor: set
  `launch_external_process=True`, because distributed workers cannot run inside
  the in-process executor. Also preserve the source launch model by setting the
  recipe's documented launch command or launcher parameter, such as
  `command="torchrun ..."` when the source requires `torchrun` or
  multi-process arguments. Do not rely on the recipe's default external command
  when the source project needs distributed launch arguments.

Confirm the selected recipe exposes `launch_external_process` with
`nvflare recipe show <recipe> --format json` before setting it. For distributed
launches, also confirm the recipe exposes a documented `command`, `launcher`, or
equivalent launch-argument surface that can express the source launch command;
if either surface is missing, report the gap and ask or fail closed rather than
assuming or silently dropping the source launch model. This section is for plain
PyTorch conversions; Lightning conversions follow their own DDP guidance.

## Export Behavior

Export handling is shared across algorithms and frameworks. Follow
`../../nvflare-shared/references/conversion-workflow.md` for `--export`, `--export-dir`, and
local command-line parser behavior.
