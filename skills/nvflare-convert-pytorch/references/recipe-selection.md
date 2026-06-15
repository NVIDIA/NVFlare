# PyTorch Recipe Selection

PyTorch identifies the training framework; it does not determine the federated
workflow. Choose the recipe from the user's FL intent, topology, and aggregation
requirements.

## Discover Recipes

Run the local recipe catalog before creating or updating `job.py`:

```bash
nvflare recipe list --framework pytorch --format json
```

Use the returned recipe metadata as the source of truth for recipe names,
modules, classes, algorithms, aggregation mode, state exchange, privacy metadata,
and optional dependencies.

After selecting a candidate recipe, inspect its parameters:

```bash
nvflare recipe show <recipe-name> --format json
```

Prefer recipe CLI metadata before Python introspection. If Python introspection
is still needed, do not guess top-level exports: the base `Recipe` class lives in
`nvflare.recipe.spec`, not `nvflare.recipe`.

## Quick Algorithm Guide

If the user does not know which FL algorithm they need, explain the choices in
plain language before editing `job.py`:

- FedAvg: the default starting point for most horizontal FL jobs. Each site
  trains locally, sends model weights or updates, and the server averages them.
  Use this when the user simply asks to federate normal PyTorch training.
- FedAvg with HE: FedAvg plus homomorphic encryption support for protected
  aggregation. Use only when the user asks for HE or encrypted aggregation.
- FedProx: FedAvg-style training with a proximal term in the client loss to
  improve stability when site data or compute behavior is very different.
- FedOpt: server-side optimizer variants such as FedAdam, FedYogi, or FedAdagrad.
  Use when the user wants server optimizer control or better behavior than plain
  averaging on heterogeneous data.
- SCAFFOLD: adds control variates to reduce client drift on non-IID data. Use
  when the user specifically asks for SCAFFOLD or drift mitigation.
- Cyclic: sends the model through clients sequentially instead of aggregating
  updates on the server. Use when the requested workflow is client-to-client or
  cyclic weight transfer.
- Swarm Learning: peer/client-parent aggregation topology instead of a normal
  server-centered FedAvg topology. Use when the user asks for swarm learning.
- FedEval: evaluation-only. Use when the user wants to distribute a checkpoint
  to sites and collect metrics without federated training updates.

For deeper background, see the algorithm papers for
[FedAvg](https://proceedings.mlr.press/v54/mcmahan17a.html),
[FedProx](https://arxiv.org/abs/1812.06127),
[FedOpt](https://openreview.net/forum?id=LkFG3lB13U5),
[SCAFFOLD](https://proceedings.mlr.press/v119/karimireddy20a.html), and
[Swarm Learning](https://www.nature.com/articles/s41586-021-03583-3). For
Cyclic recipes, use the local catalog and
`nvflare recipe show cyclic-pt --format json`.

## Selection Rules

- Use `fedavg-pt` for standard horizontal federated training where clients train
  the same PyTorch model locally and the server aggregates model weights or
  weight diffs across rounds.
- Use `fedavg-he-pt` when the user asks for FedAvg with homomorphic encryption.
- Use `fedprox-pt` when the user asks for FedProx or proximal loss behavior.
- Use `fedopt-pt` when the user asks for server-side optimizer variants such as
  FedAdam, FedYogi, or FedAdagrad behavior.
- Use `scaffold-pt` when the user asks for SCAFFOLD-style control variates or
  client-drift mitigation.
- Use `cyclic-pt` when the user asks for sequential client-to-client model
  transfer rather than server aggregation.
- Use `swarm-pt` when the user asks for swarm learning or peer/client-parent
  aggregation topology.
- Use `fedeval-pt` for evaluation-only jobs that send a checkpoint to sites and
  collect metrics without local training updates.
- Ask the user before choosing when the requested FL workflow is not clear.

## Standard FedAvg Fast Path

For a normal PyTorch-to-FedAvg conversion, keep the `job.py` recipe construction
small and portable:

```python
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe.sim_env import SimEnv

model_args = {"input_size": input_size, "num_classes": num_classes}
recipe_model = {"path": "model.ModelClass", "args": model_args}

recipe = FedAvgRecipe(
    name=job_name,
    min_clients=num_clients,
    num_rounds=num_rounds,
    model=recipe_model,
    train_script="client.py",
    train_args=train_args,
)

env = SimEnv(num_clients=num_clients, workspace_root=workspace_root)
run = recipe.execute(env)
```

Prefer a recipe model dict with the same constructor arguments used by the
client-side model:

```python
model={"path": "model.ModelClass", "args": model_args}
```

The exported job/config uses `path`. If a local recipe API still requires
`class_path`, use that key at recipe construction time, but do not prefer it
when `path` is accepted.
Some export paths serialize only the class path if given a live model instance,
which can drop required constructor values. Use explicit model config with
`path` or `class_path` plus `args`, then verify export preserves the required
model arguments.

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

For non-FedAvg workflows, use the matching recipe from the catalog and keep the
PyTorch Client API exchange aligned with that recipe's expected task names,
metadata, and parameter format.

## Non-FedAvg Recipe Rules

The FedAvg fast path is not a universal PyTorch job template. When the user asks
for FedOpt, FedProx, SCAFFOLD, Cyclic, Swarm Learning, FedEval, encryption, or a
topology-specific workflow:

- use `nvflare recipe show <recipe-name> --format json` for the selected recipe;
- supply parameters marked `"required": true`;
- leave optional parameters at defaults unless the user request, source code, or
  validation result requires them;
- keep generated source names and runtime locations consistent with this skill;
- keep shared generated files on all clients unless the recipe semantics or user
  request require site-specific roles, scripts, arguments, or launch settings;
- ask before choosing when recipe intent or topology is ambiguous.

For recipes with topology roles, such as cyclic ordering, swarm roles, vertical
data ownership, or label-owner style workflows, do not collapse the topology into
standard FedAvg. Preserve the selected recipe's required role or site parameters
and report any missing user input before validation.

## Export Behavior

Export handling is shared across algorithms and frameworks. Follow
`../../_shared/nvflare-job-lifecycle.md` for `--export`, `--export-dir`, and
local command-line parser behavior.
