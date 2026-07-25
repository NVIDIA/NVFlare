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
small and portable. Run `recipe show fedavg-pt --format json` first and follow
`../../nvflare-shared/references/pytorch-family-recipe-construction.md`; the
current FedAvg profile exposes both tensor-format and disk-offload controls:

```python
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.client.config import ExchangeFormat
from nvflare.recipe.sim_env import SimEnv

model_args = {"input_size": input_size, "num_classes": num_classes}
recipe_model = {"class_path": "model.ModelClass", "args": model_args}
metric_name = "accuracy"  # use a source higher-is-better key, or "neg_loss" for loss-like metrics

recipe = FedAvgRecipe(
    name=job_name,
    min_clients=num_clients,
    num_rounds=num_rounds,
    model=recipe_model,
    train_script="client.py",
    train_args=train_args,
    key_metric=metric_name,
    server_expected_format=ExchangeFormat.PYTORCH,
    enable_tensor_disk_offload=True,
)
recipe.add_decomposers(["nvflare.app_opt.pt.decomposers.TensorDecomposer"])

env = SimEnv(num_clients=num_clients, workspace_root=workspace_root)
run = recipe.execute(env)
```

Prefer a recipe model dict with the same constructor arguments used by the
client-side model:

```python
model={"class_path": "model.ModelClass", "args": model_args}
```

Prefer `class_path` at recipe construction time; `path` is the normalized key
used in exported job config, and recipes accept it as an alias.

The maintained `hello-pt` example uses the in-process recipe defaults and does
not enable tensor disk offload. This conversion profile intentionally differs:
keep tensors native, enable disk offload, and register the PyTorch decomposer at
the recipe boundary:

- `server_expected_format=ExchangeFormat.PYTORCH`
- `enable_tensor_disk_offload=True`
- `recipe.add_decomposers(["nvflare.app_opt.pt.decomposers.TensorDecomposer"])`

`add_decomposers(...)` adds a registration component to both server and client
apps, so `TensorDecomposer` is installed before the first tensor payload is
decoded. This keeps the framework-neutral executor unchanged and supports the
default in-process path. If per-site configuration is needed, call
`set_per_site_config(...)` first because adding decomposers prepares the client
apps.

The server-side recipe model and the client-side training model must construct
the same architecture. If the model constructor needs dimensions, class counts,
dropout settings, embedding sizes, or other architecture arguments, pass the
same values on both sides. Prefer a small shared constant, JSON/config file, or
explicit `train_args` values over hard-coded divergent defaults.

The recipe's `key_metric` must match the metric key sent by `client.py` in
`FLModel.metrics`. Preserve higher-is-better metric names on both sides: if
`client.py` sends `metrics={"f1": f1}`, construct `FedAvgRecipe(...,
key_metric="f1", ...)`. For loss-like metrics, higher values still select the
best model; send a negated scalar such as `metrics={"neg_loss": -loss}` and use
`key_metric="neg_loss"`. Do not rely on the recipe default unless the client
really reports `accuracy`.

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

## Recipe Capabilities And Execution Mode

Follow
`../../nvflare-shared/references/pytorch-family-recipe-construction.md` for
every selected recipe. It owns the capability checks for tensor settings,
decomposer registration, best-model naming, and process-based execution mode;
do not copy the FedAvg constructor above to a non-FedAvg recipe.

## Export Behavior

Export handling is shared across algorithms and frameworks. Follow
`../../nvflare-shared/references/conversion-workflow.md` for `--export`, `--export-dir`, and
local command-line parser behavior.
