# Hugging Face Trainer Conversion

## Standard Transformation

Keep normal Trainer construction intact and add one integration point:

```python
import nvflare.client.hf as flare

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

flare.init()
flare.patch(trainer)

while flare.is_running():
    trainer.evaluate()
    trainer.train()
```

The generated `client.py` entry point is FL-only: it always reaches
`flare.init()` and `flare.patch(trainer)`. Do not infer FL launch from
`CLIENT_API_TYPE` or another environment variable; the launcher does not expose
a reliable branch signal to the trainer. If preserving a standalone CLI is
required, factor shared setup into an explicit function parameter and have the
generated client call that function with `federated=True`; keep the standalone
path behind an entry point that passes `federated=False` explicitly.

When the source has no valid evaluation dataset or metric and neither
per-round evaluation nor best-model selection is requested, use a train-only
loop and set `key_metric=""` in the recipe. An empty key omits the automatic
model selector; it is not a workaround for a required lower-is-better metric:

```python
while flare.is_running():
    trainer.train()
```

Call `flare.init(rank=rank)` explicitly before `flare.get_site_name()`,
`flare.get_config()`, or other Client API context access that occurs before
`patch()`. The patch initializes the Client API when no earlier context access
is needed.

Do not add manual model loading from `flare.receive()` or model sending through
`flare.send()`. The patch wraps `train()` and `evaluate()`, loads the received
global parameters, and sends the result from rank 0.

## Preserve Trainer Semantics

- Construct the model, tokenizer/processor, datasets, collator, Trainer
  arguments, callbacks, and Trainer once before the federated loop.
- Preserve `compute_metrics` and evaluation datasets.
- Keep mid-training evaluation controlled by the Trainer's own evaluation
  strategy; it is not a separate FL task.
- Resolve exactly one per-round budget and encode it in one place:
  1. user-requested local steps become generated `TrainingArguments.max_steps`;
  2. user-requested local epochs become generated
     `TrainingArguments.num_train_epochs`, with a conflicting positive
     `max_steps` cleared because Hugging Face gives steps precedence;
  3. when the user explicitly asks to preserve the source budget, leave its
     positive `max_steps` or `num_train_epochs` unchanged;
  4. otherwise use the bounded generated default `max_steps=10` and report it.
- Let `flare.patch(trainer)` infer that Trainer budget. Do not also pass the same
  value through patch `local_steps` or `local_epochs`; that duplicate expression
  is misleading even though the patch option owns the runtime budget. Use a
  patch budget option only when the Trainer's argument surface cannot express
  the requirement, and then do not encode the same budget in Trainer arguments.
  A length-less iterable dataset requires a step budget.
- With `restore_state=True`, the patch sets the Trainer's cumulative
  `max_steps` target to `per_round_budget_steps * total_rounds` so optimizer and
  scheduler progress continues across rounds. Hugging Face may display that
  whole-job target in its progress bar, while the NVFLARE callback still stops
  each `trainer.train()` call after one per-round budget. Do not interpret the
  progress-bar denominator as work for the current round.
- Keep local-only callbacks and reporting. Leave network trackers disabled
  during validation unless explicitly requested.

## Recipe Integration

Use the PyTorch recipe family. Run `nvflare recipe show <recipe-name> --format
json`, then load
`../../nvflare-shared/references/pytorch-family-recipe-construction.md` before
constructing the recipe. That shared reference owns capability checks,
tensor-native transport and decomposer registration, disk offload, external
process selection, and common metric-selection policy.

For FedAvg, start with an explicit importable model configuration:

```python
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe

recipe = FedAvgRecipe(
    name="hf-trainer",
    model={
        "class_path": "model.ServerModel",
        "args": {"model_name_or_path": model_name_or_path},
    },
    min_clients=n_clients,
    num_rounds=num_rounds,
    train_script="client.py",
    train_args=train_args,
)
```

Add optional recipe arguments and decomposers only as directed by the selected
recipe's capability profile and the shared construction reference. Do not copy
the FedAvg constructor shape to another recipe.
Follow the shared construction reference's client-argument transport rule.
`train_args` is not necessarily shell parsed: use unquoted whitespace-free
tokens for the default in-process executor, and use shell quoting only for a
documented POSIX-tokenizing launcher. Ask or fail closed when a required value
cannot be represented by the selected public argument surface. Do not probe
internal command-splitting helpers.

The file named by `train_script` is already the primary client script. Do not
also add that same file through `recipe.add_client_file(...)`; reserve
`add_client_file()` for auxiliary imported modules. Export and inspect the job
before simulation. Reject absolute `task_script_path` values in generated
configs because exported apps must launch their packaged client script
portably.

Exported app folders are target-specific, and the layout depends on the recipe
configuration. Before asserting paths, inspect the exported job root and
enumerate the app directories it actually contains. A standard unified export
uses `app/custom`; a per-site export created through `set_per_site_config()`
uses `app_server/custom` plus each `app_<site>/custom`.

If a generated or project-local server model module is referenced from `job.py` through
`{"class_path": "model.ServerModel"}`, add it to the server app with
`recipe.add_server_file("model.py")` or the equivalent server-targeted API. A
client import is not enough when an export separates server and client apps.
Package client-used local
modules through `train_script`'s import closure or `recipe.add_client_file(...)`,
then verify the required files under the discovered layout: `app/custom` for a
unified export, or `app_server/custom` and each `app_<site>/custom` for a
per-site export. Do not reuse a path assumption from another export. Otherwise,
the server persistor will fail to construct the initial model. Installed
NVFLARE, framework, and third-party class paths stay runtime dependencies
validated through requirements installation plus import/preflight checks.

## Data And Model Selection

Follow the site-partitioning requirement in `SKILL.md`. Pass data roots through
client arguments or per-site configuration; never copy private site data into
the job.

Prefer preserving source metric names in the client metrics output. If the
generated evaluation call emits `accuracy`, configure `key_metric="accuracy"`.
Trainer commonly prefixes `compute_metrics` output, so `{"accuracy": ...}` can
become `eval_accuracy`; when that is the returned client key, configure the
server with `key_metric="eval_accuracy"` and report the mapping from source
metric name to server metric key.

`FedAvgRecipe` does not expose a lower-is-better direction flag. When a
source-backed lower-is-better metric is returned by `compute_metrics`, preserve
it and add an explicitly negated companion such as
`{"wer": wer, "neg_wer": -wer}`, then select the prefixed key
`eval_neg_wer`. If only Trainer-generated `eval_loss` exists and best-model
selection is required, ask for a source-backed selection metric or fail closed;
raw Trainer loss does not give the conversion a safe source-backed negation
hook. Use `key_metric=""` only when best-model selection is not requested. Never
select raw loss as though increasing values were improvements.
