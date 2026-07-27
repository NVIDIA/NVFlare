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
- Preserve the source local budget. By default, the patch uses positive
  `max_steps`; otherwise it converts `num_train_epochs` from the real
  dataloader. Use one explicit `local_steps` or `local_epochs` only when the user
  requests a different per-round budget. A length-less iterable dataset requires
  explicit `local_steps`.
- Keep local-only callbacks and reporting. Leave network trackers disabled
  during validation unless explicitly requested.

## Recipe Integration

Use the PyTorch recipe family. For FedAvg, use an explicit importable model
configuration and preserve tensor dtypes:

```python
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.client.config import ExchangeFormat

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
    server_expected_format=ExchangeFormat.PYTORCH,
    enable_tensor_disk_offload=True,
    key_metric=key_metric,
)
```

Confirm every argument through `nvflare recipe show fedavg-pt --format json`.
Leave `launch_external_process` unset for single-process Trainer scripts. Set
`launch_external_process=True` only when distributed/external-launch evidence
requires it and the selected recipe exposes the parameter.
Quote every user-controlled path or model identifier included in the
`train_args` command string.

Exported app folders are target-specific. If the server model module is
referenced from `job.py` through `{"class_path": "model.ServerModel"}`, add it
to the server app with `recipe.add_server_file("model.py")` or the equivalent
server-targeted API. A client import is not enough for per-site exports because
`set_per_site_config()` creates `app_server` separately from each client app.
Package client-used modules through `train_script`'s import closure or
`recipe.add_client_file(...)`, then inspect the export for
`app_server/custom/model.py` and each client app's required files; otherwise the
server persistor will fail to construct the initial model.

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
