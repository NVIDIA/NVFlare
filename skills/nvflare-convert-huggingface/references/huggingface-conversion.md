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

flare.patch(trainer)

while flare.is_running():
    trainer.evaluate()
    trainer.train()
```

When the source has no valid evaluation dataset or metric and model selection
is not requested, use a train-only loop and set `key_metric=""` in the recipe:

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
- Preserve `compute_metrics` and evaluation datasets. Call `evaluate()` before
  `train()` when the server needs metrics for the received global model.
- Keep mid-training evaluation controlled by the Trainer's own evaluation
  strategy; it is not a separate FL task.
- Preserve the source local budget. By default, the patch uses positive
  `max_steps`; otherwise it converts `num_train_epochs` from the real
  dataloader. Use one explicit `local_steps` or `local_epochs` only when the user
  requests a different per-round budget. A length-less iterable dataset requires
  explicit `local_steps`.
- Keep local-only callbacks and reporting. Leave network trackers disabled
  during validation unless explicitly requested.
- Patch only one Trainer per Python process.

## Recipe Integration

Use the PyTorch recipe family. For FedAvg, use an explicit importable model
configuration and preserve tensor dtypes:

```python
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
    launch_external_process=True,
    server_expected_format=ExchangeFormat.PYTORCH,
    enable_tensor_disk_offload=True,
    key_metric=key_metric,
)
```

Confirm every argument through `nvflare recipe show fedavg-pt --format json`.
Quote every user-controlled path or model identifier included in the
`train_args` command string.

## Data And Model Selection

Pass data roots through client arguments or per-site configuration; never copy
private site data into the job. For simulation from one source dataset, make
deterministic site-local training partitions unless shared training data was
explicitly requested.

Use the exact higher-is-better key returned by `Trainer.evaluate()` for
`key_metric`. Trainer commonly prefixes `compute_metrics` output, so
`{"accuracy": ...}` often becomes `eval_accuracy`; verify the returned key
instead of copying the unprefixed function key. If the Trainer only returns a
lower-is-better loss and the recipe does not expose a lower-is-better selector,
set `key_metric=""` or stop for a semantic decision. Do not silently treat
increasing loss as improvement.
