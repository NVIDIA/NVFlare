# PyTorch Client API Conversion

This reference covers standard PyTorch training loops that already have a
`torch.nn.Module`, optimizer, data loaders, and metrics. Load
`../../_shared/pytorch-model-exchange.md` for PyTorch-family state-dict and
tensor payload rules before changing model exchange code.

## Canonical Path

Use this path for plain PyTorch conversion:

1. Confirm PyTorch routing with `nvflare agent inspect`.
2. Select a PyTorch-family recipe with `nvflare recipe list/show`.
3. Generate `client.py` with `nvflare.client` `receive` / `send` and
   `FLModel(params=...)` as the model exchange path.
4. Generate `job.py` that builds the selected recipe and calls
   `recipe.execute(SimEnv(...))`.
5. Validate with `python job.py`, inspect terminal evidence, then export.

Follow the Source Of Truth Boundary in
`../../_shared/conversion-workflow.md`: public checks can stop the skill path;
they cannot license a source-discovered replacement.

## Conversion Pattern

- Import `nvflare.client as flare`.
- Call `flare.init()` before the training loop that participates in FLARE.
- Loop while `flare.is_running()`.
- Call `flare.receive()` to get the incoming `FLModel`.
- Load `input_model.params` into the PyTorch model with `load_state_dict`.
- Train or evaluate using the user's existing data loader and optimizer.
- Send `flare.FLModel(params=model.cpu().state_dict(), metrics=..., meta=...)`
  with `flare.send(...)`.

## PyTorch Parameter Payload Type

For `PTInProcessClientAPIExecutor`, follow the shared PyTorch-family model
exchange guidance: outbound `FLModel(params=...)` must contain `torch.Tensor`
values. `PTSendParamsConverter` excludes non-tensor params.

```python
params = {k: v.detach().cpu() for k, v in model.state_dict().items()}
assert all(isinstance(v, torch.Tensor) for v in params.values())
flare.send(flare.FLModel(params=params, metrics=metrics, meta=meta))
```

Do not convert outbound weights to NumPy before sending.

## Source Layout

For PyTorch conversions, the job source should normally contain:

- `client.py`: FLARE Client API entry point;
- `job.py`: recipe or FedJob builder, simulation entry point, and export entry
  point;
- `model.py`: copied, wrapped, or imported model definition when needed;
- `aggregators.py`: only when the conversion includes custom aggregation (see
  `../../_shared/conversion-workflow.md`, "Custom Aggregation");
- `prepare_data.py` / `download_data.py`: only when the conversion generates
  data setup code;
- `requirements.txt` or a small requirements file only when dependencies differ
  from the source project.

Use `../../_shared/runtime-output-guidance.md` for runtime workspaces, exported
job directories, and validation output locations.

Avoid names such as `fl_train.py` for the generated FLARE Client API entry
point unless the user explicitly requests that naming.

For standard FedAvg, package shared generated files for all clients. Do not
replace all-client deployment with explicit per-site deployment unless the
conversion has real per-site differences such as different scripts, arguments,
data-split settings, or launch behavior.

## Model Construction Consistency

The model created by `job.py` for the server-side initial model and the model
created by `client.py` before `load_state_dict` must have matching constructor
arguments and state-dict shapes. When the original model needs arguments such as
input dimension, vocabulary size, number of classes, hidden size, or dropout,
make those values explicit in both places.

Do not pass a live `nn.Module` instance as the recipe model input; generate the
explicit `{"class_path": ..., "args": ...}` config per
`../../_shared/conversion-workflow.md` ("Recipe Model Config"). Derive required
constructor values from the source code, dataset metadata, vocab/config
generation, checkpoint metadata, or CLI args before writing `job.py`, then pass
them explicitly through the recipe model config and the client model
construction path. If they are not statically clear, ask in interactive mode or
fail closed in unattended mode.

Acceptable patterns include:

- a shared `model_args` dict imported by both `job.py` and `client.py`;
- an explicit recipe model config such as
  `{"class_path": "model.ModelClass", "args": model_args}` (prefer
  `class_path`; `path` is the normalized job-config key);
- a small JSON/config file read by both sides;
- explicit CLI arguments passed through recipe `train_args` and parsed by
  `client.py`, with the same values used in `job.py`.

Before simulation, validate the generated model construction path when possible
by instantiating the server-side and client-side model with the same arguments
and checking that `load_state_dict` can accept the initial parameters. Treat a
state-dict key or tensor-shape mismatch as a conversion bug, not as a reason to
change the model architecture without user approval.

## Paired Evaluation Template

Training and evaluation are a pair: every converted training loop that has
source evaluation evidence must also convert that evaluation, and its metrics
must reach the server through `FLModel.metrics`. Adapt the user's existing
evaluation code into this template. Do not synthesize metric semantics,
validation loaders, label mappings, or averaging denominators from scratch
without source evidence; when evaluation is required but the source has none,
ask in interactive mode or fail closed in unattended mode.

This template is self-contained packaged guidance; do not depend on NVFLARE
repository `examples/` being present in the user's environment. The runnable
form ships at `templates/client_with_eval.py`; adapt it rather than inventing a
new structure.

```python
def evaluate(model, val_loader, device):
    model.eval()
    total, metric_sum = 0, 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # accumulate the source-backed metric; keep the source's
            # metric name and averaging denominator
            metric_sum += source_metric(outputs, labels)
            total += labels.numel()
    if total == 0:
        raise RuntimeError("evaluation data is empty; cannot report metrics")
    return metric_sum / total

input_model = flare.receive()
model.load_state_dict(input_model.params)
model.to(device)

# evaluate the received global model first so the server can do model selection
global_metric = evaluate(model, val_loader, device)

# ... local training on the user's existing loop ...

params = {k: v.detach().cpu() for k, v in model.state_dict().items()}
flare.send(flare.FLModel(params=params, metrics={metric_name: global_metric}))
```

When the task is evaluation-only or cross-site evaluation, use
`flare.is_evaluate()` to send `flare.FLModel(metrics=...)` without local
training and without params.

## Checkpoint Loading Safety

Generated code that loads PyTorch checkpoint files must use safe weight-only
loading: `torch.load(..., weights_only=True)`. A checkpoint that requires full
pickle unpickling or custom executable deserialization is not statically safe;
ask in interactive mode or fail closed in unattended mode instead of loading
it.

## Scope Boundaries

- Keep user model architecture and loss function unless the user asks for a
  change.
- Keep data loading local to the site and do not add code that copies private
  data into generated artifacts.
- For checkpoints, preserve user checkpoint semantics and document what is
  federated versus site-local.
- For metrics, send scalar summaries in the `metrics` field. Use
  `../../_shared/metrics-and-artifact-reporting.md` for generic final metrics,
  round metrics, model artifact paths, and missing-evidence reporting.

## Job Pattern Reference

Load `recipe-selection.md` before creating or updating `job.py` so the selected
recipe matches the user's requested FL workflow. Do not assume NVFLARE
repository examples are available in the user's environment.
