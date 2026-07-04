# PyTorch Client API Conversion

This reference covers standard PyTorch training loops that already have a
`torch.nn.Module`, optimizer, data loaders, and metrics. Load
`../../nvflare-shared/references/pytorch-model-exchange.md` for PyTorch-family state-dict and
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

HE is not supported (steps 4–5): homomorphic-encryption recipes reject `SimEnv`
and require provisioned `PocEnv`/`ProdEnv`, which are outside conversion scope.
Follow the HE-not-supported rule in
`../../nvflare-shared/references/pytorch-family-recipe-selection.md`: report HE
as unsupported, route it to provisioning/deployment, and ask or fail closed
instead of generating or running an HE `job.py`.

Follow the shared Source Of Truth Boundary in
`../../nvflare-shared/references/conversion-workflow.md`.

## Conversion Pattern

- Import `nvflare.client as flare`.
- Build the model, optimizer, loss, and data loaders once before the loop, not
  inside it, per the shared "Setup Outside The Round Loop" rule in
  `../../nvflare-shared/references/conversion-workflow.md`.
- Call `flare.init()` before setup hooks that need Client API context, such as
  `flare.get_config()` or `flare.get_site_name()`, while still keeping setup
  outside the federated round loop.
- Loop while `flare.is_running()`.
- Call `flare.receive()` to get the incoming `FLModel`.
- Load `input_model.params` into the PyTorch model with `load_state_dict`.
- Train or evaluate using the user's existing data loader and optimizer.
- Send the trained weights without mutating the in-place model:
  `params = {k: v.detach().cpu() for k, v in model.state_dict().items()}` then
  `flare.send(flare.FLModel(params=params, metrics=..., meta=...))`. Do not call
  `model.cpu()`, which moves the persistent model off the training device.

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

Use the canonical FLARE source layout defined in
`../../nvflare-shared/references/conversion-workflow.md` ("Generated Job Layout"):
`client.py`, `model.py`, `job.py`, and the optional `aggregators.py`, data-setup,
and requirements files. Avoid ad hoc entry-point names such as `fl_train.py`
unless the user explicitly requests that naming, and use
`../../nvflare-shared/references/runtime-output-guidance.md` for runtime
workspaces, exported job directories, and validation output locations.

For standard FedAvg, package shared generated files for all clients. Do not
replace all-client deployment with explicit per-site deployment unless the
conversion has real per-site differences such as different scripts, arguments,
data-split settings, or launch behavior.

## Model Construction Consistency

Follow the shared model-config and construction-consistency rule in
`../../nvflare-shared/references/conversion-workflow.md` ("Recipe Model Config"):
same class and constructor args on server and client, explicit
`{"class_path": ..., "args": ...}` config (no live `nn.Module` instance), and
derive-or-ask/fail-closed for required values.

PyTorch-specific delta: the client loads `input_model.params` into the model
with `load_state_dict`, so the server-initial model and the client model must
have matching state-dict shapes (same parameter names and tensor shapes), not
only matching constructor args.

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
form ships at `../assets/client_with_eval.py`; adapt it rather than inventing a
new structure. It includes a setup hook for optimizer, loss, scheduler, and
data-loader state so generated code has a concrete pre-loop setup location.

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

model = model_factory()
model.to(device)

flare.init()
train_state = train_setup_factory(model, device)
val_loader = build_val_loader()

while flare.is_running():
    input_model = flare.receive()
    model.load_state_dict(input_model.params)

    # evaluate the received global model first so the server can do model selection
    global_metric = evaluate(model, val_loader, device)

    if flare.is_evaluate():
        flare.send(flare.FLModel(metrics={metric_name: global_metric}))
        continue

    train_one_round(model, train_state)

    params = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    flare.send(flare.FLModel(params=params, metrics={metric_name: global_metric}))
```

The round `FLModel.metrics` is this pre-training evaluation of the received
global model, not a post-training metric — see
`../../nvflare-shared/references/metrics-and-artifact-reporting.md`
("Received-Model Metric Ownership").

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
  `../../nvflare-shared/references/metrics-and-artifact-reporting.md` for generic final metrics,
  round metrics, model artifact paths, and missing-evidence reporting.

## Job Pattern Reference

Load `recipe-selection.md` before creating or updating `job.py` so the selected
recipe matches the user's requested FL workflow. Do not assume NVFLARE
repository examples are available in the user's environment.
