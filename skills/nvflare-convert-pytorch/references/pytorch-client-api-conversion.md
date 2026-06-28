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

Do not read NVFLARE SDK source or docstrings to choose an alternate PyTorch
exchange path. If a public import, recipe metadata, or validation command shows
the installed NVFLARE version cannot support this path, stop and report the
exact version mismatch or skill/reference gap instead of switching to a
source-discovered strategy.

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

Do not rely on exporting a live `nn.Module` instance when the model constructor
has required arguments. Derive required constructor values from the source code,
dataset metadata, vocab/config generation, checkpoint metadata, or CLI args
before writing `job.py`, then pass them explicitly through the recipe model
config and the client model construction path.

Acceptable patterns include:

- a shared `model_args` dict imported by both `job.py` and `client.py`;
- an explicit recipe model config such as
  `{"path": "model.ModelClass", "args": model_args}` or
  `{"class_path": "model.ModelClass", "args": model_args}`;
- a small JSON/config file read by both sides;
- explicit CLI arguments passed through recipe `train_args` and parsed by
  `client.py`, with the same values used in `job.py`.

Before simulation, validate the generated model construction path when possible
by instantiating the server-side and client-side model with the same arguments
and checking that `load_state_dict` can accept the initial parameters. Treat a
state-dict key or tensor-shape mismatch as a conversion bug, not as a reason to
change the model architecture without user approval.

## Evaluation Branch

When the task is evaluation-only, use `flare.is_evaluate()` to send metrics
without local training.

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
