# PyTorch-Family Model Exchange

Use this reference only for PyTorch-family skills, including plain PyTorch and
PyTorch Lightning. Do not load it from TensorFlow, sklearn, XGBoost, or other
non-PyTorch framework skills.

## Tensor Payload Rule

For `PTInProcessClientAPIExecutor`, outbound `FLModel(params=...)` must contain
`torch.Tensor` values from the trained model state. Do not convert outbound
weights to NumPy before sending. `PTSendParamsConverter` excludes non-tensor
params.

The manual `flare.send` snippet below applies only to plain PyTorch, where
client code builds the payload itself:

```python
params = {k: v.detach().cpu() for k, v in model.state_dict().items()}
assert all(isinstance(v, torch.Tensor) for v in params.values())
flare.send(flare.FLModel(params=params, metrics=metrics, meta=meta))
```

For PyTorch Lightning, the patched trainer builds and sends the payload, so do
not write this snippet in Lightning client code. The tensor-not-NumPy rule still
applies to the PyTorch family, but Lightning enforces it through recipe/job
configuration rather than manual client payload construction.

## Exchange Format Recipe Settings

Tensor-preserving exchange is a `job.py`/recipe setting, not something to infer
by reading NVFLARE library source. Confirm the selected recipe parameters with
`nvflare recipe show <recipe-name> --format json`.

If the selected recipe exposes `server_expected_format`, prefer
`ExchangeFormat.PYTORCH` for PyTorch-family tensor-preserving exchange. If the
recipe exposes `params_transfer_type`, choose the mode that matches the user's
intent: `FULL` sends whole tensors; `DIFF` sends tensor differences.

## State-Dict Compatibility

The server-side initial model and the site-side training model must have
compatible constructor arguments, state-dict keys, and tensor shapes. If the
model constructor needs values such as input dimension, vocabulary size, number
of classes, hidden size, or dropout, make those values explicit in both the
server recipe/job config and the client model construction path.

Pay special attention to data-derived arguments, such as a `vocab_size` built
from training data. Pin them to a shared value so the server and every site
construct the same architecture.

For a vocabulary the mapping matters, not just the size: use one shared
vocabulary/tokenizer definition so every token resolves to the same ID at every
site. FedAvg averages embedding rows by position, so a per-site token-to-ID
mapping built independently from local data would silently blend unrelated
tokens even when `vocab_size` matches.

Pin only architecture or state-dict compatibility values this way. Do not treat
training-policy values or label/data-derived loss statistics as model
constructor values that must be globally pinned for exchange compatibility.
Examples include class-imbalance weights, sample weights, batch weights,
thresholds, sampler weights, and optimizer state. These values should be
computed from each site's local training partition or passed as an explicit
user-requested training policy outside the exchanged model state.

Constructing the model the same way on both sides guarantees matching keys and
shapes, so do not read NVFLARE exchange source to determine which subset of keys
is serialized. A state-dict key or tensor-shape mismatch means the server and
site constructions diverged, usually through a missing or data-derived argument.
Treat it as a conversion bug, and do not change the user model architecture to
hide the mismatch without user approval.

## Scope

This reference covers PyTorch-family tensor/state-dict exchange only. Framework
skills still own their training-loop pattern:

- plain PyTorch owns manual `nvflare.client` receive/load/train/send code;
- PyTorch Lightning owns patched `Trainer` and callback-driven model exchange.
