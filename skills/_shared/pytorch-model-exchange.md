# PyTorch-Family Model Exchange

Use this reference only for PyTorch-family skills, including plain PyTorch and
PyTorch Lightning. Do not load it from TensorFlow, sklearn, XGBoost, or other
non-PyTorch framework skills.

## Tensor Payload Rule

For `PTInProcessClientAPIExecutor`, outbound `FLModel(params=...)` must contain
`torch.Tensor` values from the trained model state. Do not convert outbound
weights to NumPy before sending. `PTSendParamsConverter` excludes non-tensor
params.

Plain PyTorch Client API code should use a pattern equivalent to:

```python
params = {k: v.detach().cpu() for k, v in model.state_dict().items()}
assert all(isinstance(v, torch.Tensor) for v in params.values())
flare.send(flare.FLModel(params=params, metrics=metrics, meta=meta))
```

## State-Dict Compatibility

The server-side initial model and the site-side training model must have
compatible constructor arguments, state-dict keys, and tensor shapes. If the
model constructor needs values such as input dimension, vocabulary size, number
of classes, hidden size, or dropout, make those values explicit in both the
server recipe/job config and the client model construction path.

Treat state-dict key or tensor-shape mismatches as conversion bugs. Do not
change the user model architecture to hide the mismatch without user approval.

## Scope

This reference covers PyTorch-family tensor/state-dict exchange only. Framework
skills still own their training-loop pattern:

- plain PyTorch owns manual `nvflare.client` receive/load/train/send code;
- PyTorch Lightning owns patched `Trainer` and callback-driven model exchange.
