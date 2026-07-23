# PyTorch Job Validation Notes

Use `../../nvflare-shared/references/validation-evidence.md` for generic validation status,
commands, blockers, and evidence reporting. Use
`../../nvflare-shared/references/metrics-and-artifact-reporting.md` for final metrics, round
metrics, model artifact paths, and missing-evidence reporting. This file only
covers PyTorch-specific validation checks.

## PyTorch-Specific Validation

- Validate that received `FLModel.params` load into the PyTorch model through
  `load_state_dict` or a deliberate compatible mapping.
- Validate that outbound `FLModel.params` comes from the trained model's
  `state_dict` and does not include optimizer or dataloader objects.
- Confirm that checkpoint loading, metric collection, and device placement still
  work after the Client API conversion.
- Confirm that scalar validation metrics are sent in `FLModel.metrics` so
  aggregation recipes can write server-side metrics artifacts.
- Confirm that the recipe's `key_metric` exactly matches one key sent in
  `FLModel.metrics`; warnings such as a validation metric missing from site
  metrics mean best-model selection is not wired correctly.
- Confirm that the selected `key_metric` is higher-is-better; for loss-like
  metrics, the client should send a negated scalar such as `neg_loss`.
- For data-prep changes, confirm the PyTorch `Dataset` or `DataLoader` receives
  the generated per-site path or arguments rather than hard-coded global paths.
- Report PyTorch-specific blockers such as non-serializable model state,
  checkpoint keys that do not match the converted model, transforms that depend
  on unavailable data, or metrics that cannot be serialized into `FLModel`.
