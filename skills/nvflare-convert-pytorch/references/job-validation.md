# PyTorch Job Validation Notes

## PyTorch-Specific Validation

- Validate that received `FLModel.params` load into the PyTorch model through
  `load_state_dict` or a deliberate compatible mapping.
- Validate that outbound `FLModel.params` comes from the trained model's
  `state_dict` and does not include optimizer or dataloader objects.
- Confirm that checkpoint loading, metric collection, and device placement still
  work after the Client API conversion.
- For data-prep changes, confirm the PyTorch `Dataset` or `DataLoader` receives
  the generated per-site path or arguments rather than hard-coded global paths.
- Report PyTorch-specific blockers such as non-serializable model state,
  checkpoint keys that do not match the converted model, transforms that depend
  on unavailable data, or metrics that cannot be serialized into `FLModel`.
