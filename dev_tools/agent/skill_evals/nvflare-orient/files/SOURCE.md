# Fixture Notes

These fixtures are synthetic, minimal projects authored for deterministic
routing evaluation. They contain no external data and are not intended to run.

- `ambiguous-project` exposes a project-owned training-session factory without
  enough static evidence to select a framework converter.
- `manual-pytorch` contains an explicit manual PyTorch optimization loop.
- `unresolved-huggingface` constructs a Hugging Face Trainer in a separate
  factory module, leaving the calling entrypoint's Trainer ownership unresolved.
- `dual-trainer` contains separate active Lightning and Hugging Face training
  entrypoints in one project.
- `diagnosis` contains a bounded client failure log for diagnosis routing.
- `web-app` contains a minimal non-FLARE package manifest for the global
  negative.
