# Standalone validation

Run from the project directory containing `pyproject.toml`, after installing
`python -m pip install -e '.[dev]'`. No parent source tree is needed. Use a
separate environment for the original and native packages when training outside
the comparison runner, since they have the same distribution name.

```bash
python -m pytest tests/unit_test -v -rs
RUN_NATIVE_SIMULATOR_TESTS=1 python -m pytest tests/integration_test/fast -v -rs
```

Unit cases control external boundaries in one interpreter; they cover native
aggregation/optimizer math, persistence, local training and policy adapters.
The integration cases export and relocate a real FLARE job, remove its source
config, then run the simulator on CPU with two clients for two rounds. They
verify final checkpoints and warm optimizer state for FedAvg and FedAdam.
These cases require local process/socket access and are opt-in so ordinary
unit collection never launches a federation. They are bounded by subprocess
timeouts and need no model/data downloads or GPU.

This research package is not automatically collected by the repository's
`tests/unit_test` CI command. The commands above are its explicit validation
entry points; do not infer CI execution from these folder names. Policy tests
that need real LeRobot require the `[lerobot]` extra and report a visible skip
when it is absent. Test on supported PyTorch >=2.4; the earlier Intel macOS validation used 2.2.2 and excluded the
newer autocast API case plus two real-LeRobot statistics cases.

For deterministic training-curve alignment, use `scripts/compare_training.py`
as described in `NATIVE_HANDOFF.md`. Only that optional tool needs an original
checkout; it is not part of the training runtime.

The relocated project was also copied outside the NVFlare repository, built as
a wheel and installed without an editable source path. With `PYTHONPATH` unset,
186 unit cases and both real simulator cases passed using that installed wheel.
The local unit invocation deselected the unsupported Torch 2.2 autocast case;
the LeRobot statistics module was unavailable. Black, isort and flake8 cover the
whole standalone project's Python files, using its local style configuration.
