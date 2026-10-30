# Focused example checks

Install `.[dev]` and run from this project directory:

```bash
python -m pytest tests/unit_test -q
RUN_NATIVE_SIMULATOR_TESTS=1 python -m pytest tests/integration_test/fast -q
```

Units cover generated policy recipes, disjoint data shares, GR00T sampler
reproducibility, FLARE adapter numerics/persistence and comparison failures. Integration tests run
exported, relocated CPU jobs with external-process FedAvg and in-process
FedAdam, checking final checkpoints and warm optimizer state. They need local
socket/process access and use bounded subprocess cleanup. These commands are
explicit validation entry points; repository unit CI does not collect this
research folder automatically. Full VLA/GPU validation belongs to the authors.
