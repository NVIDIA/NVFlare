# Focused example checks

Install `.[dev]` and run from this project directory:

```bash
python -m pytest tests/unit_test -q
RUN_NATIVE_SIMULATOR_TESTS=1 python -m pytest tests/integration_test/fast -q
```

Units cover generated policy recipes, disjoint data shares, GR00T sampler
reproducibility and FLARE adapter numerics/persistence.
Integration tests export jobs with `job.py` and run relocated CPU jobs through
`nvflare simulator`, with external-process FedAvg and in-process FedAdam, checking final checkpoints and warm optimizer state. They need local
socket/process access and use bounded subprocess cleanup. These commands are
explicit validation entry points; repository unit CI does not collect this
research folder automatically. GPU runs are recorded in the main README.
