# Fixture Source Notes

The `hello-lightning` fixtures are minimized, unconverted PyTorch Lightning
training code modeled on the NVFLARE repository example:

- Source example: `examples/hello-world/hello-lightning`

The fixture intentionally omits real datasets, data download, FLARE integration,
and full job execution details so trigger and behavior evals stay deterministic.
`train.py` and `model.py` represent plain Lightning code before any FLARE
conversion; the agent under evaluation is expected to add the
`flare.patch(trainer)` Client API integration and a `job.py`.
