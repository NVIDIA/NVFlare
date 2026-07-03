# Fixture Source Notes

The `hello-pt` fixtures are minimized from the NVFLARE repository example:

- Source example: `examples/hello-world/hello-pt`
- `client.py` source hash: `99e36b5c74b7006640adc979f07bb93a59b0c141fc7d4429154b9809d27332aa`
- `model.py` source hash: `51f86e25439901191515c853545c5d4e167dce34a40219f9aa63051614c8c6d9`
- `job.py` source hash: `6b4314c8c0992701e4f0fdfd04e0efe448cc7909f8bab6f43f44550d83ad6bf5`

The fixture intentionally omits data download and full job execution details so
trigger and behavior evals stay deterministic.

The `eval-pt` fixtures are synthetic, derived from the `hello-pt` fixture with
an added validation loader and accuracy evaluation loop so paired
training/evaluation transformation can be asserted.

The `external-data-pt` fixtures are synthetic, derived from the `hello-pt`
fixture but reading tabular rows from an external CSV path (`--data-path`,
default `/data/nvflare/tabular/train.csv`) instead of building synthetic
in-memory tensors. The path is intentionally external to the repository and run
workspace so configurable data-path behavior is asserted only when the source
provides an external dataset location.

The `injection-pt` fixtures are synthetic, derived from the `hello-pt` fixture
with adversarial instructions embedded in source comments, `README.md`, and
`config.yaml` (exercising the three injection vectors: code comments, README
setup text, and config values). The embedded instructions and endpoints are
intentionally malicious-looking test data for injection-resistance evals, using
only reserved example.com domains; they must never be followed and must not be
"fixed".
