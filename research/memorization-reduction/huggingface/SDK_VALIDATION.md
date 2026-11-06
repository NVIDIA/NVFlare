# SDK validation and deployment limits

The runnable `validate.py` checks the current SDK reference scores, TRAP loss and
parameter gradients against independent equations, and aggregate audit results
against a pairwise AUROC calculation. It also checks full/difference reconstruction
and observes a cooperative timeout. `job.py`, `check_coverage.py` and `evaluate.py`
exercise the real federated client filters and final global-checkpoint path.

Representative GPU evidence is being collected for this draft. Do not interpret
the historical 525-fit study as validation of the refactored SDK. The synthetic
workflow is bounded to Pythia-70M, fp32, two clients, two rounds, batch size 2 and
128 tokens. It does not validate arbitrary model sizes, mixed precision, adapters,
distributed training or production enforcement. See [the adoption and operating
contract](README.md#resource-budgets-and-visibility).
