# Swarm tensor disk-offload integration test

This secure, CPU-only job composes the complete feature path:

- two external-process PyTorch trainers;
- streamed model parameters larger than the 2 MiB tensor threshold;
- disk-backed terminal downloads on the selected aggregation client;
- deterministic aggregation rotation from `site-1` to `site-2`;
- final-model distribution and job-level offload-root cleanup.

Each trainer adds its site number to the received model. Starting from zero,
round 0 aggregates `(1 + 2) / 2 = 1.5`; round 1 aggregates
`(2.5 + 3.5) / 2 = 3.0`. The result validator checks both client checkpoints.

The test-only aggregator records each lazy tensor file before materialization.
After `END_RUN`, the validator verifies that every recorded job-owned root has
been removed.

Run only this integration configuration with:

```bash
cd tests/integration_test
NVFLARE_TEST_FRAMEWORK=client_api \
python -m pytest -s system_test.py -k swarm_tensor_disk_offload
```
