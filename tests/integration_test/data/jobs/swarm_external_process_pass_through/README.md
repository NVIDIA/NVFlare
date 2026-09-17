# Swarm external-process pass-through integration test

This secure, two-client Swarm job keeps real nested CCWF coverage for the
external-process Client API:

- `site-1` and `site-2` launch managed external trainers;
- `site-1` sends its large trainer result to aggregation client `site-2`;
- `site-2` submits its own external trainer result locally; and
- the aggregation role therefore exercises both remote and local nested result
  paths.

The synthetic NumPy result is 4 MiB and `np_download_chunk_size` is positive, so
FOBS uses `ViaDownloader`. In secure mode, the test also fails if the launched
trainer does not install the site auth headers received after `HELLO`.

This fixture tests ordinary pass-through only. The original trainer reference
is retained until the aggregation CJ consumes it; no Client API-specific
transfer wrapper is involved. Source/reference identity and receiver accounting
are asserted in focused tests, while this system test gates the provisioned
secure topology, real subprocess, nested Swarm workflow, and job lifecycle.

Run only this scenario with:

```bash
cd tests/integration_test
NVFLARE_TEST_FRAMEWORK=client_api pytest -s system_test.py -k swarm_external_process_pass_through
```
