# Swarm external-process result relay reproducer

This job exercises the regression introduced by PR #4906 in a secure,
two-client Swarm:

- `site-1` and `site-2` launch their trainers as external processes.
- `site-1` forwards its streamed result to the aggregation client, `site-2`.
- `site-2` also submits its external trainer result locally. This deterministic
  local-aggregation leg invokes `SwarmClientController._resolve_lazy_refs()`,
  which is the exact failing call path.

Without the fix, the local leg fails while re-encoding
`LazyDownloadRef(relay=True)`:

```text
cannot relay LazyDownloadRef because no Cell is available in fobs context
```

## Important: secure mode is required

Do not run this reproducer with Simulator or a non-secure POC. External-process
results only get `relay=True` in secure mode; without it, the affected branch is
not executed and unmodified `main` will pass.

Run the dedicated integration target, which provisions a secure project and
submits only this job:

```bash
cd tests/integration_test
NVFLARE_TEST_FRAMEWORK=swarm_external_process_relay pytest -s system_test.py
```

Expected result when this fixture is run against the unmodified `main`
implementation: the test fails with `FINISHED:EXECUTION_EXCEPTION`, and the
site-2 log contains
`RuntimeError: exception from post_cb _finalize_lazy_batch`.

Expected result with the fix: the test passes.

The NumPy model is synthetic and CPU-only. A positive
`np_download_chunk_size` enables DownloadService-backed array encoding; the
trainer returns a 4 MiB array so the transfer uses several chunks.
