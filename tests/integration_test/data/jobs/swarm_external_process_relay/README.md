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

Run it through the secure external-process integration project:

```bash
cd tests/integration_test
NVFLARE_TEST_FRAMEWORK=ext_process_streaming pytest -s system_test.py
```

The NumPy model is synthetic and CPU-only. A positive
`np_download_chunk_size` enables DownloadService-backed array encoding; the
trainer returns a 4 MiB array so the transfer uses several chunks.
