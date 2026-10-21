# Disk-backed task-worker payloads

[NVIDIA/NVFlare#5462](https://github.com/NVIDIA/NVFlare/issues/5462).

Task-lifetime executors reuse received PyTorch tensor files. The resident Client
Job (CJ) passes local file references to workers. Workers map their inputs, and
the CJ maps worker results. The existing sender serializes one tensor at a time.

- GET_TASK replies decode to pass-through references. Resolve before task-data events:
  eager tensors when a site or job data filter applies, local `_LazyRef`s otherwise.
- Unfiltered CJ input hooks receive local references with metadata and `nbytes`.
  Workers receive mapped tensors.
- CJ result hooks and filters receive mapped tensors; their edits are uploaded.
- FOBS records hold structure and tensor descriptors (file, key, size, SHA256).
  Tensor files sit alongside the record and count toward the aggregate payload limit.
- Hard-link whole single-tensor receive files; otherwise write one tensor at a time.
  The runner releases receive scratch files in `finally`. Attempt files remain
  until terminal receipt; retries remap them and active mappings survive unlink.

Handoff serialization is proportional to the largest tensor, plus the sender's
cache. Mapped pages and filesystem cache still grow with model size; callback
writes can create private pages. Per-tensor files consume filesystem inodes.
