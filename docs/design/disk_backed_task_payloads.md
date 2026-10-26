# Disk-backed task-worker payloads

Task execution creates a fresh compute worker for each server assignment while
the resident Client Job (CJ) owns federation I/O. To avoid an extra model-sized
copy at each boundary, task executors reuse received PyTorch tensor files. CJ
passes local file references to workers. Workers map inputs, and CJ maps worker
results. The existing sender serializes one tensor at a time. Payload storage is
shared within a job: a local filesystem for Process execution, or the assigned
job subtree on a durable PVC for Kubernetes.

The [lifecycle contract](task_worker_lifecycle.md) defines physical settlement and
the [receipt contract](task_assignment_admission.md) defines when resubmission
can stop. A received result, a settled worker and a locally committed checkpoint
are separate facts; this storage format does not change their ordering.

- GET_TASK replies decode to pass-through references. Resolve before task-data events:
  eager tensors when a site or job data filter applies, local `_LazyRef`s otherwise.
- Unfiltered CJ input hooks receive local references with metadata and `nbytes`.
  Workers receive mapped tensors.
- CJ result hooks and filters receive mapped tensors; their edits are uploaded.
- FOBS records hold structure and tensor descriptors (file, key, size, SHA256).
  Tensor files sit alongside the record and count toward the aggregate payload limit.
- Hard-link whole single-tensor receive files; otherwise write one tensor at a time.
  The runner releases receive scratch files in `finally`. Attempt files remain
  until terminal receipt; retries verify and remap them. Saved-result readers pin
  the attempt while decoding or mapping so a concurrent receipt cannot remove
  their source paths. Active POSIX mappings remain valid after unlink. Other
  readers that reopen filenames need explicit ownership before payload cleanup.

Handoff serialization is proportional to the largest tensor, plus the sender's
cache. Mapped pages and filesystem cache still grow with model size; callback
writes can create private pages. Per-tensor files consume filesystem inodes.
