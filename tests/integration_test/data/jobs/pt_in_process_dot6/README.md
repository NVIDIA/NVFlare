# In-Process PyTorch Tensor Download

This one-round job exercises the inbound half of the in-process framework
decomposer regression. The server sends a PyTorch model through the streamed
FOBS path as `TENSOR_DOWNLOAD` (Datum Object Type 6), and the generic
`ClientAPIExecutor` must register `TensorDecomposer` before receiving the task.

Run it with:

```bash
nvflare simulator tests/integration_test/data/jobs/pt_in_process_dot6 \
  -w /tmp/nvflare/pt_in_process_dot6 \
  -n 2 \
  -t 2
```

The job passes when both clients receive the PyTorch task, return their updates,
and round 0 finishes. Before the decomposer-registration fix, clients repeatedly
fail `get_task` with `cannot find handler for Datum Object Type 6`.
