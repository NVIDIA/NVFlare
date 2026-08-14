# Quantized PyTorch Result Streaming (NumPy Server Task)

This one-round job isolates the streamed PyTorch result regression without
manufacturing a NumPy task in a filter.

`PTFileModelPersistor` has `allow_numpy_conversion = true`, so its model
learnable and the task sent by the server contain NumPy arrays. The trainer
converts those arrays to PyTorch as ordinary training code, returns a full
PyTorch model, and `ModelQuantizer` converts that result to `float16` before it
crosses the streamed FOBS path.

The trainer exchange is `raw`, so Client API does not convert the NumPy task.
`server_expected_format` remains `pytorch` because the result path under test is
a PyTorch payload. `VerifyTaskParams` confirms the actual task representation.

This job requires `bitsandbytes`: the quantizer and dequantizer import it even
for `float16`. NVFlare's `app_opt_mac` extra intentionally omits that dependency,
so this job is not supported by the standard macOS installation.

Run it with:

```bash
nvflare simulator \
  tests/integration_test/data/jobs/pt_quantized_result_stream_numpy_server \
  -w /tmp/nvflare/pt_quantized_result_stream_numpy_server \
  -n 2 \
  -t 2
```

The assertion-only filters prove each boundary without changing the payload:

- `VerifyTaskParams` requires the server task to contain NumPy `float32`
  arrays.
- `VerifyQuantizedResult` runs after the client quantizer and again on the
  server before dequantization. It requires PyTorch `float16` tensors plus the
  expected quantization metadata.
- `VerifyDequantizedResult` runs after the server dequantizer. It requires
  restored PyTorch `float32` tensors and verifies that quantization metadata
  was removed.

Successful runs contain `VERIFIED_SERVER_TASK_FORMAT`,
`VERIFIED_QUANTIZED_RESULT`, and `VERIFIED_DEQUANTIZED_RESULT` log messages. Any
mismatch fails the task filter and therefore fails the job.

Without the in-process decomposer registration, the NumPy task still reaches
the trainer. The server-side receiver already has `TensorDecomposer`, so it
enters `recompose()`, but the in-process client sender did not register the
matching decomposer. The server therefore fails while decoding the incomplete
streamed PyTorch `submit_update` with:

```text
TensorDecomposer - ERROR - missing 'data' property from the recompose data
Adapter - CRITICAL - failed to decode streamed submit_update ... FOBS protocol error
```

Inspect the verification evidence with:

```bash
rg 'VERIFIED_(SERVER_TASK_FORMAT|QUANTIZED_RESULT|DEQUANTIZED_RESULT)' \
  /tmp/nvflare/pt_quantized_result_stream_numpy_server/{site-1,site-2,server}/log.txt
```
