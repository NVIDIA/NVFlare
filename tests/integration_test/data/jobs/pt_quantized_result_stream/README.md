# Quantized PyTorch Result Streaming (PyTorch Server Task)

This one-round job exercises the regression path where an in-process Client API
trainer returns PyTorch tensors, `ModelQuantizer` converts the client result to
`float16`, and the server decodes the streamed FOBS payload before
`ModelDequantizer` processes it.

`PTFileModelPersistor` has `allow_numpy_conversion = false`, so the server sends
native PyTorch tensors. `VerifyTaskParams` only checks that representation; it
does not transform the task.

This job requires `bitsandbytes`: the quantizer and dequantizer import it even
for `float16`. NVFlare's `app_opt_mac` extra intentionally omits that dependency,
so this job is not supported by the standard macOS installation.

Run it with:

```bash
nvflare simulator tests/integration_test/data/jobs/pt_quantized_result_stream \
  -w /tmp/nvflare/pt_quantized_result_stream \
  -n 2 \
  -t 2
```

The job passes when both client contributions are accepted and round 0 finishes.
Without the in-process decomposer registration, this PyTorch-task variant fails
first on the inbound task with the related DOT-6 error. Use the sibling
`pt_quantized_result_stream_numpy_server` job to isolate the outbound streamed
result failure.

The filter chain also verifies that quantization is not merely logged:

- `VerifyQuantizedResult` runs after the client quantizer and again on the
  server before dequantization. It requires `float16` tensors plus the expected
  quantization metadata.
- `VerifyDequantizedResult` runs after the server dequantizer. It requires
  restored `float32` tensors and verifies that quantization metadata was
  removed.
- `VerifyTaskParams` requires the received server task to contain PyTorch
  `float32` tensors.

Successful runs contain `VERIFIED_SERVER_TASK_FORMAT`,
`VERIFIED_QUANTIZED_RESULT`, and `VERIFIED_DEQUANTIZED_RESULT` log messages. Any
mismatch fails the task filter and therefore fails the job.

Inspect the verification evidence with:

```bash
rg 'VERIFIED_(SERVER_TASK_FORMAT|QUANTIZED_RESULT|DEQUANTIZED_RESULT)' \
  /tmp/nvflare/pt_quantized_result_stream/{site-1,site-2,server}/log.txt
```
