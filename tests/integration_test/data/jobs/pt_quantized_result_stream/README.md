# Quantized PyTorch Result Streaming

This one-round job exercises the regression path where an in-process Client API
trainer returns PyTorch tensors, `ModelQuantizer` converts the client result to
`float16`, and the server decodes the streamed FOBS payload before
`ModelDequantizer` processes it.

Run it with:

```bash
nvflare simulator tests/integration_test/data/jobs/pt_quantized_result_stream \
  -w /tmp/nvflare/pt_quantized_result_stream \
  -n 2 \
  -t 2
```

The job passes when both client contributions are accepted and round 0 finishes.
Before the decomposer-registration fix, the server exits while decoding
`submit_update` with `missing 'data' property from the recompose data`.
