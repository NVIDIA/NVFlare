# Tensor Download Benchmark

`tensor_download_bench.py` measures end-to-end PyTorch tensor transfer through
FOBS, `TensorDecomposer`, and the F3 Download Service. It runs the same tensors
in two receiver modes:

- `memory`: reconstruct every received value as a `torch.Tensor`.
- `disk`: write safetensors chunks to local storage and reconstruct lightweight
  lazy references. Only one small validation tensor is materialized.

The reported transfer time starts before sender-side FOBS decomposition and
ends after receiver-side reconstruction, validation, and the reply. Throughput
uses the unique tensor bytes that `TensorDecomposer` sends; tied or aliased
tensors count once, while the logical model size is reported separately. The
summary also reports sender and receiver peak RSS deltas and receiver disk
consumption.

## Full GPT-J 6B benchmark

Use the NVFlare Python environment on both machines. Start the receiver first:

```bash
~/nvflare-env/3.12/bin/python dev_tools/f3/tensor_download_bench.py recv \
    --url tcp://0.0.0.0:8002 \
    --offload-dir /fast/local/nvme
```

Then start the sender on the machine containing the checkpoint:

```bash
~/nvflare-env/3.12/bin/python dev_tools/f3/tensor_download_bench.py send \
    --url tcp://<receiver-host>:8002 \
    --checkpoint /tmp/gpt-j-6b/pytorch_model.bin
```

The default `--modes memory,disk` runs both modes against the same receiver.
Use `--modes memory` or `--modes disk` to run only one mode, and `--repeat N`
for repeated measurements.

The memory-mode receiver must have enough available RAM for the complete model
plus transient serialization buffers. Put `--offload-dir` on fast local storage
for a meaningful disk-offload result.

## Quick smoke test

`--max-bytes` selects the smallest unique tensors up to a binary byte limit.
Aliases of a selected tensor are included without consuming the limit. This
checks both transfer paths without sending the full checkpoint:

```bash
~/nvflare-env/3.12/bin/python dev_tools/f3/tensor_download_bench.py send \
    --url tcp://<receiver-host>:8002 \
    --checkpoint /tmp/gpt-j-6b/pytorch_model.bin \
    --max-bytes 128M
```

Suffixes are binary: `1K=1024`, `1M=1024K`, and `1G=1024M`.

## F3 tuning

Pass the same `comm_config.yml` to both endpoints:

```bash
~/nvflare-env/3.12/bin/python dev_tools/f3/tensor_download_bench.py recv \
    --url tcp://0.0.0.0:8002 \
    --offload-dir /fast/local/nvme \
    --f3-config dev_tools/f3/comm_config.yml

~/nvflare-env/3.12/bin/python dev_tools/f3/tensor_download_bench.py send \
    --url tcp://<receiver-host>:8002 \
    --checkpoint /tmp/gpt-j-6b/pytorch_model.bin \
    --f3-config dev_tools/f3/comm_config.yml
```

The transfer rate is an application-level rate for the full
TensorDecomposer/Download Service path, not a raw network-wire throughput
measurement. Safetensors headers and F3 protocol overhead are not included in
the byte count.
