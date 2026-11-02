# F3 Benchmark Tools

Two programs for measuring F3 streaming and tensor-transfer throughput between
two machines, plus a sample tuning config.

| File | Purpose |
|------|---------|
| `tensor_download_bench.py` | End-to-end PyTorch tensor transfer through FOBS and the F3 Download Service |
| `cellnet_bench.py` | Raw F3 cellnet streaming and baseline raw-TCP ceiling |
| `comm_config.sample.json` | Sample F3 tuning config for high-bandwidth networks |

Start the receiver on the destination machine first; then start the sender.
Use the same Python environment (`nvflare` must be importable) on both hosts.
The cellnet benchmark's RSS sampler reads Linux ``/proc/self/statm``; run it on
Linux when memory reporting is required.

---

## `tensor_download_bench.py`

Measures the full tensor transfer pipeline: FOBS decomposition →
Download Service → receiver reconstruction → validation. Runs two modes
against the same receiver process:

- **memory** — reconstruct every received tensor as a `torch.Tensor` in RAM.
- **disk** — write safetensors chunks to local storage and reconstruct lazy
  references. Only one small validation tensor is materialised.

Throughput is reported over the unique tensor bytes that `TensorDecomposer`
actually transfers (aliased or tied tensors count once). The logical model size
is reported separately. Sender and receiver peak RSS deltas and receiver disk
consumption are included in the summary.

### Quick start

```bash
# Receiver (destination host) — start first
python dev_tools/f3/tensor_download_bench.py recv \
    --url tcp://0.0.0.0:8002 \
    --offload-dir /fast/local/nvme

# Sender (host with the checkpoint)
python dev_tools/f3/tensor_download_bench.py send \
    --url tcp://<receiver-host>:8002 \
    --checkpoint /path/to/pytorch_model.bin
```

`--modes memory,disk` (default) runs both modes. Use `--modes memory` or
`--modes disk` to run only one. `--repeat N` repeats for stable median values.

The memory-mode receiver needs enough RAM for the full model plus transient
serialisation buffers. Put `--offload-dir` on fast local storage for a
meaningful disk result.

### Smoke test with a size limit

`--max-bytes` selects the smallest unique tensors up to a binary byte limit
(aliases of a selected tensor are included at no extra cost). Use this to
verify both transfer paths without sending the full checkpoint:

```bash
python dev_tools/f3/tensor_download_bench.py send \
    --url tcp://<receiver-host>:8002 \
    --checkpoint /path/to/pytorch_model.bin \
    --max-bytes 512M
```

Suffixes are binary: `1K = 1024`, `1M = 1024K`, `1G = 1024M`.

### With F3 tuning

Pass the same `comm_config.sample.json` to both endpoints:

```bash
python dev_tools/f3/tensor_download_bench.py recv \
    --url tcp://0.0.0.0:8002 \
    --offload-dir /fast/local/nvme \
    --f3-config dev_tools/f3/comm_config.sample.json

python dev_tools/f3/tensor_download_bench.py send \
    --url tcp://<receiver-host>:8002 \
    --checkpoint /path/to/pytorch_model.bin \
    --f3-config dev_tools/f3/comm_config.sample.json
```

---

## `cellnet_bench.py`

Measures F3 cellnet streaming throughput and, optionally, the raw TCP ceiling
for the same host pair so you can see how much overhead F3 adds.

### F3 cellnet

```bash
# Receiver
python dev_tools/f3/cellnet_bench.py recv --url tcp://0.0.0.0:8002

# Sender — unreliable (faster, no retransmit)
python dev_tools/f3/cellnet_bench.py send \
    --url tcp://<receiver-host>:8002 --reliable false

# Sender — reliable
python dev_tools/f3/cellnet_bench.py send \
    --url tcp://<receiver-host>:8002 --reliable true
```

Default payload is 10 GiB (`--size-gb 10`). The receiver stays up between
runs, so you can A/B `reliable=true` vs `reliable=false` without restarting.

### With F3 tuning

```bash
python dev_tools/f3/cellnet_bench.py recv \
    --url tcp://0.0.0.0:8002 \
    --f3-config dev_tools/f3/comm_config.sample.json

python dev_tools/f3/cellnet_bench.py send \
    --url tcp://<receiver-host>:8002 \
    --f3-config dev_tools/f3/comm_config.sample.json
```

### Raw TCP ceiling

Measures the Python/TCP throughput limit for the host pair, independent of F3.
Stop the cellnet receiver before reusing the port.

```bash
# Receiver
python dev_tools/f3/cellnet_bench.py recv \
    --transport tcp --url tcp://0.0.0.0:8002

# Sender
python dev_tools/f3/cellnet_bench.py send \
    --transport tcp --url tcp://<receiver-host>:8002
```

Raw TCP mode defaults to a 1 GiB untimed warm-up followed by a 100 GiB
measured payload (about 30 seconds at 25 Gbit/s). Override with
`--size-gb`. The default application buffer is 16 MiB; change it with
`--buffer-mb` on **both** endpoints. Override the nominal link rate used for
utilisation reporting with `--target-gbps`.

---

## `comm_config.sample.json`

Benchmark-selected F3 streaming defaults for a nominal 25 Gbit/s network.
Pass the same file to both endpoints with `--f3-config`. Key defaults:

| Setting | Value |
|---------|-------|
| `streaming_chunk_size` | 1 MiB |
| `streaming_window_size` | 64 MiB (64 chunks) |
| `streaming_ack_interval` | 16 MiB |
| `streaming_retry_max_pending_bytes` | 128 MiB |

The receiver's out-of-sequence chunk limit is left unset so it is derived from
the effective window and chunk sizes. For these benchmark defaults, the derived
limit is 65 chunks.

Byte-size fields accept integer bytes or binary suffixes (`1K`, `1M`, `1G`).

Any native F3 comm_config option can be added. Common flat options:

```json
{
  "max_message_size": "1G",
  "tcp_no_delay": true,
  "allow_adhoc_conns": true,
  "adhoc_conn_scheme": "tcp",
  "internal_conn_scheme": "tcp",
  "backbone_conn_gen": 2,
  "subnet_heartbeat_interval": 30,
  "subnet_trouble_threshold": 90,
  "comm_driver_path": "/path/to/custom/drivers",
  "heartbeat_interval": 60,
  "use_aio_grpc": false,
  "streaming_max_blob_size": "1G",
  "streaming_ack_wait": 300,
  "streaming_read_timeout": 300,
  "streaming_send_timeout": 30.0,
  "streaming_ack_progress_timeout": 60.0,
  "streaming_ack_progress_check_interval": 5.0,
  "streaming_reliable": false,
  "streaming_retry_wait": 5.0,
  "streaming_retry_timeout": 60.0,
  "sfm_send_stall_timeout": 0.0,
  "sfm_close_stalled_connection": false,
  "sfm_send_stall_consecutive_checks": 3
}
```

Nested driver and connection settings are passed through unchanged:

```json
{
  "internal": {"scheme": "tcp", "resources": {"host": "localhost"}},
  "adhoc": {"scheme": "tcp", "resources": {"host": "localhost", "secure": false}},
  "backbone": {"connect_generation": [1, 2]},
  "grpc": {
    "max_workers": 100,
    "options": [["grpc.max_send_message_length", "1G"], ["grpc.max_receive_message_length", "1G"]]
  }
}
```
