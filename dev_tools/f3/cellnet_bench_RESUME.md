# cellnet_bench resume notes

Goal: A/B compare `reliable=true` vs `reliable=false` for
`dev_tools/f3/cellnet_bench.py send`,
5 runs each, alternating so any network drift hits both modes equally.

## Setup
- Receiver should already be running at `tcp://64.247.196.155:8002` (started via
  `python dev_tools/f3/cellnet_bench.py recv --url tcp://0.0.0.0:8002`). If not,
  start it first.
- On the sender machine, make sure `dev_tools/f3/cellnet_bench.py` is present
  and `nvflare` is importable (`python3 -m pip install -e .[dev]` or `.[dev_mac]`).

## Optional F3 tuning

The bundled [`comm_config.yml`](comm_config.yml) is a native F3 configuration
with the benchmark-selected defaults for a 25 Gbit/s network:

- 1 MiB streaming chunks
- 64 MiB streaming window
- 16 MiB acknowledgment interval
- up to 16 out-of-sequence chunks
- 128 MiB reliable-retry pending-data limit

Pass the same file to both endpoints so sender and receiver use compatible
streaming parameters:

```
python dev_tools/f3/cellnet_bench.py recv \
  --url tcp://0.0.0.0:8002 \
  --f3-config dev_tools/f3/comm_config.yml

python dev_tools/f3/cellnet_bench.py send \
  --url tcp://64.247.196.155:8002 \
  --f3-config dev_tools/f3/comm_config.yml
```

`--f3-config` (also available as `--config`) accepts the native flat and nested
options consumed by `CommConfigurator`, so it can tune streaming, retry,
heartbeat, connection, backbone, and gRPC settings. The file must be named
`comm_config.yml` or `comm_config.yaml`, matching F3's configuration discovery
convention. The benchmark validates that chunk, window, and acknowledgment
sizes are positive, that the window is at least one chunk, and that the
acknowledgment interval does not exceed the window. Explicit command-line
options such as `--reliable` override the corresponding YAML setting.

F3 byte-count fields accept integer bytes or binary unit suffixes. Suffixes are
case-insensitive and use these exact multipliers:

- `1K = 1024` bytes
- `1M = 1024K`
- `1G = 1024M`
- `1T = 1024G`

The explicit forms `KiB`, `MiB`, and `GiB` are also accepted. For example,
`128K`, `16M`, and `2G` resolve to integer byte counts before F3 is initialized.
This applies to message, chunk, blob, window, acknowledgment, retry-buffer, and
gRPC send/receive message-length settings.

## Raw TCP ceiling (no F3)

Stop the Cellnet receiver before reusing its port, then start the optimized raw
TCP receiver:

```
python dev_tools/f3/cellnet_bench.py recv --transport tcp --url tcp://0.0.0.0:8002
```

Run the sender. Raw TCP mode defaults to a 1 GiB untimed warm-up and a 100 GiB
measured payload so a 25 Gbit/s link is measured for roughly 30 seconds:

```
python dev_tools/f3/cellnet_bench.py send --transport tcp --url tcp://64.247.196.155:8002
```

The default application buffer is 16 MiB. Override it on both endpoints with
`--buffer-mb` when testing another size. Kernel send and receive buffers use OS
TCP autotuning by default; this is important on Linux because setting
`SO_SNDBUF` or `SO_RCVBUF` manually can disable autotuning. Use
`--socket-buffer-mb` on both endpoints only when the host networking sysctls
have been tuned to permit the requested value.

Raw TCP mode preallocates and reuses the buffer, uses `recv_into()` to avoid
per-read allocations, and waits for the receiver to drain all bytes before
stopping the sender timer. The `TCP_BASELINE_NO_F3` line reports machine-readable
byte count, elapsed time, MiB/s, Gbit/s, and utilization of the nominal 25
Gbit/s link; use it as the raw Python/TCP ceiling for that host, network, and
buffer configuration. Override the nominal rate with `--target-gbps`.

## Run order (10 runs, alternating)

```
python dev_tools/f3/cellnet_bench.py send --url tcp://64.247.196.155:8002 --reliable true   # pair 1
python dev_tools/f3/cellnet_bench.py send --url tcp://64.247.196.155:8002 --reliable false  # pair 1
python dev_tools/f3/cellnet_bench.py send --url tcp://64.247.196.155:8002 --reliable true   # pair 2
python dev_tools/f3/cellnet_bench.py send --url tcp://64.247.196.155:8002 --reliable false  # pair 2
... repeat through pair 5
```

Or as a one-liner that logs each run and prints the RESULT line at the end:

```
mkdir -p bench_logs
for i in 1 2 3 4 5; do
  for r in true false; do
    log="bench_logs/run${i}_reliable_${r}.log"
    echo "=== pair $i reliable=$r ==="
    python dev_tools/f3/cellnet_bench.py send --url tcp://64.247.196.155:8002 --reliable $r 2>&1 | tee "$log"
  done
done
grep -H "RESULT" bench_logs/*.log
```

## What to record
Each run prints a line like:

```
[send] RESULT reliable=true: sent 10,737,418,240 bytes in 87.42 seconds (117.6 MB/s)
```

Collect the `seconds` and `MB/s` from each of the 10 runs, then compute mean/median
per mode and the relative difference.

## Notes from previous attempt
- This machine hit a sandbox `PermissionError` reaching `64.247.196.155:8002`, so
  no runs completed here. On the other machine, expect no such restriction —
  just run the commands directly.
- Default payload is 10 GB (`--size-gb 10`); reduce with `--size-gb` if needed.
- The script sleeps 2s after each run to let the final ACK settle before exiting.
