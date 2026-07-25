# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Two-machine cellnet and raw TCP streaming benchmark.

Streams a large payload from a sender to a receiver and prints the time it took
on both sides. Cellnet runs default to 10 GiB. Raw TCP runs default to a 1 GiB
untimed warm-up followed by a 100 GiB measurement, long enough for a stable
25 Gbit/s ceiling measurement.

Cellnet receiver (start this first):

    python dev_tools/f3/cellnet_bench.py recv --url tcp://0.0.0.0:8002

Cellnet sender:

    python dev_tools/f3/cellnet_bench.py send --url tcp://<receiver-host>:8002 --reliable false
    python dev_tools/f3/cellnet_bench.py send --url tcp://<receiver-host>:8002 --reliable true

Optional F3 tuning (use the same YAML on both endpoints):

    python dev_tools/f3/cellnet_bench.py recv --url tcp://0.0.0.0:8002 \
        --f3-config dev_tools/f3/comm_config.yml
    python dev_tools/f3/cellnet_bench.py send --url tcp://<receiver-host>:8002 \
        --f3-config dev_tools/f3/comm_config.yml

Raw TCP receiver and sender (no F3):

    python dev_tools/f3/cellnet_bench.py recv --transport tcp --url tcp://0.0.0.0:8002
    python dev_tools/f3/cellnet_bench.py send --transport tcp --url tcp://<receiver-host>:8002

The receiver stays up and serves any number of runs (Ctrl-C to stop), so you
can A/B reliable=true vs reliable=false against the same receiver process.
Raw TCP mode reuses a large preallocated application buffer and uses recv_into()
to avoid per-chunk allocation. It reports the raw Python/TCP throughput ceiling
for the tested hosts, network, and buffer configuration.
"""

import argparse
import copy
import logging
import os
import re
import socket
import struct
import threading
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import yaml

from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.comm_config import CommConfigurator
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.stream_cell import StreamCell
from nvflare.fuel.f3.streaming.stream_types import Stream, StreamFuture
from nvflare.fuel.utils.config_service import ConfigService

CHANNEL = "bench"
TOPIC = "stream"
RX_FQCN = "server"  # the passive (listening) cell's FQCN must be "server"
TX_FQCN = "sender"

MB = 1024 * 1024
GB = 1024 * MB
BLOCK_SIZE = MB

TCP_MAGIC = b"NVFTCP2\0"
TCP_HEADER = struct.Struct("!8sQQ")
TCP_ACK = struct.Struct("!QQ")
TCP_READY = b"\x01"
TCP_MEASURE = b"\x02"
DEFAULT_TCP_BUFFER_SIZE = 16 * MB
DEFAULT_CELLNET_SIZE = 10 * GB
DEFAULT_TCP_SIZE = 100 * GB
DEFAULT_TCP_WARMUP_SIZE = GB
DEFAULT_TARGET_GBPS = 25.0
DEFAULT_F3_CHUNK_SIZE = MB
DEFAULT_F3_WINDOW_SIZE = 64 * MB

BYTE_SIZE_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMGT]?)(?:I?B)?\s*$", re.IGNORECASE)
BYTE_SIZE_MULTIPLIERS = {
    "": 1,
    "K": 1024,
    "M": 1024**2,
    "G": 1024**3,
    "T": 1024**4,
}
F3_BYTE_SIZE_KEYS = {
    "max_message_size",
    "streaming_chunk_size",
    "streaming_max_blob_size",
    "streaming_window_size",
    "streaming_ack_interval",
    "streaming_retry_max_pending_bytes",
}
GRPC_BYTE_SIZE_OPTIONS = {
    "grpc.max_send_message_length",
    "grpc.max_receive_message_length",
}


class GeneratedStream(Stream):
    """Synthetic stream serving `size` bytes without holding them all in memory."""

    def __init__(self, size: int, block_size: int = BLOCK_SIZE):
        super().__init__(size=size, headers=None)
        self.remaining = size
        self.block_size = block_size
        # One immutable random block reused for every chunk: safe to share, no per-read copy
        self.block = os.urandom(block_size)

    def read(self, size: int) -> bytes:
        if self.remaining <= 0:
            return b""
        n = min(size, self.block_size, self.remaining)
        self.remaining -= n
        self.pos += n
        return self.block if n == self.block_size else self.block[:n]


def rate(num_bytes: int, seconds: float) -> str:
    if seconds <= 0:
        return "n/a"
    return f"{num_bytes / MB / seconds:,.1f} MB/s"


def tcp_rate(num_bytes: int, seconds: float) -> str:
    if seconds <= 0:
        return "n/a"
    return f"{num_bytes / MB / seconds:,.1f} MiB/s, {num_bytes * 8 / 1_000_000_000 / seconds:,.3f} Gbit/s"


def target_utilization(num_bytes: int, seconds: float, target_gbps: float) -> float:
    if seconds <= 0 or target_gbps <= 0:
        return 0.0
    actual_gbps = num_bytes * 8 / 1_000_000_000 / seconds
    return actual_gbps / target_gbps * 100


def parse_byte_size(value, name: str = "size") -> int:
    """Parse a byte count using binary K/M/G/T suffixes."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a byte size, got {value!r}")
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an integer or byte-size string, got {value!r}")

    match = BYTE_SIZE_PATTERN.fullmatch(value)
    if not match:
        raise ValueError(
            f"{name} has invalid byte size {value!r}; use bytes or a binary suffix such as 128K, 16M, or 2G"
        )

    try:
        number = Decimal(match.group(1))
    except InvalidOperation as ex:
        raise ValueError(f"{name} has invalid byte size {value!r}") from ex
    multiplier = BYTE_SIZE_MULTIPLIERS[match.group(2).upper()]
    num_bytes = number * multiplier
    if num_bytes != num_bytes.to_integral_value():
        raise ValueError(f"{name} byte size {value!r} does not resolve to a whole number of bytes")
    return int(num_bytes)


def normalize_f3_config(config: dict) -> dict:
    """Convert unit-suffixed F3 byte-count settings to integer bytes."""
    normalized = copy.deepcopy(config)
    for name in F3_BYTE_SIZE_KEYS:
        if name in normalized:
            normalized[name] = parse_byte_size(normalized[name], name)

    grpc = normalized.get("grpc")
    if isinstance(grpc, dict):
        options = grpc.get("options")
        if isinstance(options, list):
            for option in options:
                if isinstance(option, list) and len(option) == 2 and option[0] in GRPC_BYTE_SIZE_OPTIONS:
                    option[1] = parse_byte_size(option[1], option[0])
    return normalized


def _positive_int(config: dict, name: str, default: int) -> int:
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def validate_f3_config(config: dict):
    if not isinstance(config, dict):
        raise ValueError(f"F3 configuration must be a YAML mapping, got {type(config).__name__}")

    chunk_size = _positive_int(config, "streaming_chunk_size", DEFAULT_F3_CHUNK_SIZE)
    window_size = _positive_int(config, "streaming_window_size", DEFAULT_F3_WINDOW_SIZE)
    if window_size < chunk_size:
        raise ValueError(
            f"streaming_window_size ({window_size:,}) must be at least streaming_chunk_size ({chunk_size:,})"
        )

    if "streaming_ack_interval" in config:
        ack_interval = _positive_int(config, "streaming_ack_interval", 1)
        if ack_interval > window_size:
            raise ValueError(
                f"streaming_ack_interval ({ack_interval:,}) must not exceed streaming_window_size ({window_size:,})"
            )


def configure_f3(config_file: str) -> tuple[Path, dict]:
    """Load a native F3 comm_config YAML before any cells are created."""
    path = Path(config_file).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"F3 configuration file does not exist: {path}")
    if path.suffix.lower() not in (".yml", ".yaml"):
        raise ValueError(f"F3 configuration must be a .yml or .yaml file: {path}")
    if path.stem != "comm_config":
        raise ValueError(f"F3 configuration must be named comm_config.yml or comm_config.yaml, got {path.name!r}")

    with path.open(encoding="utf-8") as config_stream:
        requested_config = yaml.safe_load(config_stream)
    if requested_config is None:
        requested_config = {}
    if not isinstance(requested_config, dict):
        raise ValueError(f"F3 configuration must be a YAML mapping, got {type(requested_config).__name__}")
    normalized_config = normalize_f3_config(requested_config)
    validate_f3_config(normalized_config)

    # CommConfigurator discovers the native comm_config by basename. Initializing
    # ConfigService with this directory preserves every flat and nested F3 option.
    ConfigService.reset()
    ConfigService.initialize(section_files={}, config_path=[str(path.parent)])
    CommConfigurator.reset()
    loaded_config = CommConfigurator().get_config()
    if loaded_config is None:
        raise ValueError(
            f"F3 could not load {path}; install NVFlare's CONFIG extra to enable YAML configuration support"
        )
    if loaded_config != requested_config:
        raise ValueError(f"F3 loaded a different comm_config from {path.parent}; remove conflicting config files")
    # F3 exposes its loaded native configuration as a mutable dict. Replace the
    # unit-bearing strings with normalized byte counts before any F3 component
    # reads flat or nested settings.
    loaded_config.clear()
    loaded_config.update(normalized_config)
    return path, loaded_config


def f3_config_summary() -> str:
    config = CommConfigurator()
    chunk_size = config.get_streaming_chunk_size(DEFAULT_F3_CHUNK_SIZE)
    window_size = config.get_streaming_window_size(DEFAULT_F3_WINDOW_SIZE)
    return (
        f"streaming_chunk_size={chunk_size:,} ({chunk_size / MB:,.1f} MiB), "
        f"streaming_window_size={window_size:,} ({window_size / MB:,.1f} MiB)"
    )


def parse_tcp_url(url: str) -> tuple[str, int]:
    parsed = urlparse(url)
    if parsed.scheme != "tcp":
        raise ValueError(f"raw TCP mode requires a tcp:// URL, got {url!r}")
    if parsed.username or parsed.password or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ValueError(f"raw TCP URL must contain only a host and port, got {url!r}")
    if not parsed.hostname:
        raise ValueError(f"raw TCP URL is missing a host: {url!r}")
    try:
        port = parsed.port
    except ValueError as ex:
        raise ValueError(f"invalid port in raw TCP URL {url!r}") from ex
    if port is None:
        raise ValueError(f"raw TCP URL is missing a port: {url!r}")
    return parsed.hostname, port


def _configure_tcp_socket(sock: socket.socket, socket_buffer_size: int):
    """Optionally request kernel buffers; zero preserves the OS TCP autotuner."""
    if socket_buffer_size > 0:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, socket_buffer_size)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, socket_buffer_size)


def _socket_buffer_summary(sock: socket.socket, socket_buffer_size: int) -> str:
    send_buffer = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
    recv_buffer = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    mode = "OS autotuning" if socket_buffer_size == 0 else f"requested={socket_buffer_size / MB:,.1f} MiB"
    return f"kernel buffers: {mode}, reported send={send_buffer / MB:,.1f} MiB " f"receive={recv_buffer / MB:,.1f} MiB"


def _open_tcp_client(host: str, port: int, socket_buffer_size: int = 0, timeout: float = 30.0) -> socket.socket:
    last_error = None
    for family, sock_type, protocol, _, address in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM):
        sock = socket.socket(family, sock_type, protocol)
        try:
            _configure_tcp_socket(sock, socket_buffer_size)
            sock.settimeout(timeout)
            sock.connect(address)
            sock.settimeout(None)
            return sock
        except OSError as ex:
            last_error = ex
            sock.close()
    raise ConnectionError(f"could not connect to {host}:{port}") from last_error


def _open_tcp_listener(host: str, port: int, socket_buffer_size: int = 0) -> socket.socket:
    last_error = None
    flags = socket.AI_PASSIVE if not host else 0
    for family, sock_type, protocol, _, address in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM, flags=flags):
        sock = socket.socket(family, sock_type, protocol)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            _configure_tcp_socket(sock, socket_buffer_size)
            sock.bind(address)
            sock.listen(16)
            return sock
        except OSError as ex:
            last_error = ex
            sock.close()
    raise OSError(f"could not listen on {host}:{port}") from last_error


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    buf = bytearray(size)
    view = memoryview(buf)
    received = 0
    while received < size:
        n = sock.recv_into(view[received:])
        if n == 0:
            raise ConnectionError(f"connection closed after {received} of {size} protocol bytes")
        received += n
    return bytes(buf)


def _allocate_tcp_buffer(size: int) -> bytearray:
    """Allocate and touch every memory page so page faults are outside the timed transfer."""
    buf = bytearray(size)
    page_size = os.sysconf("SC_PAGESIZE")
    for offset in range(0, size, page_size):
        buf[offset] = 1
    return buf


def _send_buffer(sock: socket.socket, view: memoryview, size: int, buffer_size: int):
    full_blocks, tail = divmod(size, buffer_size)
    for _ in range(full_blocks):
        sock.sendall(view)
    if tail:
        sock.sendall(view[:tail])


def _drain_buffer(sock: socket.socket, view: memoryview, size: int, buffer_size: int) -> int:
    received = 0
    while received < size:
        n = sock.recv_into(view, min(buffer_size, size - received))
        if n == 0:
            raise ConnectionError(f"connection closed after {received:,} of {size:,} payload bytes")
        received += n
    return received


def _send_tcp_payload(
    sock: socket.socket, size: int, buffer_size: int, warmup_size: int = 0
) -> tuple[float, int, float]:
    """Send one raw TCP run and return sender time, receiver byte count, and receiver time."""
    # Allocate and fault in the reusable block before either side starts its timer.
    block = _allocate_tcp_buffer(buffer_size)
    view = memoryview(block)
    sock.sendall(TCP_HEADER.pack(TCP_MAGIC, warmup_size, size))
    if _recv_exact(sock, len(TCP_READY)) != TCP_READY:
        raise RuntimeError("receiver did not acknowledge the raw TCP benchmark header")

    _send_buffer(sock, view, warmup_size, buffer_size)
    if _recv_exact(sock, len(TCP_MEASURE)) != TCP_MEASURE:
        raise RuntimeError("receiver did not acknowledge the raw TCP warm-up")

    start = time.perf_counter()
    _send_buffer(sock, view, size, buffer_size)

    ack = _recv_exact(sock, TCP_ACK.size)
    elapsed = time.perf_counter() - start
    received, receiver_elapsed_ns = TCP_ACK.unpack(ack)
    return elapsed, received, receiver_elapsed_ns / 1_000_000_000


def _receive_tcp_payload(sock: socket.socket, buffer_size: int) -> tuple[int, float, int]:
    """Receive one raw TCP run into a reusable buffer and acknowledge after draining it."""
    magic, warmup_size, expected = TCP_HEADER.unpack(_recv_exact(sock, TCP_HEADER.size))
    if magic != TCP_MAGIC:
        raise ValueError("connection did not send a cellnet_bench raw TCP header")

    # recv_into() avoids allocating a bytes object for every chunk. Allocate before
    # READY so setup time is excluded from both sender and receiver measurements.
    block = _allocate_tcp_buffer(buffer_size)
    view = memoryview(block)
    sock.sendall(TCP_READY)

    _drain_buffer(sock, view, warmup_size, buffer_size)
    sock.sendall(TCP_MEASURE)

    start = time.perf_counter()
    received = _drain_buffer(sock, view, expected, buffer_size)
    elapsed = time.perf_counter() - start
    sock.sendall(TCP_ACK.pack(received, int(elapsed * 1_000_000_000)))
    return received, elapsed, warmup_size


def run_tcp_receiver(url: str, buffer_size: int, socket_buffer_size: int = 0, target_gbps: float = DEFAULT_TARGET_GBPS):
    host, port = parse_tcp_url(url)
    listener = _open_tcp_listener(host, port, socket_buffer_size)
    print(
        f"[tcp-recv] listening on {url} with no F3; application buffer={buffer_size / MB:,.1f} MiB, "
        f"{_socket_buffer_summary(listener, socket_buffer_size)}"
    )
    print("[tcp-recv] waiting for raw TCP runs (Ctrl-C to stop)")
    try:
        while True:
            conn, address = listener.accept()
            with conn:
                print(f"[tcp-recv] accepted {address}; {_socket_buffer_summary(conn, socket_buffer_size)}")
                try:
                    received, elapsed, warmup_size = _receive_tcp_payload(conn, buffer_size)
                    utilization = target_utilization(received, elapsed, target_gbps)
                    print(
                        f"[tcp-recv] TCP_BASELINE_NO_F3 warmup_bytes={warmup_size} bytes={received} "
                        f"seconds={elapsed:.6f} "
                        f"mib_per_sec={received / MB / elapsed:.1f} "
                        f"gbit_per_sec={received * 8 / 1_000_000_000 / elapsed:.3f} "
                        f"target_gbit_per_sec={target_gbps:.3f} target_utilization_pct={utilization:.1f}"
                    )
                    print(f"[tcp-recv] raw Python/TCP ceiling for this run: {tcp_rate(received, elapsed)}")
                    print(f"[tcp-recv] utilization of nominal {target_gbps:.1f} Gbit/s: {utilization:.1f}%")
                except (ConnectionError, OSError, ValueError) as ex:
                    print(f"[tcp-recv] ERROR from {address}: {ex}")
    except KeyboardInterrupt:
        print("[tcp-recv] stopping")
    finally:
        listener.close()


def run_tcp_sender(
    url: str,
    size: int,
    buffer_size: int,
    warmup_size: int = 0,
    socket_buffer_size: int = 0,
    target_gbps: float = DEFAULT_TARGET_GBPS,
):
    host, port = parse_tcp_url(url)
    print(f"[tcp-send] connecting directly to {host}:{port} (no F3) ...")
    with _open_tcp_client(host, port, socket_buffer_size) as sock:
        print(
            f"[tcp-send] connected; warm-up={warmup_size / GB:.2f} GiB, measured payload={size / GB:.2f} GiB, "
            f"application buffer={buffer_size / MB:,.1f} MiB, {_socket_buffer_summary(sock, socket_buffer_size)}"
        )
        elapsed, received, receiver_elapsed = _send_tcp_payload(sock, size, buffer_size, warmup_size)

    if received != size:
        raise RuntimeError(f"receiver reported {received:,} bytes, expected {size:,}")
    utilization = target_utilization(size, elapsed, target_gbps)
    print(
        f"[tcp-send] TCP_BASELINE_NO_F3 warmup_bytes={warmup_size} bytes={size} seconds={elapsed:.6f} "
        f"mib_per_sec={size / MB / elapsed:.1f} gbit_per_sec={size * 8 / 1_000_000_000 / elapsed:.3f} "
        f"receiver_seconds={receiver_elapsed:.6f} receiver_mib_per_sec={size / MB / receiver_elapsed:.1f} "
        f"target_gbit_per_sec={target_gbps:.3f} target_utilization_pct={utilization:.1f}"
    )
    print(
        f"[tcp-send] raw Python/TCP ceiling for this run: {tcp_rate(size, elapsed)} "
        f"(receiver drain: {tcp_rate(size, receiver_elapsed)})"
    )
    print(f"[tcp-send] utilization of nominal {target_gbps:.1f} Gbit/s: {utilization:.1f}%")


def rss_bytes() -> int:
    """Current RSS of this process, in bytes (Linux)."""
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGESIZE")


class MemSampler:
    """Samples process RSS in a background thread; reports baseline/peak/mean."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.baseline = rss_bytes()
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.interval):
            self.samples.append(rss_bytes())

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)
        self.samples.append(rss_bytes())

    @property
    def peak(self) -> int:
        return max(self.samples, default=self.baseline)

    @property
    def mean(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else float(self.baseline)

    def summary(self) -> str:
        return (
            f"baseline={self.baseline / MB:,.1f} MB peak={self.peak / MB:,.1f} MB "
            f"mean={self.mean / MB:,.1f} MB delta_peak={(self.peak - self.baseline) / MB:,.1f} MB"
        )


def stream_cb(future: StreamFuture, stream: Stream, resume: bool, **kwargs):
    """Receiver-side callback: drain the stream and report throughput."""
    sid = future.get_stream_id()
    expected = stream.get_size()
    read_size = CommConfigurator().get_streaming_chunk_size(DEFAULT_F3_CHUNK_SIZE)
    print(f"[recv] stream {sid} started, expected size: {expected / GB:.2f} GB")

    received = 0
    next_report = GB
    start = time.perf_counter()
    while True:
        buf = stream.read(read_size)
        if not buf:
            break
        received += len(buf)
        if received >= next_report:
            elapsed = time.perf_counter() - start
            print(f"[recv] stream {sid}: {received / GB:.1f} GB in {elapsed:,.1f}s ({rate(received, elapsed)})")
            next_report += GB

    elapsed = time.perf_counter() - start
    status = "OK" if received == expected else f"SIZE MISMATCH (expected {expected:,})"
    print(
        f"[recv] stream {sid} DONE: {received:,} bytes in {elapsed:,.2f} seconds "
        f"({rate(received, elapsed)}) {status}"
    )


def run_receiver(url: str):
    cell = CoreCell(RX_FQCN, url, secure=False, credentials={})
    stream_cell = StreamCell(cell)
    stream_cell.register_stream_cb(CHANNEL, TOPIC, stream_cb)
    cell.start()
    print(f"[recv] listening on {url}, waiting for streams (Ctrl-C to stop)")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("[recv] stopping")
    finally:
        cell.stop()


def run_sender(url: str, size: int, reliable: Optional[bool]):
    connected = threading.Event()
    cell = CoreCell(TX_FQCN, url, secure=False, credentials={})
    cell.set_cell_connected_cb(lambda agent: connected.set())
    stream_cell = StreamCell(cell)
    cell.start()

    print(f"[send] connecting to {url} ...")
    if not connected.wait(timeout=30):
        cell.stop()
        raise SystemExit(f"[send] ERROR: could not connect to receiver at {url} within 30 seconds")

    effective_reliable = CommConfigurator().get_streaming_reliable(False) if reliable is None else reliable
    chunk_size = stream_cell.get_chunk_size()
    print(
        f"[send] connected, sending {size / GB:.2f} GB with reliable={effective_reliable}, "
        f"chunk_size={chunk_size / MB:,.1f} MiB"
    )
    stream = GeneratedStream(size, block_size=chunk_size)
    mem = MemSampler()
    mem.start()
    start = time.perf_counter()
    future = stream_cell.send_stream(CHANNEL, TOPIC, RX_FQCN, Message(None, stream), reliable=reliable)

    stop_progress = threading.Event()

    def report_progress():
        while not stop_progress.wait(5.0):
            sent = future.get_progress()
            elapsed = time.perf_counter() - start
            print(
                f"[send] progress: {sent / GB:.2f} GB in {elapsed:,.1f}s "
                f"({rate(sent, elapsed)}) rss={rss_bytes() / MB:,.1f} MB"
            )

    threading.Thread(target=report_progress, daemon=True).start()

    try:
        bytes_sent = future.result()
        elapsed = time.perf_counter() - start
        mem.stop()
        print(
            f"[send] RESULT reliable={effective_reliable}: sent {bytes_sent:,} bytes "
            f"in {elapsed:,.2f} seconds ({rate(bytes_sent, elapsed)})"
        )
        print(f"[send] MEMORY reliable={effective_reliable}: {mem.summary()}")
    finally:
        stop_progress.set()

    # Give the last ACK exchange a moment to settle before tearing the cell down
    time.sleep(2)
    cell.stop()


def main():
    parser = argparse.ArgumentParser(description="cellnet or raw TCP streaming benchmark (2 machines)")
    parser.add_argument("--log-level", default="WARNING", help="python log level for cellnet internals")
    sub = parser.add_subparsers(dest="role", required=True)

    p_recv = sub.add_parser("recv", help="run the receiver (start this first)")
    p_recv.add_argument("--url", default="tcp://0.0.0.0:8002", help="listening URL (default tcp://0.0.0.0:8002)")
    p_recv.add_argument(
        "--transport",
        choices=["cellnet", "tcp"],
        default="cellnet",
        help="cellnet (F3) or optimized raw TCP with no F3 (default cellnet)",
    )
    p_recv.add_argument(
        "--buffer-mb",
        type=float,
        default=DEFAULT_TCP_BUFFER_SIZE / MB,
        help="raw TCP application transfer buffer in MiB (default 16)",
    )
    p_recv.add_argument(
        "--socket-buffer-mb",
        type=float,
        default=0,
        help="raw TCP kernel send/receive buffer in MiB; 0 preserves OS autotuning (default 0)",
    )
    p_recv.add_argument(
        "--target-gbps",
        type=float,
        default=DEFAULT_TARGET_GBPS,
        help="nominal network rate used to report utilization (default 25)",
    )
    p_recv.add_argument(
        "--f3-config",
        "--config",
        dest="f3_config",
        help="optional native F3 comm_config.yml/comm_config.yaml file (cellnet transport only)",
    )

    p_send = sub.add_parser("send", help="run the sender")
    p_send.add_argument("--url", required=True, help="receiver URL, e.g. tcp://10.1.2.3:8002")
    p_send.add_argument(
        "--size-gb",
        type=float,
        default=None,
        help="measured payload in GiB (default 10 for cellnet, 100 for raw TCP)",
    )
    p_send.add_argument(
        "--reliable",
        choices=["true", "false"],
        default=None,
        help="override streaming_reliable; otherwise use the F3 config or false",
    )
    p_send.add_argument(
        "--transport",
        choices=["cellnet", "tcp"],
        default="cellnet",
        help="cellnet (F3) or optimized raw TCP with no F3 (default cellnet)",
    )
    p_send.add_argument(
        "--buffer-mb",
        type=float,
        default=DEFAULT_TCP_BUFFER_SIZE / MB,
        help="raw TCP application transfer buffer in MiB (default 16)",
    )
    p_send.add_argument(
        "--warmup-gb",
        type=float,
        default=DEFAULT_TCP_WARMUP_SIZE / GB,
        help="untimed raw TCP warm-up in GiB (default 1)",
    )
    p_send.add_argument(
        "--socket-buffer-mb",
        type=float,
        default=0,
        help="raw TCP kernel send/receive buffer in MiB; 0 preserves OS autotuning (default 0)",
    )
    p_send.add_argument(
        "--target-gbps",
        type=float,
        default=DEFAULT_TARGET_GBPS,
        help="nominal network rate used to report utilization (default 25)",
    )
    p_send.add_argument(
        "--f3-config",
        "--config",
        dest="f3_config",
        help="optional native F3 comm_config.yml/comm_config.yaml file (cellnet transport only)",
    )

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.WARNING))
    if args.f3_config:
        if args.transport != "cellnet":
            parser.error("--f3-config applies only to --transport cellnet")
        try:
            config_path, _ = configure_f3(args.f3_config)
        except (OSError, ValueError, yaml.YAMLError) as ex:
            parser.error(str(ex))
        print(f"[f3-config] loaded {config_path}: {f3_config_summary()}")

    buffer_size = int(args.buffer_mb * MB)
    if buffer_size <= 0:
        parser.error("--buffer-mb must be greater than zero")
    socket_buffer_size = int(args.socket_buffer_mb * MB)
    if socket_buffer_size < 0:
        parser.error("--socket-buffer-mb must be zero or greater")
    if args.target_gbps <= 0:
        parser.error("--target-gbps must be greater than zero")

    if args.role == "recv":
        if args.transport == "tcp":
            run_tcp_receiver(args.url, buffer_size, socket_buffer_size, args.target_gbps)
        else:
            run_receiver(args.url)
    else:
        default_size = DEFAULT_TCP_SIZE if args.transport == "tcp" else DEFAULT_CELLNET_SIZE
        size = int(args.size_gb * GB) if args.size_gb is not None else default_size
        if size <= 0:
            parser.error("--size-gb must be greater than zero")
        if args.transport == "tcp":
            warmup_size = int(args.warmup_gb * GB)
            if warmup_size < 0:
                parser.error("--warmup-gb must be zero or greater")
            run_tcp_sender(
                args.url,
                size,
                buffer_size,
                warmup_size,
                socket_buffer_size,
                args.target_gbps,
            )
        else:
            reliable = None if args.reliable is None else args.reliable == "true"
            run_sender(args.url, size, reliable)


if __name__ == "__main__":
    main()
