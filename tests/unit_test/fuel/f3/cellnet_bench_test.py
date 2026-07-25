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

import threading

import pytest

from dev_tools.f3 import cellnet_bench
from nvflare.fuel.f3.streaming.byte_receiver import ACK_INTERVAL
from nvflare.fuel.f3.streaming.byte_streamer import STREAM_CHUNK_SIZE, STREAM_WINDOW_SIZE


def test_benchmark_defaults_match_f3_streaming_defaults():
    assert cellnet_bench.DEFAULT_F3_CHUNK_SIZE == STREAM_CHUNK_SIZE == 1024**2
    assert cellnet_bench.DEFAULT_F3_WINDOW_SIZE == STREAM_WINDOW_SIZE == 64 * 1024**2
    assert ACK_INTERVAL == 16 * 1024**2


@pytest.mark.parametrize(
    "url, expected",
    [
        ("tcp://127.0.0.1:8002", ("127.0.0.1", 8002)),
        ("tcp://example.test:1234/", ("example.test", 1234)),
        ("tcp://[::1]:9000", ("::1", 9000)),
    ],
)
def test_parse_tcp_url(url, expected):
    assert cellnet_bench.parse_tcp_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8002",
        "tcp://127.0.0.1",
        "tcp://127.0.0.1:invalid",
        "tcp://127.0.0.1:8002/path",
    ],
)
def test_parse_tcp_url_rejects_invalid_url(url):
    with pytest.raises(ValueError):
        cellnet_bench.parse_tcp_url(url)


def test_raw_tcp_sender_reports_end_to_end_baseline(capsys):
    buffer_size = 256 * 1024
    warmup_size = buffer_size * 2 + 17
    payload_size = buffer_size * 8 + 123
    listener = cellnet_bench._open_tcp_listener("127.0.0.1", 0)
    port = listener.getsockname()[1]
    receiver_result = {}
    receiver_error = []

    def receive_once():
        try:
            with listener:
                conn, _ = listener.accept()
                with conn:
                    receiver_result["value"] = cellnet_bench._receive_tcp_payload(conn, buffer_size)
        except Exception as ex:
            receiver_error.append(ex)

    receiver = threading.Thread(target=receive_once)
    receiver.start()
    cellnet_bench.run_tcp_sender(
        f"tcp://127.0.0.1:{port}",
        payload_size,
        buffer_size,
        warmup_size=warmup_size,
    )
    receiver.join(timeout=10)

    assert not receiver.is_alive()
    assert not receiver_error
    assert receiver_result["value"][0] == payload_size
    output = capsys.readouterr().out
    assert "[tcp-send] TCP_BASELINE_NO_F3" in output
    assert f"warmup_bytes={warmup_size}" in output
    assert f"bytes={payload_size}" in output
    assert "mib_per_sec=" in output
    assert "gbit_per_sec=" in output
    assert "receiver_mib_per_sec=" in output
    assert "target_gbit_per_sec=25.000" in output
    assert "target_utilization_pct=" in output


@pytest.mark.parametrize(
    "value, expected",
    [
        (1024, 1024),
        ("1024", 1024),
        ("128K", 128 * 1024),
        ("16M", 16 * 1024**2),
        ("2G", 2 * 1024**3),
        ("1.5M", 1_572_864),
        ("2GiB", 2 * 1024**3),
        ("16mb", 16 * 1024**2),
    ],
)
def test_parse_byte_size_uses_binary_units(value, expected):
    assert cellnet_bench.parse_byte_size(value) == expected


def test_configure_f3_loads_chunk_and_window_sizes(tmp_path):
    config_file = tmp_path / "comm_config.yml"
    config_file.write_text(
        "\n".join(
            [
                "streaming_chunk_size: 4M",
                "streaming_window_size: 256M",
                "streaming_ack_interval: 64M",
                "grpc:",
                "  max_workers: 32",
                "  options:",
                "    - [grpc.max_send_message_length, 2G]",
            ]
        ),
        encoding="utf-8",
    )

    try:
        loaded_path, config = cellnet_bench.configure_f3(str(config_file))
        configurator = cellnet_bench.CommConfigurator()

        assert loaded_path == config_file
        assert config["grpc"]["max_workers"] == 32
        assert config["grpc"]["options"][0][1] == 2 * 1024**3
        assert configurator.get_streaming_chunk_size(1) == 4194304
        assert configurator.get_streaming_window_size(1) == 268435456
        assert "streaming_chunk_size=4,194,304" in cellnet_bench.f3_config_summary()
    finally:
        cellnet_bench.ConfigService.reset()
        cellnet_bench.CommConfigurator.reset()


@pytest.mark.parametrize(
    "config_text, error",
    [
        ("- not-a-mapping\n", "must be a YAML mapping"),
        ("streaming_chunk_size: 0\n", "streaming_chunk_size must be a positive integer"),
        ("streaming_chunk_size: 16Mbps\n", "has invalid byte size"),
        ("streaming_chunk_size: 0.1K\n", "does not resolve to a whole number of bytes"),
        (
            "streaming_chunk_size: 4194304\nstreaming_window_size: 1048576\n",
            "streaming_window_size .* must be at least streaming_chunk_size",
        ),
        (
            "streaming_window_size: 16777216\nstreaming_ack_interval: 33554432\n",
            "streaming_ack_interval .* must not exceed streaming_window_size",
        ),
    ],
)
def test_configure_f3_rejects_invalid_streaming_settings(tmp_path, config_text, error):
    config_file = tmp_path / "comm_config.yml"
    config_file.write_text(config_text, encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        cellnet_bench.configure_f3(str(config_file))


def test_generated_stream_uses_configured_block_size():
    block_size = 2 * cellnet_bench.MB
    stream = cellnet_bench.GeneratedStream(block_size + 17, block_size=block_size)

    assert len(stream.read(block_size)) == block_size
    assert len(stream.read(block_size)) == 17
