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

"""Unit tests for TensorDownloadable.

Note: Deep copy protection is now handled at broadcast level in WFCommServer,
not in TensorDownloadable itself. These tests verify the Downloadable's basic behavior.
"""

import gc
import json
import time

import pytest
import torch
from safetensors.torch import load as load_tensors

import nvflare.app_opt.pt.tensor_downloader as tensor_downloader
from nvflare.app_opt.pt.tensor_downloader import (
    DiskTensorConsumer,
    TensorConsumer,
    TensorDownloadable,
    _serialize_tensor_item,
    _StreamedTensorItem,
    add_tensors,
    download_tensors,
)
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.streaming.download_service import DirectDownloadChunk, ProduceRC
from nvflare.fuel.f3.streaming.obj_downloader import ObjectDownloader
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.network_utils import get_open_ports


def _materialize_item(item) -> bytes:
    """Apply the same FOBS externalize/internalize step used by Cell messages."""
    return fobs.loads(fobs.dumps(item, buffer_list=True))


class TestTensorDownloadableBasic:
    """Test basic TensorDownloadable functionality."""

    def test_basic_functionality(self):
        """Verify basic Downloadable creation and data access."""
        # Create tensors
        tensors = {
            "weights": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "bias": torch.tensor([0.5, 1.5]),
        }

        # Create downloadable
        downloadable = TensorDownloadable(tensors=tensors, max_chunk_size=1024)

        # Verify basic properties
        assert downloadable.size == 2
        assert set(downloadable.keys) == {"weights", "bias"}
        assert torch.allclose(downloadable.base_obj["weights"], tensors["weights"])
        assert torch.allclose(downloadable.base_obj["bias"], tensors["bias"])

    def test_shares_memory_with_original(self):
        """Verify that Downloadable references original tensors (no copy at this level)."""
        original = {"layer": torch.tensor([1.0, 2.0, 3.0])}

        downloadable = TensorDownloadable(tensors=original, max_chunk_size=1024)

        # Should share memory (snapshot is done at broadcast level, not here)
        assert (
            downloadable.base_obj["layer"].data_ptr() == original["layer"].data_ptr()
        ), "Downloadable should share memory with original (copy is done at broadcast level)"

    def test_modification_affects_downloadable(self):
        """Verify that modifications to original DO affect Downloadable (by design).

        Note: Protection against this is now handled at broadcast level in WFCommServer.
        """
        tensors = {"model": torch.tensor([1.0, 2.0])}

        downloadable = TensorDownloadable(tensors=tensors, max_chunk_size=1024)

        # Modify original
        tensors["model"][0] = 999.0

        # Downloadable IS affected (this is expected - protection is at broadcast level)
        assert downloadable.base_obj["model"][0].item() == 999.0

    def test_prefetches_next_tensor(self):
        tensors = {
            "first": torch.tensor([1.0]),
            "second": torch.tensor([2.0]),
        }
        downloadable = TensorDownloadable(tensors=tensors, max_chunk_size=1)

        rc, first_items, state = downloadable.produce({}, "receiver")

        assert rc == ProduceRC.OK
        assert load_tensors(_materialize_item(first_items[0]))["first"].item() == 1.0
        assert 1 in downloadable._prefetch_futures

        rc, second_items, state = downloadable.produce(state, "receiver")

        assert rc == ProduceRC.OK
        assert load_tensors(_materialize_item(second_items[0]))["second"].item() == 2.0
        assert not downloadable._prefetch_futures

    def test_prefetch_does_not_queue_two_oversized_tensors(self):
        tensors = {
            "first": torch.tensor([1.0]),
            "second": torch.zeros(1024),
            "third": torch.zeros(1024),
        }
        downloadable = TensorDownloadable(tensors=tensors, max_chunk_size=1)

        downloadable.produce({}, "receiver")

        assert 1 in downloadable._prefetch_futures
        assert 2 not in downloadable._prefetch_futures
        downloadable.release()

    def test_release_disables_prefetch_and_produce(self):
        tensors = {
            "first": torch.tensor([1.0]),
            "second": torch.tensor([2.0]),
        }
        downloadable = TensorDownloadable(tensors=tensors, max_chunk_size=1)

        downloadable.release()

        assert downloadable.base_obj is None
        downloadable.prefetch_item(1)
        assert not downloadable._prefetch_futures
        assert downloadable.get_item_size(0) is None
        with pytest.raises(RuntimeError, match="released"):
            downloadable.produce_item(0)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float64,
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int64,
        torch.int32,
        torch.int16,
        torch.int8,
        torch.uint8,
        torch.bool,
    ],
)
def test_direct_tensor_round_trip(dtype, monkeypatch):
    tensor = torch.arange(12, dtype=torch.int64).to(dtype).reshape(3, 4)
    monkeypatch.setattr(tensor_downloader, "TEN_MEGA", 0)
    consumer = TensorConsumer(None, {})

    item = _serialize_tensor_item('weight "\N{SNOWMAN}"', tensor, stream_tensor=True)
    payload = bytearray(b"".join(item.data))
    received = consumer.consume_direct_chunk(payload)[0]

    assert isinstance(item, DirectDownloadChunk)
    assert isinstance(received, _StreamedTensorItem)
    assert received.key == 'weight "\N{SNOWMAN}"'
    assert torch.equal(received.tensor, tensor)
    assert len(received) == len(payload)


def test_direct_tensor_handles_scalar(monkeypatch):
    monkeypatch.setattr(tensor_downloader, "TEN_MEGA", 0)
    tensor = torch.tensor(3.5)
    item = _serialize_tensor_item("weight", tensor, stream_tensor=True)
    payload = bytearray(b"".join(item.data))
    restored = TensorConsumer(None, {}).consume_direct_chunk(payload)[0]

    assert isinstance(restored, _StreamedTensorItem)
    assert torch.equal(restored.tensor, tensor)


def test_direct_tensor_handles_empty_tensor(monkeypatch):
    monkeypatch.setattr(tensor_downloader, "TEN_MEGA", 0)
    tensor = torch.empty(0, dtype=torch.float32)
    item = _serialize_tensor_item("weight", tensor, stream_tensor=True)
    payload = bytearray(b"".join(item.data))
    restored = TensorConsumer(None, {}).consume_direct_chunk(payload)[0]

    assert restored.tensor.shape == (0,)
    assert restored.tensor.dtype == torch.float32


def test_direct_tensor_snapshots_before_streaming(monkeypatch):
    tensor = torch.tensor([1.0, 2.0, 3.0])
    monkeypatch.setattr(tensor_downloader, "TEN_MEGA", 0)

    item = _serialize_tensor_item("weight", tensor, stream_tensor=True)
    tensor.fill_(99.0)

    payload = bytearray(b"".join(item.data))
    restored = TensorConsumer(None, {}).consume_direct_chunk(payload)[0]
    assert torch.equal(restored.tensor, torch.tensor([1.0, 2.0, 3.0]))


def test_direct_tensor_uses_received_writable_buffer_without_copy(monkeypatch):
    monkeypatch.setattr(tensor_downloader, "TEN_MEGA", 0)
    item = _serialize_tensor_item("weight", torch.arange(4, dtype=torch.float32), stream_tensor=True)
    payload = bytearray(b"".join(item.data))
    received = TensorConsumer(None, {}).consume_direct_chunk(payload)[0]

    received.tensor[0] = 99.0
    body_offset = tensor_downloader._DIRECT_TENSOR_HEADER.unpack_from(payload)[2]

    assert torch.frombuffer(payload, dtype=torch.float32, count=1, offset=body_offset).item() == 99.0


def test_direct_tensor_retains_received_buffer_ownership(monkeypatch):
    monkeypatch.setattr(tensor_downloader, "TEN_MEGA", 0)
    item = _serialize_tensor_item("weight", torch.arange(4, dtype=torch.float32), stream_tensor=True)
    payload = bytearray(b"".join(item.data))
    tensor = TensorConsumer(None, {}).consume_direct_chunk(payload)[0].tensor

    del item
    del payload
    gc.collect()

    assert torch.equal(tensor, torch.arange(4, dtype=torch.float32))
    tensor.add_(1)
    assert torch.equal(tensor, torch.arange(1, 5, dtype=torch.float32))


def test_direct_tensor_rejects_readonly_or_malformed_payload(monkeypatch):
    monkeypatch.setattr(tensor_downloader, "TEN_MEGA", 0)
    item = _serialize_tensor_item("weight", torch.arange(4, dtype=torch.float32), stream_tensor=True)
    payload = b"".join(item.data)
    consumer = TensorConsumer(None, {})

    with pytest.raises(ValueError, match="writable"):
        consumer.consume_direct_chunk(payload)

    malformed = bytearray(payload)
    malformed[:8] = b"BADMAGIC"
    with pytest.raises(ValueError, match="magic"):
        consumer.consume_direct_chunk(malformed)

    oversized_header = bytearray(
        b"".join(_serialize_tensor_item("weight", torch.arange(100, dtype=torch.float32), stream_tensor=True).data)
    )
    magic, metadata_size, header_size = tensor_downloader._DIRECT_TENSOR_HEADER.unpack_from(oversized_header)
    tensor_downloader._DIRECT_TENSOR_HEADER.pack_into(
        oversized_header, 0, magic, metadata_size, header_size + tensor_downloader._DIRECT_TENSOR_ALIGNMENT
    )
    with pytest.raises(ValueError, match="header size"):
        consumer.consume_direct_chunk(oversized_header)


def test_direct_tensor_rejects_shape_larger_than_payload():
    metadata = json.dumps(
        {"key": "weight", "dtype": "F32", "shape": [2**1000], "size": 0}, separators=(",", ":")
    ).encode()
    unpadded_size = tensor_downloader._DIRECT_TENSOR_HEADER.size + len(metadata)
    header_size = (
        (unpadded_size + tensor_downloader._DIRECT_TENSOR_ALIGNMENT - 1)
        // tensor_downloader._DIRECT_TENSOR_ALIGNMENT
        * tensor_downloader._DIRECT_TENSOR_ALIGNMENT
    )
    payload = bytearray(
        tensor_downloader._DIRECT_TENSOR_HEADER.pack(tensor_downloader._DIRECT_TENSOR_MAGIC, len(metadata), header_size)
        + metadata
        + bytes(header_size - unpadded_size)
    )

    with pytest.raises(ValueError, match="exceeds the payload size"):
        TensorConsumer(None, {}).consume_direct_chunk(payload)


def test_small_tensor_stays_on_legacy_inline_path():
    item = _serialize_tensor_item("weight", torch.arange(1024, dtype=torch.float32), stream_tensor=True)

    assert isinstance(item, bytes)


def test_negotiated_single_receiver_produces_direct_large_tensor():
    tensor = torch.arange(3 * 1024 * 1024, dtype=torch.float32)
    downloadable = TensorDownloadable({"weight": tensor}, max_chunk_size=1024)
    downloadable.num_receivers = 1
    consumer = TensorConsumer(None, {})

    rc, items, state = downloadable.produce(consumer.get_initial_state(), "receiver")
    payload = bytearray(b"".join(items[0].data))
    received = consumer.consume_direct_chunk(payload)
    result = consumer.consume_items(received, None)

    assert rc == ProduceRC.OK
    assert torch.equal(result["weight"], tensor)
    assert isinstance(received[0], _StreamedTensorItem)
    assert state[tensor_downloader._TENSOR_STREAM_STATE_KEY] == tensor_downloader._TENSOR_STREAM_MEMORY_V1


def test_multi_receiver_stays_on_legacy_path():
    tensor = torch.arange(3 * 1024 * 1024, dtype=torch.float32)
    downloadable = TensorDownloadable({"weight": tensor}, max_chunk_size=1024)
    downloadable.num_receivers = 2

    rc, items, state = downloadable.produce(TensorConsumer(None, {}).get_initial_state(), "receiver")

    assert rc == ProduceRC.OK
    assert isinstance(items[0], bytes)
    assert tensor_downloader._TENSOR_STREAM_STATE_KEY not in state


def test_legacy_receiver_state_keeps_new_sender_on_legacy_path():
    tensor = torch.arange(3 * 1024 * 1024, dtype=torch.float32)
    downloadable = TensorDownloadable({"weight": tensor}, max_chunk_size=1024)
    downloadable.num_receivers = 1

    _, items, _ = downloadable.produce(None, "receiver")

    assert isinstance(items[0], bytes)


def test_unsupported_dtype_stays_on_legacy_path(monkeypatch):
    monkeypatch.setattr(tensor_downloader, "TEN_MEGA", 0)
    tensor = torch.arange(4, dtype=torch.float32).to(torch.complex64)

    item = _serialize_tensor_item("weight", tensor, stream_tensor=True)

    assert isinstance(item, bytes)


def test_disk_consumer_does_not_advertise_direct_memory(tmp_path):
    consumer = DiskTensorConsumer(str(tmp_path))
    assert consumer.get_initial_state() is None
    consumer.release()


def test_direct_item_is_exclusive_with_large_chunk_budget(monkeypatch):
    monkeypatch.setattr(tensor_downloader, "TEN_MEGA", 1024)
    tensors = {
        "small_before": torch.ones(8),
        "large": torch.ones(1024),
        "small_after": torch.ones(8),
    }
    downloadable = TensorDownloadable(tensors, max_chunk_size=1024 * 1024)
    downloadable.num_receivers = 1
    state = TensorConsumer(None, {}).get_initial_state()

    _, first, state = downloadable.produce(state, "receiver")
    _, second, state = downloadable.produce(state, "receiver")
    _, third, _ = downloadable.produce(state, "receiver")

    assert len(first) == 1 and isinstance(first[0], bytes)
    assert len(second) == 1 and isinstance(second[0], DirectDownloadChunk)
    assert len(third) == 1 and isinstance(third[0], bytes)


def test_non_contiguous_tensor_retains_safetensors_validation():
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()

    with pytest.raises(ValueError, match="non contiguous"):
        _serialize_tensor_item("weight", tensor, stream_tensor=True)


@pytest.mark.timeout(30)
def test_direct_tensor_round_trip_over_real_cell_is_writable_and_owned(monkeypatch):
    monkeypatch.setattr(tensor_downloader, "TEN_MEGA", 0)
    direct_buffers = []
    original_consume_direct = TensorConsumer.consume_direct_chunk

    def track_direct_buffer(self, data):
        view = memoryview(data)
        direct_buffers.append((view.readonly, view.c_contiguous))
        return original_consume_direct(self, data)

    monkeypatch.setattr(TensorConsumer, "consume_direct_chunk", track_direct_buffer)

    port = get_open_ports(1)[0]
    server_name = "server"
    server = Cell(server_name, f"tcp://localhost:{port}", secure=False, credentials={})
    client = Cell(f"tensor-client-{port}", f"tcp://localhost:{port}", secure=False, credentials={})
    downloader = None
    server.core_cell.start()
    client.core_cell.start()
    try:
        deadline = time.monotonic() + 5.0
        while not client.core_cell.is_cell_connected(server_name) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert client.core_cell.is_cell_connected(server_name)

        source = torch.arange(4096, dtype=torch.float32)
        downloader = ObjectDownloader(cell=server, timeout=20.0, num_receivers=1)
        ref_id = add_tensors(downloader, {"weight": source}, max_chunk_size=1024)

        error, result = download_tensors(server_name, ref_id, 10.0, client)

        assert error is None
        assert direct_buffers == [(False, True)]
        tensor = result["weight"]
        del result
        gc.collect()
        assert torch.equal(tensor, source)
        tensor.add_(1)
        assert torch.equal(tensor, source + 1)
    finally:
        if downloader is not None:
            downloader.delete_transaction()
        client.core_cell.stop()
        server.core_cell.stop()
        CoreCell.ALL_CELLS.pop(client.core_cell.get_fqcn(), None)
        CoreCell.ALL_CELLS.pop(server.core_cell.get_fqcn(), None)
