# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import json
import os
import struct
import sys
import tempfile
import threading
import weakref
from typing import Any, List, Optional, Tuple

import torch
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

from nvflare.app_common.utils.tensor_disk_offload_context import _TENSOR_DISK_OFFLOAD_ROOT_DIR
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.streaming.cacheable import CacheableObject, ItemConsumer
from nvflare.fuel.f3.streaming.download_service import DirectDownloadChunk, ProduceRC, download_object
from nvflare.fuel.f3.streaming.obj_downloader import ObjectDownloader
from nvflare.fuel.f3.streaming.stream_utils import stream_thread_pool
from nvflare.fuel.utils.fobs.datum import TEN_MEGA

from .lazy_tensor_dict import LazyTensorDict, _cleanup_temp_dir

_TWO_MB = 2 * 1024 * 1024
_ACTIVE_DISK_TENSOR_CONSUMERS = weakref.WeakSet()
_ACTIVE_DISK_TENSOR_CONSUMERS_LOCK = threading.Lock()
_TENSOR_STREAM_STATE_KEY = "__nvflare_tensor_stream__"
_TENSOR_STREAM_MEMORY_V1 = "direct_memory_v1"
_DIRECT_TENSOR_MAGIC = b"NVTDIR01"
_DIRECT_TENSOR_HEADER = struct.Struct("<8sII")
_DIRECT_TENSOR_ALIGNMENT = 64
_DIRECT_TENSOR_MAX_METADATA = 1024 * 1024

_SAFETENSORS_DTYPE_NAMES = (
    ("float64", "F64"),
    ("float32", "F32"),
    ("float16", "F16"),
    ("bfloat16", "BF16"),
    ("int64", "I64"),
    ("int32", "I32"),
    ("int16", "I16"),
    ("int8", "I8"),
    ("uint8", "U8"),
    ("bool", "BOOL"),
)
_SAFETENSORS_DTYPES = {
    dtype: code
    for torch_name, code in _SAFETENSORS_DTYPE_NAMES
    if (dtype := getattr(torch, torch_name, None)) is not None
}


class _StreamedTensorItem:
    """Receiver-created marker that cannot be supplied as an ordinary data item."""

    __slots__ = ("key", "tensor", "wire_size")

    def __init__(self, key: str, tensor: torch.Tensor, wire_size: int):
        self.key = key
        self.tensor = tensor
        self.wire_size = wire_size

    def __len__(self) -> int:
        return self.wire_size


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate direct tensor metadata key {key!r}")
        result[key] = value
    return result


def _can_stream_tensor_directly(tensor: torch.Tensor) -> bool:
    """Whether a tensor can be represented by the direct-memory protocol."""
    return (
        sys.byteorder == "little"
        and tensor.layout == torch.strided
        and tensor.device.type == "cpu"
        and tensor.is_contiguous()
        and tensor.dtype in _SAFETENSORS_DTYPES
        and not tensor.is_conj()
        and not getattr(tensor, "is_neg", lambda: False)()
    )


def _serialize_tensor_item(key: str, tensor: torch.Tensor, stream_tensor: bool = False):
    """Create a snapshot for one tensor download item.

    Negotiated large tensors use a raw bytes-like reply so F3 receives directly
    into writable storage. Legacy peers and small/unsupported tensors retain
    the existing safetensors representation.
    """
    if (
        not stream_tensor
        or not isinstance(key, str)
        or key == "__metadata__"
        or tensor.numel() * tensor.element_size() < TEN_MEGA
        or not _can_stream_tensor_directly(tensor)
    ):
        return save_tensors({key: tensor})

    metadata = {
        "key": key,
        "dtype": _SAFETENSORS_DTYPES[tensor.dtype],
        "shape": list(tensor.shape),
        "size": tensor.numel() * tensor.element_size(),
    }
    metadata_bytes = json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(metadata_bytes) > _DIRECT_TENSOR_MAX_METADATA:
        return save_tensors({key: tensor})
    unpadded_size = _DIRECT_TENSOR_HEADER.size + len(metadata_bytes)
    header_size = (unpadded_size + _DIRECT_TENSOR_ALIGNMENT - 1) // _DIRECT_TENSOR_ALIGNMENT
    header_size *= _DIRECT_TENSOR_ALIGNMENT
    prefix = _DIRECT_TENSOR_HEADER.pack(_DIRECT_TENSOR_MAGIC, len(metadata_bytes), header_size)
    prefix += metadata_bytes + bytes(header_size - unpadded_size)
    snapshot = tensor.detach().clone(memory_format=torch.contiguous_format)
    body = memoryview(snapshot.reshape(-1).view(torch.uint8).numpy())
    return DirectDownloadChunk([prefix, body])


def cleanup_active_disk_tensor_downloads(reason: str = "download aborted") -> None:
    """Clean partial tensor offload dirs still owned by active disk consumers."""
    with _ACTIVE_DISK_TENSOR_CONSUMERS_LOCK:
        consumers = list(_ACTIVE_DISK_TENSOR_CONSUMERS)

    for consumer in consumers:
        consumer.download_failed("active_disk_tensor_download", reason)


class TensorDownloadable(CacheableObject):

    def __init__(self, tensors: dict[str, torch.Tensor], max_chunk_size: int):
        self.size = len(tensors)
        self.keys = list(tensors.keys())
        self._prefetch_lock = threading.Lock()
        self._prefetch_futures = {}
        self._released = False
        self._stream_tensors = None
        super().__init__(tensors, max_chunk_size)

    def get_item_count(self) -> int:
        return self.size

    def produce(self, state: dict, requester: str):
        requested_mode = state.get(_TENSOR_STREAM_STATE_KEY) if isinstance(state, dict) else None
        with self._prefetch_lock:
            if self._stream_tensors is None:
                self._stream_tensors = bool(self.num_receivers == 1 and requested_mode == _TENSOR_STREAM_MEMORY_V1)
            stream_tensors = self._stream_tensors

        rc, data, new_state = super().produce(state, requester)
        if stream_tensors and rc == ProduceRC.OK:
            new_state = dict(new_state)
            new_state[_TENSOR_STREAM_STATE_KEY] = _TENSOR_STREAM_MEMORY_V1
        return rc, data, new_state

    def produce_item(self, index: int):
        key = self.keys[index]
        with self._prefetch_lock:
            future = self._prefetch_futures.pop(index, None)
            stream_tensors = bool(self._stream_tensors)
        if future:
            return future.result()
        base_obj = self.base_obj
        if base_obj is None:
            raise RuntimeError(f"item {index} requested after tensors were released")
        return _serialize_tensor_item(key, base_obj[key], stream_tensors)

    def prefetch_item(self, index: int):
        with self._prefetch_lock:
            if self._released or index in self._prefetch_futures:
                return
            base_obj = self.base_obj
            if base_obj is None:
                return
            key = self.keys[index]
            tensor = base_obj[key]
            future = stream_thread_pool.submit(_serialize_tensor_item, key, tensor, bool(self._stream_tensors))
            if future:
                self._prefetch_futures[index] = future

    def get_item_size(self, index: int) -> Optional[int]:
        base_obj = self.base_obj
        if base_obj is None:
            return None
        tensor = base_obj[self.keys[index]]
        return tensor.numel() * tensor.element_size()

    def is_item_exclusive(self, index: int, item: Any = None) -> bool:
        if item is not None:
            return isinstance(item, DirectDownloadChunk)

        base_obj = self.base_obj
        if base_obj is None:
            return False
        key = self.keys[index]
        tensor = base_obj[key]
        size = tensor.numel() * tensor.element_size()
        with self._prefetch_lock:
            stream_tensors = bool(self._stream_tensors)
        return bool(
            stream_tensors
            and isinstance(key, str)
            and key != "__metadata__"
            and size >= TEN_MEGA
            and _can_stream_tensor_directly(tensor)
        )

    def release(self):
        with self._prefetch_lock:
            self._released = True
            futures = list(self._prefetch_futures.values())
            self._prefetch_futures.clear()
        for future in futures:
            future.cancel()
        super().release()


class TensorConsumer(ItemConsumer):

    def __init__(self, tensors_received_cb, cb_kwargs):
        ItemConsumer.__init__(self)
        self.tensors_received_cb = tensors_received_cb
        self.cb_kwargs = cb_kwargs
        if tensors_received_cb is not None and not callable(tensors_received_cb):
            raise ValueError("tensors_received_cb must be callable")

    def get_initial_state(self) -> Optional[dict]:
        if sys.byteorder != "little":
            return None
        return {_TENSOR_STREAM_STATE_KEY: _TENSOR_STREAM_MEMORY_V1}

    def consume_direct_chunk(self, data) -> List[_StreamedTensorItem]:
        if sys.byteorder != "little":
            raise ValueError("direct tensor replies require a little-endian receiver")

        buffer = memoryview(data)
        if not buffer.c_contiguous:
            raise ValueError("direct tensor payload must be C-contiguous")
        if buffer.readonly:
            raise ValueError("direct tensor payload must be writable")
        buffer = buffer.cast("B")
        if len(buffer) < _DIRECT_TENSOR_HEADER.size:
            raise ValueError("direct tensor payload is too short")
        magic, metadata_size, header_size = _DIRECT_TENSOR_HEADER.unpack_from(buffer)
        if magic != _DIRECT_TENSOR_MAGIC:
            raise ValueError("invalid direct tensor magic")
        expected_header_size = (
            (_DIRECT_TENSOR_HEADER.size + metadata_size + _DIRECT_TENSOR_ALIGNMENT - 1)
            // _DIRECT_TENSOR_ALIGNMENT
            * _DIRECT_TENSOR_ALIGNMENT
        )
        if (
            metadata_size > _DIRECT_TENSOR_MAX_METADATA
            or header_size != expected_header_size
            or header_size > len(buffer)
        ):
            raise ValueError("invalid direct tensor header size")
        if any(buffer[_DIRECT_TENSOR_HEADER.size + metadata_size : header_size]):
            raise ValueError("invalid direct tensor header padding")
        try:
            metadata = json.loads(
                bytes(buffer[_DIRECT_TENSOR_HEADER.size : _DIRECT_TENSOR_HEADER.size + metadata_size]),
                object_pairs_hook=_reject_duplicate_json_keys,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as ex:
            raise ValueError("invalid direct tensor metadata") from ex
        if not isinstance(metadata, dict) or set(metadata) != {"key", "dtype", "shape", "size"}:
            raise ValueError("invalid direct tensor metadata schema")

        key = metadata.get("key")
        if not isinstance(key, str) or key == "__metadata__":
            raise ValueError(f"invalid direct tensor key {key!r}")
        dtype_code = metadata.get("dtype")
        dtype_by_code = {code: dtype for dtype, code in _SAFETENSORS_DTYPES.items()}
        dtype = dtype_by_code.get(dtype_code)
        if dtype is None:
            raise ValueError(f"unsupported direct tensor dtype {dtype_code!r}")

        shape = metadata.get("shape")
        if not isinstance(shape, list) or len(shape) > 64 or any(type(dim) is not int or dim < 0 for dim in shape):
            raise ValueError(f"invalid direct tensor shape {shape!r}")

        body_size = len(buffer) - header_size
        if type(metadata.get("size")) is not int or metadata["size"] != body_size:
            raise ValueError(f"direct tensor size {metadata.get('size')!r} does not match {body_size}-byte payload")

        element_size = torch.empty((), dtype=dtype).element_size()
        max_numel = body_size // element_size
        numel = 0 if 0 in shape else 1
        if numel:
            for dim in shape:
                if dim and numel > max_numel // dim:
                    raise ValueError("direct tensor shape exceeds the payload size")
                numel *= dim
        expected_size = numel * element_size
        if expected_size != body_size:
            raise ValueError(f"direct tensor shape and dtype require {expected_size} bytes but payload has {body_size}")

        try:
            if numel:
                tensor = torch.frombuffer(buffer, dtype=dtype, count=numel, offset=header_size).reshape(tuple(shape))
            else:
                tensor = torch.empty(tuple(shape), dtype=dtype)
        except (RuntimeError, ValueError) as ex:
            raise ValueError(f"invalid direct tensor shape {shape!r}") from ex
        return [_StreamedTensorItem(key, tensor, len(buffer))]

    def consume_items(self, items: List[Any], result: Any) -> Any:
        if not isinstance(items, list):
            raise TypeError(f"items must be list but got {type(items)}")
        if result is None:
            result = {}

        tensors = {}
        for item in items:
            if isinstance(item, _StreamedTensorItem):
                td = {item.key: item.tensor}
            else:
                td = load_tensors(item)
            if not isinstance(td, dict):
                raise ValueError("cannot load received bytes to tensors")
            tensors.update(td)

        if self.tensors_received_cb:
            cb_result = self.tensors_received_cb(tensors, **self.cb_kwargs)
            if isinstance(cb_result, dict):
                result.update(cb_result)
        else:
            result.update(tensors)
        return result


def add_tensors(
    downloader: ObjectDownloader,
    tensors: dict[str, torch.Tensor],
    max_chunk_size: int = _TWO_MB,
) -> str:
    """Add tensors to be downloaded to the specified downloader.

    Args:
        downloader: the downloader to add tensors to.
        tensors: state dict to be downloaded
        max_chunk_size: max chunk size

    Returns: reference id for the state dict.

    """
    obj = TensorDownloadable(tensors, max_chunk_size)
    return downloader.add_object(obj)


def download_tensors(
    from_fqcn: str,
    ref_id: str,
    per_request_timeout: float,
    cell: Cell,
    secure=False,
    optional=False,
    abort_signal=None,
    tensors_received_cb=None,
    progress_cb=None,
    **cb_kwargs,
) -> Tuple[str, Optional[dict[str, torch.Tensor]]]:
    """Download the referenced state dict from the source.

    Args:
        from_fqcn: FQCN of the data source.
        ref_id: reference ID of the state dict to be downloaded.
        per_request_timeout: timeout for requests sent to the data source.
        cell: cell to be used for communicating to the data source.
        secure: P2P private mode for communication
        optional: supress log messages of communication
        abort_signal: signal for aborting download.
        tensors_received_cb: the callback to be called when one set of tensors are received

    Returns: tuple of (error message if any, downloaded state dict).

    """
    consumer = TensorConsumer(tensors_received_cb, cb_kwargs)
    download_object(
        from_fqcn=from_fqcn,
        ref_id=ref_id,
        consumer=consumer,
        per_request_timeout=per_request_timeout,
        cell=cell,
        secure=secure,
        optional=optional,
        abort_signal=abort_signal,
        progress_cb=progress_cb,
    )
    return consumer.error, consumer.result


def _extract_safetensors_keys(data: bytes) -> list[str]:
    """Extract tensor key names from safetensors header without deserializing tensors."""
    if len(data) < 8:
        raise ValueError("Invalid safetensors data: too short")

    header_size = struct.unpack("<Q", data[:8])[0]
    if header_size == 0:
        raise ValueError("Invalid safetensors data: empty header")

    header_end = 8 + header_size
    if header_end > len(data):
        raise ValueError("Invalid safetensors data: header size exceeds payload length")

    try:
        header = json.loads(data[8:header_end])
    except Exception as e:
        raise ValueError("Invalid safetensors data: invalid JSON header") from e

    if not isinstance(header, dict):
        raise ValueError("Invalid safetensors data: header must be JSON object")

    return [k for k in header.keys() if k != "__metadata__"]


class DiskTensorConsumer(ItemConsumer):
    """Writes raw safetensors bytes to disk without deserializing to tensors."""

    def __init__(self, temp_dir: str):
        ItemConsumer.__init__(self)
        self._temp_dir = temp_dir
        self._cleaned = False
        self._file_counter = 0
        self._io_lock = threading.Lock()
        with _ACTIVE_DISK_TENSOR_CONSUMERS_LOCK:
            _ACTIVE_DISK_TENSOR_CONSUMERS.add(self)

    def release(self) -> None:
        with _ACTIVE_DISK_TENSOR_CONSUMERS_LOCK:
            _ACTIVE_DISK_TENSOR_CONSUMERS.discard(self)
            self._cleaned = True

    def cleanup(self) -> None:
        # Pipelined downloads can have a chunk write in progress while workflow
        # finalization aborts active consumers. Wait for that write to finish so
        # rmtree cannot race an open/create operation and leave a partial directory.
        with self._io_lock:
            with _ACTIVE_DISK_TENSOR_CONSUMERS_LOCK:
                if self._cleaned:
                    return
                self._cleaned = True
                _ACTIVE_DISK_TENSOR_CONSUMERS.discard(self)

            _cleanup_temp_dir(self._temp_dir)

    def consume_items(self, items: List[Any], result: Any) -> Any:
        if not isinstance(items, list):
            raise TypeError(f"items must be list but got {type(items)}")
        if result is None:
            result = {}

        with self._io_lock:
            for item in items:
                keys = _extract_safetensors_keys(item)
                file_path = os.path.join(self._temp_dir, f"chunk_{self._file_counter}.safetensors")
                self._file_counter += 1
                with open(file_path, "wb") as f:
                    f.write(item)
                for key in keys:
                    if key in result:
                        raise ValueError(
                            f"Duplicate tensor key '{key}' seen in multiple safetensors chunks; "
                            "streaming data may be malformed."
                        )
                    result[key] = (file_path, key)

        return result

    def download_failed(self, ref_id, reason: str):
        super().download_failed(ref_id, reason)
        # Eager cleanup on download callback error; the outer caller may also
        # attempt cleanup via consumer.error path. Double cleanup is intentional
        # and safe because _cleanup_temp_dir handles already-removed paths.
        self.cleanup()


def download_tensors_to_disk(
    from_fqcn: str,
    ref_id: str,
    per_request_timeout: float,
    cell: Cell,
    secure=False,
    optional=False,
    abort_signal=None,
    progress_cb=None,
    root_dir: Optional[str] = None,
) -> Tuple[str, Optional[LazyTensorDict]]:
    """Download tensors to disk instead of memory.

    Args:
        root_dir: optional call-scoped destination root. When omitted, use the
            root configured on the Cell for backward compatibility.

    Returns: tuple of (error message if any, LazyTensorDict for lazy access).
    """
    if root_dir is None:
        root_dir = cell.get_fobs_context().get(_TENSOR_DISK_OFFLOAD_ROOT_DIR)
    if not root_dir:
        raise RuntimeError(f"{_TENSOR_DISK_OFFLOAD_ROOT_DIR} is not set in FOBS context")
    temp_dir = tempfile.mkdtemp(prefix="nvflare_tensors_", dir=root_dir)

    consumer = DiskTensorConsumer(temp_dir)
    try:
        download_object(
            from_fqcn=from_fqcn,
            ref_id=ref_id,
            consumer=consumer,
            per_request_timeout=per_request_timeout,
            cell=cell,
            secure=secure,
            optional=optional,
            abort_signal=abort_signal,
            progress_cb=progress_cb,
        )
    except Exception:
        consumer.cleanup()
        raise

    if consumer.error:
        consumer.cleanup()
        return consumer.error, None

    key_to_file = consumer.result if consumer.result is not None else {}
    consumer.release()
    return None, LazyTensorDict(key_to_file=key_to_file, temp_dir=temp_dir)
