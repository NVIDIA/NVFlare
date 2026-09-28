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
import os
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
from nvflare.fuel.f3.streaming.download_service import download_object
from nvflare.fuel.f3.streaming.obj_downloader import ObjectDownloader
from nvflare.fuel.f3.streaming.stream_types import DownloadCancelled
from nvflare.fuel.f3.streaming.stream_utils import stream_thread_pool

from .lazy_tensor_dict import LazyTensorDict, _cleanup_temp_dir, _LazyRef, read_safetensors_metadata

_TWO_MB = 2 * 1024 * 1024
_ACTIVE_DISK_TENSOR_CONSUMERS = weakref.WeakSet()
_ACTIVE_DISK_TENSOR_CONSUMERS_LOCK = threading.Lock()


def cleanup_active_disk_tensor_downloads(reason: str = "download aborted", root_dir: Optional[str] = None) -> None:
    """Clean partial tensor offload dirs still owned by active disk consumers.

    Args:
        reason: local cancellation reason recorded on each selected consumer.
        root_dir: when set, only clean consumers writing below this root.
    """
    with _ACTIVE_DISK_TENSOR_CONSUMERS_LOCK:
        consumers = list(_ACTIVE_DISK_TENSOR_CONSUMERS)

    for consumer in consumers:
        if root_dir is None or consumer.is_under_root(root_dir):
            consumer.cleanup(cancel_reason=reason)


class TensorDownloadable(CacheableObject):
    """Downloadable over a dict of tensors or disk-backed lazy tensor refs."""

    def __init__(self, tensors: dict, max_chunk_size: int):
        self.size = len(tensors)
        self.keys = list(tensors.keys())
        self._prefetch_lock = threading.Lock()
        self._prefetch_futures = {}
        self._released = False
        super().__init__(tensors, max_chunk_size)
        if any(isinstance(value, _LazyRef) for value in tensors.values()):
            # Lazy refs are re-read from disk for each receiver. A shared chunk cache would
            # otherwise grow toward the model size when receivers progress at different speeds.
            self.clear_cache()

    def get_item_count(self) -> int:
        return self.size

    def produce_item(self, index: int) -> bytes:
        key = self.keys[index]
        with self._prefetch_lock:
            future = self._prefetch_futures.pop(index, None)
        if future:
            return future.result()
        base_obj = self.base_obj
        if base_obj is None:
            raise RuntimeError(f"item {index} requested after tensors were released")
        tensor = base_obj[key]
        if isinstance(tensor, _LazyRef):
            tensor = tensor.materialize()
        return save_tensors({key: tensor})

    def prefetch_item(self, index: int):
        with self._prefetch_lock:
            if self._released or index in self._prefetch_futures:
                return
            base_obj = self.base_obj
            if base_obj is None:
                return
            key = self.keys[index]
            if isinstance(base_obj[key], _LazyRef):
                # Avoid retaining another tensor or serialized payload ahead of the receiver.
                return
            future = stream_thread_pool.submit(save_tensors, {key: base_obj[key]})
            if future:
                self._prefetch_futures[index] = future

    def get_item_size(self, index: int) -> Optional[int]:
        base_obj = self.base_obj
        if base_obj is None:
            return None
        value = base_obj[self.keys[index]]
        if isinstance(value, _LazyRef):
            return value.get_metadata().nbytes
        return value.numel() * value.element_size()

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

    def consume_items(self, items: List[Any], result: Any) -> Any:
        if not isinstance(items, list):
            raise TypeError(f"items must be list but got {type(items)}")
        if result is None:
            result = {}

        tensors = {}
        for item in items:
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


class DiskTensorConsumer(ItemConsumer):
    """Writes raw safetensors bytes to disk without deserializing to tensors."""

    def __init__(self, temp_dir: str):
        ItemConsumer.__init__(self)
        self._temp_dir = temp_dir
        self._cleaned = False
        self._cancel_reason = None
        self._file_counter = 0
        self.metadata = {}
        self._io_lock = threading.Lock()
        with _ACTIVE_DISK_TENSOR_CONSUMERS_LOCK:
            _ACTIVE_DISK_TENSOR_CONSUMERS.add(self)

    def release(self) -> None:
        # Serialize ownership transfer with cleanup: a cancelled download must
        # never return refs into files that finalization has already removed.
        with self._io_lock:
            if self._cancel_reason is not None:
                raise DownloadCancelled(self._cancel_reason)
            if not self.error:
                with _ACTIVE_DISK_TENSOR_CONSUMERS_LOCK:
                    _ACTIVE_DISK_TENSOR_CONSUMERS.discard(self)
                    self._cleaned = True

    def is_under_root(self, root_dir: str) -> bool:
        try:
            return os.path.commonpath(
                (os.path.realpath(self._temp_dir), os.path.realpath(root_dir))
            ) == os.path.realpath(root_dir)
        except (TypeError, ValueError):
            return False

    def cleanup(self, cancel_reason: Optional[str] = None) -> None:
        # Pipelined downloads can have a chunk write in progress while workflow
        # finalization aborts active consumers. Wait for that write to finish so
        # rmtree cannot race an open/create operation and leave a partial directory.
        with self._io_lock:
            with _ACTIVE_DISK_TENSOR_CONSUMERS_LOCK:
                if self._cleaned:
                    return
                if cancel_reason is not None and not self.error:
                    self._cancel_reason = cancel_reason
                    self.error = cancel_reason
                self._cleaned = True
                _ACTIVE_DISK_TENSOR_CONSUMERS.discard(self)

            _cleanup_temp_dir(self._temp_dir)

    def consume_items(self, items: List[Any], result: Any) -> Any:
        if not isinstance(items, list):
            raise TypeError(f"items must be list but got {type(items)}")
        if result is None:
            result = {}

        with self._io_lock:
            if self._cleaned:
                raise RuntimeError("tensor download was cleaned up")
            for item in items:
                metadata = read_safetensors_metadata(item)
                file_path = os.path.join(self._temp_dir, f"chunk_{self._file_counter}.safetensors")
                self._file_counter += 1
                with open(file_path, "wb") as f:
                    f.write(item)
                for key in metadata:
                    if key in result:
                        raise ValueError(
                            f"Duplicate tensor key '{key}' seen in multiple safetensors chunks; "
                            "streaming data may be malformed."
                        )
                    result[key] = (file_path, key)
                self.metadata.update(metadata)

        return result

    def download_failed(self, ref_id, reason: str):
        with self._io_lock:
            # A late chunk or transport error must not overwrite local cancellation.
            if self._cancel_reason is None:
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

    Raises: DownloadCancelled if workflow cleanup cancelled this download.
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
        if consumer._cancel_reason is not None:
            raise DownloadCancelled(consumer._cancel_reason)
        raise

    consumer.release()
    if consumer.error:
        consumer.cleanup()
        return consumer.error, None

    key_to_file = consumer.result if consumer.result is not None else {}
    return None, LazyTensorDict(key_to_file=key_to_file, temp_dir=temp_dir, metadata=consumer.metadata)
