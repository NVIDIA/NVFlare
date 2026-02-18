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
import shutil
import struct
import tempfile
import uuid
from typing import Any, List, Optional, Tuple

import torch
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.streaming.cacheable import CacheableObject, ItemConsumer
from nvflare.fuel.f3.streaming.download_service import download_object
from nvflare.fuel.f3.streaming.obj_downloader import ObjectDownloader

from .lazy_tensor_dict import LazyTensorDict

_TWO_MB = 2 * 1024 * 1024


class TensorDownloadable(CacheableObject):

    def __init__(self, tensors: dict[str, torch.Tensor], max_chunk_size: int):
        self.size = len(tensors)
        self.keys = list(tensors.keys())
        super().__init__(tensors, max_chunk_size)

    def get_item_count(self) -> int:
        return self.size

    def produce_item(self, index: int) -> bytes:
        key = self.keys[index]
        tensor_to_send = {key: self.base_obj[key]}
        return save_tensors(tensor_to_send)


class TensorConsumer(ItemConsumer):

    def __init__(self, tensors_received_cb, cb_kwargs):
        ItemConsumer.__init__(self)
        self.tensors_received_cb = tensors_received_cb
        self.cb_kwargs = cb_kwargs
        if tensors_received_cb is not None and not callable(tensors_received_cb):
            raise ValueError("tensors_received_cb must be callable")

    def consume_items(self, items: List[Any], result: Any) -> Any:
        assert isinstance(items, list)
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
    )
    return consumer.error, consumer.result


def _extract_safetensors_keys(data: bytes) -> list[str]:
    """Extract tensor key names from safetensors header without deserializing tensors."""
    if len(data) < 8:
        raise ValueError("Invalid safetensors data: too short")
    header_size = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8 : 8 + header_size])
    return [k for k in header.keys() if k != "__metadata__"]


class DiskTensorConsumer(ItemConsumer):
    """Writes raw safetensors bytes to disk without deserializing to tensors."""

    def __init__(self, temp_dir: str):
        ItemConsumer.__init__(self)
        self._temp_dir = temp_dir
        self._file_counter = 0

    def consume_items(self, items: List[Any], result: Any) -> Any:
        assert isinstance(items, list)
        if result is None:
            result = {}

        for item in items:
            keys = _extract_safetensors_keys(item)
            file_path = os.path.join(self._temp_dir, f"chunk_{self._file_counter}.safetensors")
            self._file_counter += 1
            with open(file_path, "wb") as f:
                f.write(item)
            for key in keys:
                result[key] = (file_path, key)

        return result

    def download_failed(self, ref_id, reason: str):
        super().download_failed(ref_id, reason)
        shutil.rmtree(self._temp_dir, ignore_errors=True)


def download_tensors_to_disk(
    from_fqcn: str,
    ref_id: str,
    per_request_timeout: float,
    cell: Cell,
    secure=False,
    optional=False,
    abort_signal=None,
) -> Tuple[str, Optional[LazyTensorDict]]:
    """Download tensors to disk instead of memory.

    Returns: tuple of (error message if any, LazyTensorDict for lazy access).
    """
    temp_dir = os.path.join(tempfile.gettempdir(), f"nvflare_tensors_{uuid.uuid4().hex}")
    os.makedirs(temp_dir, exist_ok=True)

    consumer = DiskTensorConsumer(temp_dir)
    download_object(
        from_fqcn=from_fqcn,
        ref_id=ref_id,
        consumer=consumer,
        per_request_timeout=per_request_timeout,
        cell=cell,
        secure=secure,
        optional=optional,
        abort_signal=abort_signal,
    )

    if consumer.error:
        return consumer.error, None

    key_to_file = consumer.result if consumer.result else {}
    return None, LazyTensorDict(key_to_file=key_to_file, temp_dir=temp_dir)
