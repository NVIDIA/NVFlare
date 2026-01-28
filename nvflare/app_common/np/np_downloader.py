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
from io import BytesIO
from typing import Any, List, Optional, Tuple

import numpy as np

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.streaming.cacheable import CacheableObject, ItemConsumer
from nvflare.fuel.f3.streaming.download_service import download_object
from nvflare.fuel.f3.streaming.obj_downloader import ObjectDownloader

_TWO_MB = 2 * 1024 * 1024


class ArrayDownloadable(CacheableObject):
    """Downloadable for NumPy arrays using reference-based storage for memory efficiency.

    IMPORTANT: This class stores arrays by reference (not copies) to avoid memory overhead.
    The dict of arrays is snapshotted at creation time to ensure slow clients get the correct
    model even with min_responses < total_clients scenarios.

    Safe patterns:
    - Dict updates: Replacing dict entries (params["key"] = new_array) is safe - slow clients
      get the snapshotted reference
    - Client side: flare.send() is synchronous - user code blocks until after serialization
    - Server side: Dict replacement or entry updates are safe due to snapshot

    Unsafe pattern (avoid in custom code):
    - In-place array modification: Modifying array values while downloads are in progress
      (e.g., array[:] = 0, array += value) is unsafe as arrays are referenced

    CRITICAL WARNING for min_responses < total_clients:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    When updating model.params after broadcast, ALWAYS use assignment, NEVER in-place:

    ✓ SAFE:   model.params["key"] = model.params["key"] + update
    ✗ UNSAFE: model.params["key"] += update

    In-place operations (+=, [:]=, np.add(..., out=arr)) will corrupt slow clients' models!
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    """

    def __init__(self, arrays: dict[str, np.ndarray], max_chunk_size: int):
        # Create shallow copy of dict to snapshot array references at broadcast time.
        # This ensures slow clients get the correct model even if server updates the
        # original dict before they download (min_responses < total_clients scenario).
        # The arrays themselves are still referenced (not copied) for memory efficiency.
        arrays_snapshot = {k: v for k, v in arrays.items()}
        self.size = len(arrays_snapshot)
        self.keys = list(arrays_snapshot.keys())
        super().__init__(arrays_snapshot, max_chunk_size)

    def get_item_count(self) -> int:
        return self.size

    def produce_item(self, index: int) -> bytes:
        """Serialize an array by accessing it from the original reference.

        Note: This accesses self.base_obj[key] which is a reference to the original array.
        This is safe because NVFlare workflows ensure no concurrent modifications during serialization.
        """
        key = self.keys[index]
        arrays_to_send = {key: self.base_obj[key]}
        stream = BytesIO()
        np.savez(allow_pickle=False, file=stream, **arrays_to_send)
        return stream.getvalue()


class ArrayConsumer(ItemConsumer):

    def __init__(self, arrays_received_cb, cb_kwargs):
        ItemConsumer.__init__(self)
        self.arrays_received_cb = arrays_received_cb
        self.cb_kwargs = cb_kwargs
        if arrays_received_cb is not None and not callable(arrays_received_cb):
            raise ValueError("arrays_received_cb must be callable")

    @staticmethod
    def _to_dict(item: bytes) -> dict:
        result = {}
        stream = BytesIO(item)
        with np.load(stream, allow_pickle=False) as npz_obj:
            for k in npz_obj.files:
                result[k] = npz_obj[k]
        return result

    def consume_items(self, items: List[Any], result: Any) -> Any:
        assert isinstance(items, list)
        if result is None:
            result = {}

        arrays = {}
        for item in items:
            td = self._to_dict(item)
            if not isinstance(td, dict):
                raise ValueError("cannot load received bytes to arrays")
            arrays.update(td)

        if self.arrays_received_cb is not None:
            cb_result = self.arrays_received_cb(arrays, **self.cb_kwargs)
            if isinstance(cb_result, dict):
                result.update(cb_result)
        else:
            result.update(arrays)
        return result


def add_arrays(
    downloader: ObjectDownloader,
    arrays: dict[str, np.ndarray],
    max_chunk_size: int = _TWO_MB,
) -> str:
    """Add arrays to be downloaded to the specified downloader.

    Args:
        downloader: the downloader to add arrays to.
        arrays: arrays to be downloaded
        max_chunk_size: max chunk size

    Returns: reference id for the arrays.

    """
    obj = ArrayDownloadable(arrays, max_chunk_size)
    return downloader.add_object(obj)


def download_arrays(
    from_fqcn: str,
    ref_id: str,
    per_request_timeout: float,
    cell: Cell,
    secure=False,
    optional=False,
    abort_signal=None,
    arrays_received_cb=None,
    **cb_kwargs,
) -> Tuple[str, Optional[dict[str, np.ndarray]]]:
    """Download the referenced arrays from the source.

    Args:
        from_fqcn: FQCN of the data source.
        ref_id: reference ID of the arrays to be downloaded.
        per_request_timeout: timeout for requests sent to the data source.
        cell: cell to be used for communicating to the data source.
        secure: P2P private mode for communication
        optional: suppress log messages of communication
        abort_signal: signal for aborting download.
        arrays_received_cb: the callback to be called when one set of arrays are received

    Returns: tuple of (error message if any, downloaded state dict).

    """
    consumer = ArrayConsumer(arrays_received_cb, cb_kwargs)
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
