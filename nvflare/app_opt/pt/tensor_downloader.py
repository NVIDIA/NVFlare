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
from typing import Any, List, Optional, Tuple

import torch
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.streaming.cached_obj_downloader import CachedObjDownloader, CountableObject, ItemConsumer


class _TensorSource(CountableObject):

    def __init__(self, tensors: dict[str, torch.Tensor]):
        self.tensors = tensors
        self.size = len(tensors)
        self.keys = list(tensors.keys())

    def get_item_count(self) -> int:
        return self.size

    def produce_item(self, index: int) -> Any:
        key = self.keys[index]
        tensor_to_send = {key: self.tensors[key]}
        return save_tensors(tensor_to_send)


class _ChunkConsumer(ItemConsumer):

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


class TensorDownloader:

    @classmethod
    def new_transaction(
        cls,
        cell: Cell,
        num_receivers: int,
        timeout: float = 5.0,
        timeout_cb=None,
        **cb_kwargs,
    ):
        """Create a new tensor download transaction.

        Args:
            cell: the cell for communication with recipients
            num_receivers: number of receivers
            timeout: timeout for the transaction
            timeout_cb: CB to be called when the transaction is timed out
            **cb_kwargs: args to be passed to the CB

        Returns: transaction id

        The timeout_cb must follow this signature:

            cb(tx_id, tensors: List[dict[str, torch.Tensor], **cb_args)

        """
        return CachedObjDownloader.new_transaction(
            cell=cell,
            num_receivers=num_receivers,
            timeout=timeout,
            timeout_cb=cls._tx_timeout,
            cb_info=(timeout_cb, cb_kwargs),
        )

    @classmethod
    def _tx_timeout(cls, tx_id: str, objs: List[Any], cb_info: tuple):
        app_timeout_cb, cb_kwargs = cb_info
        if app_timeout_cb:
            sds = [obj.tensors for obj in objs]
            app_timeout_cb(tx_id, sds, **cb_kwargs)

    @classmethod
    def add_tensors(
        cls,
        transaction_id: str,
        tensors: dict[str, torch.Tensor],
        num_tensors_per_chunk: int = 1,
    ) -> str:
        """Add a file to be downloaded to the specified transaction.

        Args:
            transaction_id: ID of the transaction
            tensors: state dict to be downloaded
            num_tensors_per_chunk: number of tensors per chunk

        Returns: reference id for the state dict.

        """
        obj = _TensorSource(tensors)
        return CachedObjDownloader.add_object(
            transaction_id=transaction_id,
            obj=obj,
            num_items_per_chunk=num_tensors_per_chunk,
        )

    @classmethod
    def download_tensors(
        cls,
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
        consumer = _ChunkConsumer(tensors_received_cb, cb_kwargs)
        return CachedObjDownloader.download_object(
            from_fqcn=from_fqcn,
            ref_id=ref_id,
            item_consumer=consumer,
            per_request_timeout=per_request_timeout,
            cell=cell,
            secure=secure,
            optional=optional,
            abort_signal=abort_signal,
        )
