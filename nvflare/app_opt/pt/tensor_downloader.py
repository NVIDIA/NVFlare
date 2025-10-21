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
from nvflare.fuel.f3.streaming.cacheable import CacheableObject, ItemConsumer
from nvflare.fuel.f3.streaming.obj_downloader import ObjDownloader, download_object


class TensorDownloadable(CacheableObject):

    def __init__(self, tensors: dict[str, torch.Tensor], num_items_per_chunk: int):
        self.tensors = tensors
        self.size = len(tensors)
        self.keys = list(tensors.keys())
        super().__init__(num_items_per_chunk)

    def get_item_count(self) -> int:
        return self.size

    def produce_item(self, index: int) -> Any:
        key = self.keys[index]
        tensor_to_send = {key: self.tensors[key]}
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


class TensorDownloader:

    @classmethod
    def new_transaction(
        cls,
        cell: Cell,
        num_receivers: int,
        timeout: float = 5.0,
        transaction_done_cb=None,
        **cb_kwargs,
    ):
        """Create a new tensor download transaction.

        Args:
            cell: the cell for communication with recipients
            num_receivers: number of receivers
            timeout: timeout for the transaction
            transaction_done_cb: CB to be called when the transaction is done
            **cb_kwargs: args to be passed to the CB

        Returns: transaction id

        The transaction_done_cb must follow this signature:

            cb(tx_id, tensors: List[dict[str, torch.Tensor], **cb_args)

        """
        return ObjDownloader.new_transaction(
            cell=cell,
            num_receivers=num_receivers,
            timeout=timeout,
            transaction_done_cb=cls._tx_done,
            cb_info=(transaction_done_cb, cb_kwargs),
        )

    @classmethod
    def _tx_done(cls, tx_id: str, status: str, objs, cb_info):
        transaction_done_cb, cb_kwargs = cb_info
        if transaction_done_cb:
            tensors = [obj.tensors for obj in objs]
            transaction_done_cb(tx_id, tensors, **cb_kwargs)

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
        obj = TensorDownloadable(tensors, num_tensors_per_chunk)
        return ObjDownloader.add_object(
            transaction_id=transaction_id,
            obj=obj,
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
