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
from typing import Any, List, Optional

import torch
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.streaming.obj_downloader import Consumer, ObjDownloader, Producer, ProduceRC, download_object


class _StateKey:
    NUM_RECEIVED_TENSORS = "num_received_tensors"
    NUM_TENSORS_INCLUDED = "num_tensors_included"


class _SourceTensor:

    def __init__(self, state_dict: dict[str, torch.Tensor]):
        self.state_dict = state_dict
        self.size = len(state_dict)
        self.keys = list(state_dict.keys())


class _ChunkProducer(Producer):

    def __init__(self, num_tensors_per_chunk=1):
        Producer.__init__(self)
        self.num_tensors_per_chunk = num_tensors_per_chunk

    def produce(self, ref_id: str, obj: _SourceTensor, state: dict, requester: str) -> (str, Any, dict):
        assert isinstance(obj, _SourceTensor)
        num_received_tensors = 0
        if state:
            num_received_tensors = state.get(_StateKey.NUM_RECEIVED_TENSORS, 0)

        if not isinstance(num_received_tensors, int) or num_received_tensors < 0:
            self.logger.error(f"bad {_StateKey.NUM_RECEIVED_TENSORS} {num_received_tensors} from {requester}")
            return ProduceRC.ERROR, None, None

        if num_received_tensors >= obj.size:
            # already done
            return ProduceRC.EOF, None, None

        start = num_received_tensors
        end = min(num_received_tensors + self.num_tensors_per_chunk, obj.size)
        tensor_to_send = {}
        for i in range(start, end):
            next_key = obj.keys[i]
            tensor_to_send[next_key] = obj.state_dict[next_key]

        chunk = save_tensors(tensor_to_send)
        self.logger.debug(f"{num_received_tensors=}; sending {len(chunk)} bytes")
        return ProduceRC.OK, chunk, {_StateKey.NUM_TENSORS_INCLUDED: len(tensor_to_send)}


class _ChunkConsumer(Consumer):

    def __init__(self, tensors_received_cb, cb_kwargs):
        Consumer.__init__(self)
        self.num_tensors_received = 0
        self.result = {}
        self.error = None
        self.tensors_received_cb = tensors_received_cb
        self.cb_kwargs = cb_kwargs
        if tensors_received_cb is not None and not callable(tensors_received_cb):
            raise ValueError("tensors_received_cb must be callable")

    def consume(self, ref_id, state: dict, data: Any) -> dict:
        assert isinstance(data, bytes)
        td = load_tensors(data)
        if not isinstance(td, dict):
            raise ValueError("cannot load received bytes to tensors")

        num_tensors_included = state.get(_StateKey.NUM_TENSORS_INCLUDED, 0)
        if num_tensors_included != len(td):
            raise ValueError(f"tensor count mismatch: {num_tensors_included=}, received={len(td)}")

        if self.tensors_received_cb:
            result = self.tensors_received_cb(td, **self.cb_kwargs)
            if isinstance(result, dict):
                self.result.update(result)
        else:
            self.result.update(td)
        self.num_tensors_received += num_tensors_included
        self.logger.debug(f"received {len(td)} tensor(s)")
        return {_StateKey.NUM_RECEIVED_TENSORS: self.num_tensors_received}

    def download_failed(self, ref_id, reason: str):
        self.logger.error(f"failed to download state dict with ref {ref_id}: {reason}")
        self.error = reason
        self.result = None

    def download_completed(self, ref_id: str):
        self.logger.debug(f"received state dict with {self.num_tensors_received} tensors")


class TensorDownloader(ObjDownloader):

    @classmethod
    def new_transaction(
        cls,
        cell: Cell,
        num_tensors_per_chunk: int = 1,
        timeout: float = 5.0,
        timeout_cb=None,
        **cb_kwargs,
    ):
        """Create a new tensor download transaction.

        Args:
            cell: the cell for communication with recipients
            num_tensors_per_chunk: number of tensors to send for each chunk
            timeout: timeout for the transaction
            timeout_cb: CB to be called when the transaction is timed out
            **cb_kwargs: args to be passed to the CB

        Returns: transaction id

        The timeout_cb must follow this signature:

            cb(tx_id, state_dicts: List[dict[str, torch.Tensor], **cb_args)

        """
        return ObjDownloader.new_transaction(
            cell=cell,
            producer=_ChunkProducer(num_tensors_per_chunk=num_tensors_per_chunk),
            timeout=timeout,
            timeout_cb=cls._tx_timeout,
            app_timeout_cb=timeout_cb,
            **cb_kwargs,
        )

    @classmethod
    def _tx_timeout(cls, tx_id: str, objs: List[Any], app_timeout_cb, **cb_kwargs):
        if app_timeout_cb:
            sds = [obj.state_dict for obj in objs]
            app_timeout_cb(tx_id, sds, **cb_kwargs)

    @classmethod
    def add_state_dict(
        cls,
        transaction_id: str,
        state_dict: dict[str, torch.Tensor],
        ref_id=None,
        state_dict_downloaded_cb=None,
        **cb_kwargs,
    ) -> str:
        """Add a file to be downloaded to the specified transaction.

        Args:
            transaction_id: ID of the transaction
            state_dict: state dict to be downloaded
            ref_id: ref id to be used, if provided
            state_dict_downloaded_cb: CB to be called when the state dict is done downloading
            **cb_kwargs: args to be passed to the CB

        Returns: reference id for the file.

        The state_dict_downloaded_cb must follow this signature:

            cb(ref_id: str, to_site: str, status: str, state_dict: dict[str, torch.Tensor], **cb_kwargs)

        """
        obj = _SourceTensor(state_dict)
        return ObjDownloader.add_download_object(
            transaction_id=transaction_id,
            obj=obj,
            ref_id=ref_id,
            obj_downloaded_cb=cls._source_tensor_downloaded,
            app_downloaded_cb=state_dict_downloaded_cb,
            **cb_kwargs,
        )

    @classmethod
    def _source_tensor_downloaded(
        cls, ref_id: str, to_site: str, status: str, obj: _SourceTensor, app_downloaded_cb, **cb_kwargs
    ):
        if app_downloaded_cb:
            app_downloaded_cb(ref_id, to_site, status, obj.state_dict, **cb_kwargs)

    @classmethod
    def download_state_dict(
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
    ) -> (str, Optional[dict[str, torch.Tensor]]):
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
