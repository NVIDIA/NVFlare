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
import threading
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.streaming.obj_downloader import Consumer, ObjDownloader, ProduceRC, download_object
from nvflare.fuel.f3.streaming.smod import SelfManagedObject, SelfManagedObjectDownloader
from nvflare.fuel.utils.log_utils import get_obj_logger


class _StateKey:
    START = "start"
    COUNT = "count"


class CountableObject(ABC):

    @abstractmethod
    def get_item_count(self) -> int:
        pass

    @abstractmethod
    def produce_item(self, index: int) -> Any:
        pass


class ItemConsumer(ABC):

    @abstractmethod
    def consume_items(self, items: List[Any], result: Any) -> Any:
        """Process items and return updated result."""
        pass


class _CacheableObject(SelfManagedObject):

    def __init__(self, obj: CountableObject, num_items_per_chunk: int):
        super().__init__()
        self.obj = obj
        self.num_items_per_chunk = num_items_per_chunk
        self.size = obj.get_item_count()
        self.cache = [(None, 0)] * self.size
        self.lock = threading.Lock()
        self.num_receivers = 0
        self.logger = get_obj_logger(self)

    def set_transaction(self, tx_id, ref_id):
        tx_info = ObjDownloader.get_transaction_info(tx_id)
        self.num_receivers = tx_info.num_receivers
        self.logger.info(f"set transaction info: {tx_id=}, {ref_id=} {self.num_receivers=}")

    def downloaded_to_all(self):
        self.logger.info(f"object has been downloaded to all {self.num_receivers=} sites - clear cache")
        self.clear_cache()

    def clear_cache(self):
        with self.lock:
            self.cache = None

    def _get_item(self, index: int) -> bytes:
        with self.lock:
            data, _ = self.cache[index]
            if data is None:
                data = self.obj.produce_item(index)
                self.cache[index] = (data, 0)
            else:
                self.logger.info(f"got item {index} from cache")
            return data

    def _adjust_cache(self, start: int, count: int):
        with self.lock:
            for i in range(start, start + count):
                data, num_received = self.cache[i]
                num_received += 1
                if num_received >= self.num_receivers:
                    self.logger.info(f"item {i} was received by {num_received} sites - clear cache")
                    self.cache[i] = (None, num_received)
                else:
                    self.cache[i] = (data, num_received)

    def produce(self, state: dict, requester: str) -> Tuple[str, Any, dict]:
        if not state:
            # first request
            start = 0
        else:
            received_start = state.get(_StateKey.START, 0)
            received_count = state.get(_StateKey.COUNT, 0)
            if received_count > 0:
                self._adjust_cache(received_start, received_count)

            start = received_start + received_count

        end = min(start + self.num_items_per_chunk, self.size)
        count = end - start

        if count <= 0:
            # already done
            return ProduceRC.EOF, None, {}

        result = []
        for i in range(start, end):
            item = self._get_item(i)
            result.append(item)
        return ProduceRC.OK, result, {_StateKey.START: start, _StateKey.COUNT: count}


class _ConsumerWrapper(Consumer):

    def __init__(self, c: ItemConsumer):
        super().__init__()
        self.consumer = c
        self.error = None
        self.result = None

    def consume(self, ref_id: str, state: dict, data: Any) -> dict:
        assert isinstance(data, list)
        self.result = self.consumer.consume_items(data, self.result)
        return state

    def download_failed(self, ref_id, reason: str):
        self.logger.error(f"failed to download object with ref {ref_id}: {reason}")
        self.error = reason
        self.result = None

    def download_completed(self, ref_id: str):
        self.logger.debug(f"received object with ref {ref_id}")


class CachedObjDownloader:

    @classmethod
    def new_transaction(
        cls,
        cell: Cell,
        timeout: float,
        num_receivers: int,
        timeout_cb=None,
        **cb_kwargs,
    ):
        return SelfManagedObjectDownloader.new_transaction(
            cell,
            timeout=timeout,
            num_receivers=num_receivers,
            timeout_cb=cls._handle_timeout,
            cb_info=(timeout_cb, cb_kwargs),
        )

    @classmethod
    def _handle_timeout(cls, tx_id: str, objs: list, cb_info):
        app_timeout_cb, cb_kwargs = cb_info
        if app_timeout_cb:
            original_objs = [obj.obj for obj in objs]
            app_timeout_cb(tx_id, original_objs, **cb_kwargs)

        for obj in objs:
            assert isinstance(obj, _CacheableObject)
            obj.clear_cache()

    @classmethod
    def add_object(
        cls,
        transaction_id: str,
        obj: CountableObject,
        num_items_per_chunk: int = 1,
    ) -> str:
        """Add an object to be downloaded to the specified transaction.

        Args:
            transaction_id: ID of the transaction
            obj: object to be downloaded
            num_items_per_chunk: number of items per chunk

        Returns: reference id for the object.

        """
        obj = _CacheableObject(obj, num_items_per_chunk)
        return SelfManagedObjectDownloader.add_download_object(
            transaction_id=transaction_id,
            obj=obj,
        )

    @classmethod
    def download_object(
        cls,
        from_fqcn: str,
        ref_id: str,
        item_consumer: ItemConsumer,
        per_request_timeout: float,
        cell: Cell,
        secure=False,
        optional=False,
        abort_signal=None,
    ) -> Tuple[str, Any]:
        """Download the referenced object from the source.

        Args:
            from_fqcn: FQCN of the data source.
            ref_id: reference ID of the state dict to be downloaded.
            item_consumer: item consumer.
            per_request_timeout: timeout for requests sent to the data source.
            cell: cell to be used for communicating to the data source.
            secure: P2P private mode for communication
            optional: supress log messages of communication
            abort_signal: signal for aborting download.

        Returns: tuple of (error message if any, object).

        """
        consumer = _ConsumerWrapper(item_consumer)
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
