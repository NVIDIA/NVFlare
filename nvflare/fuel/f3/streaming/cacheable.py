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
from abc import abstractmethod
from typing import Any, List, Optional, Tuple

from nvflare.fuel.f3.streaming.download_service import Consumer, Downloadable, DownloadService, ProduceRC
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.fuel.utils.validation_utils import check_non_negative_int


class _StateKey:
    START = "start"
    COUNT = "count"


class CacheableObject(Downloadable):
    """This class provides cache capability for managing chunks generated during streaming.
    When the object is to be sent to multiple sites, each chunk is generated only once and cached for other
    sites. Once all sites received the chunk, it's removed from the cache.

    """

    def __init__(self, obj: Any, max_chunk_size: int):
        """Constructor of CacheableObject.

        Args:
            obj: the object to be downloaded.
            max_chunk_size: max number of bytes for each chunk.

        Notes: The object must be able to be divided into multiple items. A chunk is generated for each item.
        """
        super().__init__(obj)
        check_non_negative_int("max_chunk_size", max_chunk_size)
        self.max_chunk_size = max_chunk_size
        self.size = self.get_item_count()
        self.cache: list[tuple[Optional[bytes], int]] = [(None, 0)] * self.size
        self.lock = threading.Lock()
        self.num_receivers = 0
        self.logger = get_obj_logger(self)

    @abstractmethod
    def get_item_count(self) -> int:
        """The subclass must implement this method to return the number of items the object contains.

        Returns: the number of items the object contains

        """
        pass

    @abstractmethod
    def produce_item(self, index: int) -> bytes:
        """This method is called to produce the chunk for the specified item.

        Args:
            index: index of the item.

        Returns: a chunk for the item

        """
        pass

    def set_transaction(self, tx_id, ref_id):
        tx_info = DownloadService.get_transaction_info(tx_id)
        self.num_receivers = tx_info.num_receivers
        self.logger.info(f"set transaction info: {tx_id=}, {ref_id=} {self.num_receivers=}")

    def downloaded_to_all(self):
        self.logger.info(f"object has been downloaded to all {self.num_receivers} sites - clear cache")
        self.clear_cache()

    def transaction_done(self, transaction_id: str, status: str):
        self.clear_cache()

    def clear_cache(self):
        with self.lock:
            self.cache = None

    def _get_item(self, index: int, requester: str) -> bytes:
        with self.lock:
            if not self.cache:
                # the cache has been cleared
                data = None
            else:
                data, _ = self.cache[index]

            if data is None:
                data = self.produce_item(index)
                if self.cache:
                    self.cache[index] = (data, 0)
                    self.logger.debug(f"created and cached item {index} for {requester}: {len(data)} bytes")
            else:
                self.logger.debug(f"got item {index} from cache for {requester}")
            return data

    def _adjust_cache(self, start: int, count: int):
        with self.lock:
            if not self.cache:
                # cache has been cleared
                return

            for i in range(start, start + count):
                data, num_received = self.cache[i]
                num_received += 1
                if num_received >= self.num_receivers:
                    self.logger.debug(f"item {i} was received by {num_received} sites - clear cache")
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

        if start >= self.size:
            # already done
            return ProduceRC.EOF, None, {}

        result = []
        total_size = 0

        for i in range(start, self.size):
            item = self._get_item(i, requester)
            item_size = len(item)
            if not result or total_size + item_size < self.max_chunk_size:
                result.append(item)
                total_size += item_size
            else:
                break

        self.logger.debug(f"produced {len(result)} items for {requester}: {total_size} bytes")
        return ProduceRC.OK, result, {_StateKey.START: start, _StateKey.COUNT: len(result)}


class ItemConsumer(Consumer):

    def __init__(self):
        super().__init__()
        self.error = None
        self.result = None

    @abstractmethod
    def consume_items(self, items: List[Any], result: Any) -> Any:
        """Process items and return updated result."""
        pass

    def consume(self, ref_id: str, state: dict, data: Any) -> dict:
        assert isinstance(data, list)
        self.result = self.consume_items(data, self.result)
        return state

    def download_failed(self, ref_id, reason: str):
        self.logger.error(f"failed to download object with ref {ref_id}: {reason}")
        self.error = reason
        self.result = None

    def download_completed(self, ref_id: str):
        self.logger.debug(f"received object with ref {ref_id}")
