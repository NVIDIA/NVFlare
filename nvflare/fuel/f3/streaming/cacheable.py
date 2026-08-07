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
from concurrent.futures import Future
from typing import Any, List, Optional, Tuple

from nvflare.fuel.f3.streaming.download_service import Consumer, Downloadable, DownloadService, ProduceRC
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.fuel.utils.validation_utils import check_non_negative_int


class _StateKey:
    START = "start"
    COUNT = "count"


class CacheableObject(Downloadable):
    """This class provides cache capability for managing chunks generated during streaming.
    When the object is to be sent to multiple receivers, each chunk is generated only once and cached for other
    receivers. Once all receivers received the chunk, it's removed from the cache.

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
        # Multiple receivers can request the same uncached item concurrently.
        # Keep one shared future per item so only one of them performs the
        # potentially expensive production work (for example safetensors.save).
        self._production_futures: dict[int, Future] = {}
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

    def prefetch_item(self, index: int):
        """Optionally start producing an item before the receiver requests it.

        The default implementation does nothing. Subclasses with expensive
        ``produce_item`` implementations can override this to keep one bounded
        item of work ahead of the network transfer.
        """
        pass

    def get_item_size(self, index: int) -> Optional[int]:
        """Return a lower-bound item size when it is known without production."""
        return None

    def set_transaction(self, tx_id, ref_id):
        tx_info = DownloadService.get_transaction_info(tx_id)
        self.num_receivers = tx_info.num_receivers
        self.logger.info(f"set transaction info: {tx_id=}, {ref_id=} {self.num_receivers=}")

    def downloaded_to_all(self):
        self.logger.info(f"object has been downloaded to all {self.num_receivers} receivers - clear cache")
        self.clear_cache()

    def transaction_done(self, transaction_id: str, status: str):
        self.clear_cache()

    def clear_cache(self):
        """Clear the chunk cache only.

        Does NOT touch base_obj — the source object is released separately
        via release() after the transaction_done_cb has been invoked, so the
        callback can still observe the original data if needed.
        """
        with self.lock:
            self.cache = None

    def release(self):
        """Drop the reference to the source object.

        Called by _Transaction.transaction_done() AFTER the transaction_done_cb
        fires.  Setting base_obj to None drops the last infrastructure reference
        to the source data (e.g. a 5 GiB numpy dict), allowing it to be
        reclaimed by the GC immediately rather than waiting for a future cycle.

        Overrides Downloadable.release() (which is a no-op by default).
        """
        with self.lock:
            self.base_obj = None

    def _get_item(self, index: int, requester: str) -> bytes:
        with self.lock:
            cache_available = bool(self.cache)
            data = None if not cache_available else self.cache[index][0]
            base_obj = self.base_obj  # snapshot under lock for thread-safety
            if data is not None:
                self.logger.debug(f"got item {index} from cache for {requester}")
                return data

            if base_obj is None:
                # release() was already called — no new chunk requests should
                # arrive after transaction_done(), but guard defensively.
                raise RuntimeError(f"item {index} requested after base_obj released for {requester}")

            production_future = self._production_futures.get(index)
            if production_future is None:
                production_future = Future()
                self._production_futures[index] = production_future
                produce = True
            else:
                produce = False

        if not produce:
            # Wait outside self.lock. The producer stores the result in the cache
            # before resolving this future, so later requests use the cache too.
            self.logger.debug(f"waiting for in-flight item {index} for {requester}")
            return production_future.result()

        try:
            # produce_item() reads self.base_obj internally and is called outside
            # the lock. A concurrent release() cannot normally occur here because
            # release() runs only after the download service confirms completion.
            data = self.produce_item(index)

            # Publish the result before detaching the flight. A new requester
            # therefore sees either the cached bytes or the shared future, never
            # a gap in between that could trigger duplicate production.
            with self.lock:
                if self.cache:
                    existing, count = self.cache[index]
                    if existing is None:
                        self.cache[index] = (data, count)
                        self.logger.debug(f"created and cached item {index} for {requester}: {len(data)} bytes")
                    else:
                        data = existing
                        self.logger.debug(f"got item {index} from cache for {requester}")
        except BaseException as ex:
            # Detach before waking waiters. A later request can retry while every
            # receiver already holding this future observes the same failure.
            with self.lock:
                if self._production_futures.get(index) is production_future:
                    self._production_futures.pop(index)
            production_future.set_exception(ex)
            raise

        # Keep an uncached flight discoverable until it is settled. This matters
        # when clear_cache() races production: there are no cached bytes to bridge
        # a detach-before-settle gap for a new requester.
        try:
            production_future.set_result(data)
        finally:
            with self.lock:
                if self._production_futures.get(index) is production_future:
                    self._production_futures.pop(index)
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
                    self.logger.debug(f"item {i} was received by {num_received} receivers - clear cache")
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
        should_prefetch = True

        for i in range(start, self.size):
            if result:
                estimated_size = self.get_item_size(i)
                if estimated_size is not None and total_size + estimated_size >= self.max_chunk_size:
                    break
            item = self._get_item(i, requester)
            item_size = len(item)
            if not result or total_size + item_size < self.max_chunk_size:
                result.append(item)
                total_size += item_size
            else:
                # _get_item() already produced and cached this item.
                should_prefetch = False
                break

        next_index = start + len(result)
        if should_prefetch:
            # Keep up to two items ahead without allowing two oversized
            # serializations to multiply peak RSS. The first item is always
            # eligible because an individual item may legitimately exceed
            # max_chunk_size; additional items must fit in the byte budget.
            prefetched_size = 0
            for _pi in range(next_index, min(next_index + 2, self.size)):
                estimated_size = self.get_item_size(_pi)
                if prefetched_size and (
                    estimated_size is None or prefetched_size + estimated_size > self.max_chunk_size
                ):
                    break
                self.prefetch_item(_pi)
                if estimated_size is None:
                    break
                prefetched_size += estimated_size

        self.logger.debug(f"produced {len(result)} items for {requester}: {total_size} bytes")
        return ProduceRC.OK, result, {_StateKey.START: start, _StateKey.COUNT: len(result)}


class ItemConsumer(Consumer):

    supports_pipelining = True

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
