# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Shared DownloadService test helpers.

Used by download_service_test.py and transfer_outcome_test.py so the isolated
service subclass stays in one place: every class-level table DownloadService
grows must be overridden here, or an "isolated" test subclass silently shares
production state.
"""

import threading
import weakref
from typing import Any, Tuple
from unittest.mock import Mock, patch

import pytest

from nvflare.fuel.f3.streaming.download_service import Downloadable, DownloadService, ProduceRC


class MockDownloadable(Downloadable):
    """Mock downloadable for testing."""

    def __init__(self, data_chunks: list, fail_on_chunk: int = -1):
        super().__init__(data_chunks)
        self.data_chunks = data_chunks
        self.fail_on_chunk = fail_on_chunk
        self.current_chunk = 0
        self.downloaded_to_one_calls = []
        self.downloaded_to_all_called = False
        self.downloaded_to_all_call_count = 0
        self.transaction_done_calls = []
        self.released = False
        self.tx_id = None
        self.ref_id = None

    def set_transaction(self, tx_id: str, ref_id: str):
        self.tx_id = tx_id
        self.ref_id = ref_id

    def produce(self, state: dict, requester: str) -> Tuple[str, Any, dict]:
        if not state:
            chunk_idx = 0
        else:
            chunk_idx = state.get("chunk_idx", 0)

        if self.fail_on_chunk >= 0 and chunk_idx == self.fail_on_chunk:
            return ProduceRC.ERROR, None, {}

        if chunk_idx >= len(self.data_chunks):
            return ProduceRC.EOF, None, {}

        return ProduceRC.OK, self.data_chunks[chunk_idx], {"chunk_idx": chunk_idx + 1}

    def downloaded_to_one(self, to_receiver: str, status: str):
        self.downloaded_to_one_calls.append((to_receiver, status))

    def downloaded_to_all(self):
        self.downloaded_to_all_called = True
        self.downloaded_to_all_call_count += 1

    def transaction_done(self, transaction_id: str, status: str):
        self.transaction_done_calls.append((transaction_id, status))

    def release(self):
        self.released = True


def make_isolated_download_service():
    class IsolatedDownloadService(DownloadService):
        _tx_table = {}
        _ref_table = {}
        _finished_refs = {}
        _tx_outcomes = {}
        _tx_incarnations = {}
        _outcome_lock = threading.Lock()
        _logger = Mock()
        _tx_lock = threading.Lock()
        _initialized_cells = weakref.WeakKeyDictionary()

    return IsolatedDownloadService


def run_monitor_once(service_cls, now):
    from nvflare.fuel.f3.streaming import download_service as download_service_module

    class MonitorIterationDone(Exception):
        pass

    monitor_thread = threading.current_thread()
    real_time = download_service_module.time.time
    real_sleep = download_service_module.time.sleep

    def test_thread_time():
        if threading.current_thread() is monitor_thread:
            return now
        return real_time()

    def test_thread_sleep(seconds):
        if threading.current_thread() is monitor_thread:
            raise MonitorIterationDone
        real_sleep(seconds)

    with (
        patch.object(download_service_module.time, "time", side_effect=test_thread_time),
        patch.object(download_service_module.time, "sleep", side_effect=test_thread_sleep),
    ):
        with pytest.raises(MonitorIterationDone):
            service_cls._monitor_tx()
