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


import threading
from contextlib import contextmanager

import pytest

from nvflare.fuel.f3.streaming.download_service import Downloadable
from nvflare.fuel.utils.fobs.decomposers.via_downloader import _GuardedDownloadable


class Source(Downloadable):
    def __init__(self):
        super().__init__(object())
        self.started = threading.Event()
        self.finish = threading.Event()
        self.releases = 0

    def produce(self, state, requester):
        self.started.set()
        assert self.finish.wait(5)
        assert self.base_obj is not None
        return "OK", "payload", {}

    def release(self):
        self.releases += 1
        self.base_obj = None


def test_release_after_transaction_timeout_waits_for_actual_reader():
    source = Source()
    leases = []

    @contextmanager
    def guard():
        token = object()
        leases.append(token)
        try:
            yield
        finally:
            leases.remove(token)

    wrapped = _GuardedDownloadable(source, guard)
    result = []
    thread = threading.Thread(target=lambda: result.append(wrapped.produce({}, "server")))
    thread.start()
    try:
        assert source.started.wait(5)
        wrapped.release()
        wrapped.release()
        assert source.releases == 0
        assert len(leases) == 2
        source.finish.set()
        thread.join(5)
        assert not thread.is_alive()
        assert result == [("OK", "payload", {})]
        assert source.releases == 1
        assert not leases
        with pytest.raises(RuntimeError, match="released"):
            wrapped.produce({}, "server")
    finally:
        source.finish.set()
        thread.join(5)


def test_source_release_failure_still_closes_lease():
    leases = []

    @contextmanager
    def guard():
        leases.append(True)
        try:
            yield
        finally:
            leases.pop()

    source = Source()

    def fail():
        raise RuntimeError("release failed")

    source.release = fail
    wrapped = _GuardedDownloadable(source, guard)
    with pytest.raises(RuntimeError, match="release failed"):
        wrapped.release()
    wrapped.release()
    assert not leases
