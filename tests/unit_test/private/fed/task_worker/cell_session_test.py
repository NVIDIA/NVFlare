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

"""Protocol edge cases of the cell session, over fake cells; the full flow runs in task_execution_test."""

from unittest.mock import Mock

import pytest

from nvflare.apis.shareable import Shareable
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply
from nvflare.private.fed.task_worker import cell_session
from nvflare.private.fed.task_worker.cell_session import CellWorkerSession, CellWorkerTransport

pytestmark = pytest.mark.timeout(60)

CJ_FQCN = "site-1.job-1"


class FakeCell:
    """Delivers requests straight to the peer's registered callback, stamping the origin header."""

    def __init__(self, network, fqcn):
        self.network = network
        self.fqcn = fqcn
        self.callbacks = {}
        self.listener = None
        self.core_cell = self

    def get_fqcn(self):
        return self.fqcn

    def make_internal_listener(self, scheme=None, resources=None):
        self.listener = f"{scheme}://127.0.0.1:1"

    def get_internal_listener_url(self):
        return self.listener

    def is_cell_connected(self, target):
        return target in self.network

    def register_request_cb(self, channel, topic, cb, *args):
        self.callbacks[(channel, topic)] = (cb, args)

    def start(self):
        pass

    def stop(self):
        pass

    def send_request(self, channel, topic, target, request, timeout=None, **kwargs):
        peer = self.network.get(target)
        callback = peer.callbacks.get((channel, topic)) if peer is not None else None
        if callback is None:
            return make_reply(CellReturnCode.TARGET_UNREACHABLE)
        request.set_header(MessageHeaderKey.ORIGIN, self.fqcn)
        cb, args = callback
        return cb(request, *args)


@pytest.fixture
def network(monkeypatch):
    cells = {}
    cells[CJ_FQCN] = FakeCell(cells, CJ_FQCN)
    monkeypatch.setattr(cell_session, "Cell", lambda fqcn, **_kwargs: cells.setdefault(fqcn, FakeCell(cells, fqcn)))
    monkeypatch.setattr(cell_session, "_sessions", {})
    cell_session.listen(cells[CJ_FQCN])
    return cells


def _open_pair(network, on_analytics=None, **overrides):
    cj = CellWorkerSession(network[CJ_FQCN], on_analytics or Mock())
    return cj, CellWorkerTransport({**cj.bootstrap_fields(), **overrides})


def test_hello_requires_the_launch_token_of_the_prescribed_identity(network):
    _, transport = _open_pair(network, token="forged")
    with pytest.raises(RuntimeError, match="launch token mismatch"):
        transport.open()

    _, transport = _open_pair(network, worker_fqcn=CJ_FQCN + ".task_worker_99")
    with pytest.raises(RuntimeError, match="unknown task worker"):
        transport.open()

    cj, transport = _open_pair(network)
    transport.open()
    assert transport.cj_fqcn == CJ_FQCN
    assert cj.bootstrap_fields()["connect_url"] == "tcp://127.0.0.1:1"


def test_two_sessions_on_one_cell_route_records_by_worker(network):
    records_1, records_2 = [], []
    cj_1, transport_1 = _open_pair(network, records_1.append)
    cj_2, transport_2 = _open_pair(network, records_2.append)
    transport_1.open()
    transport_2.open()
    first, second = Shareable({"metric": 1}), Shareable({"metric": 2})
    transport_1.log(first)
    transport_2.log(second)
    assert records_1 == [first] and records_2 == [second]
    cj_1.close()
    with pytest.raises(RuntimeError, match="unknown task worker"):
        transport_1.log(first)
    transport_2.log(first)
    assert records_2 == [second, first]


def test_rejected_or_unproven_records_fail_the_logging_worker(network):
    cj, transport = _open_pair(network, Mock(side_effect=ValueError("unsupported task worker analytics event")))
    transport.open()
    with pytest.raises(RuntimeError, match="ValueError: unsupported task worker analytics event"):
        transport.log(Shareable())
    transport.config["token"] = "forged"
    with pytest.raises(RuntimeError, match="launch token mismatch"):
        transport.log(Shareable())
    cj.close()
    assert cell_session._sessions == {}
