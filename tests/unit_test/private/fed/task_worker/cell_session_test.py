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

"""The cell session over real clear loopback cells; mTLS is in cell_session_mtls_test, the full flow in task_execution_test."""

import threading
import time
import uuid
from unittest.mock import Mock

import pytest

from nvflare.apis.shareable import Shareable
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.fuel.f3.stats_pool import StatsPoolManager
from nvflare.fuel.utils import fobs
from nvflare.private.fed.task_worker import cell_session
from nvflare.private.fed.task_worker.cell_session import CellWorkerSession, CellWorkerTransport

pytestmark = pytest.mark.timeout(60)


# The site's internal settings name the CP's own listener, as a provisioned kit does.
_CP_PORT = 8102
_SITE_INTERNAL = {"host": "cp-1.invalid", "port": _CP_PORT, "connection_security": "clear"}


def _client_job_cell():
    """A Client Job cell that connects nowhere, with the CP's address in its site internal settings."""
    cell = Cell(f"site-1.job-{uuid.uuid4().hex}", "tcp://127.0.0.1:0", secure=False, credentials={})
    cell.core_cell.connector_manager.int_resources = dict(_SITE_INTERNAL)
    cell.start()
    return cell


@pytest.fixture
def cells():
    original_cells, original_pools = dict(CoreCell.ALL_CELLS), dict(StatsPoolManager.pools)
    created = []
    try:
        yield created
    finally:
        for item in reversed(created):
            item.stop() if isinstance(item, Cell) else item.close()
        CoreCell.ALL_CELLS.clear(), CoreCell.ALL_CELLS.update(original_cells)
        StatsPoolManager.pools.clear(), StatsPoolManager.pools.update(original_pools)


@pytest.fixture
def cj(cells):
    """A Client Job cell whose workers join it through its internal listener."""
    cell = _client_job_cell()
    cells.append(cell)
    cell_session.listen(cell, "localhost")
    return cell, cells


def test_the_listener_is_at_the_launcher_host_on_its_own_port_not_the_cps_address(cj):
    url = cj[0].get_internal_listener_url()
    assert url.startswith("tcp://localhost:") and not url.endswith(f":{_CP_PORT}")
    assert cj[0].core_cell.connector_manager.int_resources == _SITE_INTERNAL, "the site settings are left alone"


def test_concurrent_listen_calls_open_one_listener(cells):
    cell = _client_job_cell()
    cells.append(cell)
    calls = Mock(wraps=cell.core_cell.connector_manager.get_internal_listener)
    cell.core_cell.connector_manager.get_internal_listener = calls
    listeners = [threading.Thread(target=cell_session.listen, args=(cell, "localhost")) for _ in range(8)]
    for listener in listeners:
        listener.start()
    for listener in listeners:
        listener.join(10)
    calls.assert_called_once()
    assert cell.get_internal_listener_url().startswith("tcp://localhost:")


def _worker(cj, on_analytics=None, **overrides):
    cell, transports = cj
    session = CellWorkerSession(cell, on_analytics or Mock())
    transport = CellWorkerTransport({**session.bootstrap_fields(), **overrides})
    transports.append(transport)
    return session, transport


def _log_request(payload=None):
    return new_cell_message({}, fobs.dumps(Shareable()) if payload is None else payload)


def test_a_clear_site_gives_workers_the_cj_listener_and_no_credentials(cj):
    session, _ = _worker(cj)
    fields = session.bootstrap_fields()
    assert fields["connect_url"] == cj[0].get_internal_listener_url()
    assert "credentials" not in fields
    assert fields["worker_fqcn"].startswith(cj[0].get_fqcn() + ".task_worker_")


def test_hello_needs_the_session_of_the_prescribed_worker_name(cj):
    _, transport = _worker(cj, worker_fqcn=cj[0].get_fqcn() + ".task_worker_99")
    with pytest.raises(RuntimeError, match="unknown task worker"):
        transport.open()

    _, transport = _worker(cj)
    transport.open()
    assert transport.cj_fqcn == cj[0].get_fqcn()


def test_two_sessions_route_records_by_worker_and_report_decode_errors(cj):
    records_1, records_2 = [], []
    session_1, transport_1 = _worker(cj, records_1.append)
    _, transport_2 = _worker(cj, records_2.append)
    transport_1.open()
    transport_2.open()
    first, second = Shareable({"metric": 1}), Shareable({"metric": 2})
    transport_1.log(first)
    transport_2.log(second)
    assert records_1 == [first] and records_2 == [second]

    reply = session_1.handle(cell_session.Topic.LOG, _log_request(b"not fobs"))
    assert "Error" in reply.get_header(MessageHeaderKey.ERROR), "a payload the CJ cannot decode is rejected with why"

    session_1.close(timeout=0)
    with pytest.raises(RuntimeError, match="unknown task worker"):
        transport_1.log(first)
    transport_2.log(first)
    assert records_2 == [second, first]


def test_a_rejected_record_fails_the_logging_worker(cj):
    _, transport = _worker(cj, Mock(side_effect=ValueError("unsupported task worker analytics")))
    transport.open()
    with pytest.raises(RuntimeError, match="ValueError: unsupported task worker analytics"):
        transport.log(Shareable())


def test_close_rejects_queued_records_and_waits_for_a_stalled_receiver_only_up_to_its_timeout(cj):
    fired = []
    receiving, release = threading.Event(), threading.Event()

    def receiver(data):
        fired.append(data)
        receiving.set()
        release.wait(5)

    session, _ = _worker(cj, receiver)
    stalled = threading.Thread(target=session.handle, args=(cell_session.Topic.LOG, _log_request()), daemon=True)
    stalled.start()
    assert receiving.wait(5)
    started = time.monotonic()
    session.close(timeout=0.2)
    assert time.monotonic() - started < 2, "close() is bounded even while a receiver is stalled"
    reply = session.handle(cell_session.Topic.LOG, _log_request())
    release.set()
    stalled.join(5)
    assert reply.get_header(MessageHeaderKey.ERROR) == "session is closed"
    assert len(fired) == 1, "only the record that started before close() reached the receiver"
