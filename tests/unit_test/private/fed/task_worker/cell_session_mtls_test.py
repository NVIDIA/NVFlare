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

"""Live mTLS: a worker joins its Client Job as the Client Job joins its CP, with the job's own certificate.

Each party runs in its own spawned process, as in production, because comm config and cells are process-wide.
"""

import json
import multiprocessing as mp
import os
import time
import traceback

import pytest

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.message import Message
from nvflare.fuel.utils.config_service import ConfigService
from tests.unit_test.fuel.f3.cellnet.job_cert_binding_live_test import _JOB_A, _JOB_B, _free_port, _write_pki

pytestmark = pytest.mark.timeout(180)

_CJ_FQCN = f"site-1.{_JOB_B}"
_ADMIN = ("admin", "abort")


def _run_cj(pki, config_dir, ready_q, stop_ev, record_q):
    """The Client Job: job B's cell with job B's certificate, the site's internal mTLS, and an admin route."""
    from nvflare.private.fed.task_worker import cell_session

    cell = None
    try:
        ConfigService.initialize(section_files={}, config_path=[config_dir])
        credentials = {
            DriverParams.CA_CERT.value: pki["rootCA.pem"],
            DriverParams.CLIENT_CERT.value: pki["job_b.crt"],
            DriverParams.CLIENT_KEY.value: pki["job_b.key"],
            DriverParams.SERVER_CERT.value: pki["job_b.crt"],
            DriverParams.SERVER_KEY.value: pki["job_b.key"],
            DriverParams.CONNECTION_SECURITY.value: "mtls",
        }
        cell = Cell(
            _CJ_FQCN,
            # unused root URL: the job cell's own uplink is not under test, and its connect attempts are refused
            f"stcp://localhost:{_free_port()}",
            secure=True,
            credentials=credentials,
            create_internal_listener=False,
            auth_identity="site-1",
            # as FederatedClientBase builds it for a CJ: cells under the job cell authenticate as the job
            auth_identity_map={"site-1": "site-1", _CJ_FQCN: "site-1"},
        )
        cell.register_request_cb(*_ADMIN, lambda request: record_q.put("admin") or Message())
        cell.start()
        cell_session.listen(cell)
        session = cell_session.CellWorkerSession(cell, lambda data: record_q.put(dict(data)))
        ready_q.put(session.bootstrap_fields())
        stop_ev.wait(150)
    except Exception:
        ready_q.put(traceback.format_exc())
    finally:
        if cell:
            cell.stop()


def _run_worker(fields, result_q):
    from nvflare.apis.shareable import Shareable
    from nvflare.private.fed.task_worker.cell_session import CellWorkerTransport

    transport = CellWorkerTransport(fields)
    try:
        transport.open()
        transport.log(Shareable({"loss": 0.5}))
        result_q.put("ok")
    except Exception:
        result_q.put(traceback.format_exc())
    finally:
        transport.close()


def _run_intruder(connect_url, credentials, fqcn, result_q):
    """Anything else that finds the port: it tries the Client Job's admin route."""
    cell = None
    try:
        cell = Cell(
            fqcn,
            "",
            secure=bool(credentials),
            credentials=credentials,
            create_internal_listener=False,
            parent_url=connect_url,
            parent_resources={DriverParams.CONNECTION_SECURITY.value: "mtls"},
            auth_identity_map={_CJ_FQCN: "site-1"},
        )
        cell.start()
        deadline = time.time() + 5
        while time.time() < deadline and not cell.is_cell_connected(_CJ_FQCN):
            time.sleep(0.1)
        reply = cell.send_request(_ADMIN[0], _ADMIN[1], _CJ_FQCN, Message(), timeout=3)
        result_q.put(reply.get_header(MessageHeaderKey.RETURN_CODE))
    except Exception:
        result_q.put(traceback.format_exc())
    finally:
        if cell:
            cell.stop()


def _spawn(target, *args):
    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    proc = ctx.Process(target=target, args=(*args, result_q))
    proc.start()
    try:
        return result_q.get(timeout=60)
    finally:
        proc.join(15)


@pytest.fixture(scope="module")
def client_job(tmp_path_factory):
    ctx = mp.get_context("spawn")
    pki = _write_pki(str(tmp_path_factory.mktemp("pki")))
    config_dir = str(tmp_path_factory.mktemp("config"))
    with open(os.path.join(config_dir, "comm_config.json"), "w") as f:
        json.dump(
            {"internal": {"scheme": "stcp", "resources": {"host": "localhost", "connection_security": "mtls"}}}, f
        )
    stop_ev = ctx.Event()
    ready_q, record_q = ctx.Queue(), ctx.Queue()
    proc = ctx.Process(target=_run_cj, args=(pki, config_dir, ready_q, stop_ev, record_q))
    proc.start()
    try:
        fields = ready_q.get(timeout=60)
        assert isinstance(fields, dict), fields
        yield fields, pki, record_q
    finally:
        stop_ev.set()
        proc.join(15)


def _drain(record_q, wait=2.0):
    records, deadline = [], time.time() + wait
    while time.time() < deadline:
        try:
            records.append(record_q.get(timeout=0.2))
        except Exception:
            pass
    return records


def test_the_listener_follows_the_site_internal_mtls_and_hands_workers_the_job_credential(client_job):
    fields, pki, _ = client_job
    assert fields["connect_url"].startswith("stcp://")
    assert fields["connection_security"] == "mtls"
    assert fields["identity"] == "site-1"
    assert fields["credentials"] == {
        DriverParams.CA_CERT.value: pki["rootCA.pem"],
        DriverParams.CLIENT_CERT.value: pki["job_b.crt"],
        DriverParams.CLIENT_KEY.value: pki["job_b.key"],
    }


def test_a_worker_with_the_job_credential_delivers_metrics(client_job):
    fields, _, record_q = client_job
    assert _spawn(_run_worker, fields) == "ok"
    assert [record.get("loss") for record in _drain(record_q)] == [0.5]


@pytest.mark.parametrize(
    "credential, reaches_admin",
    [(None, False), ("job_a", False), ("job_b", True)],
    ids=["no_certificate", "another_jobs_certificate", "this_jobs_certificate"],
)
def test_only_this_jobs_credential_reaches_the_client_job(client_job, credential, reaches_admin):
    """The job credential is the boundary, as for the CP: a holder is inside the job, anyone else is out."""
    fields, pki, record_q = client_job
    credentials = {}
    if credential:
        credentials = {
            DriverParams.CA_CERT.value: pki["rootCA.pem"],
            DriverParams.CLIENT_CERT.value: pki[f"{credential}.crt"],
            DriverParams.CLIENT_KEY.value: pki[f"{credential}.key"],
            DriverParams.CONNECTION_SECURITY.value: "mtls",
        }
    rc = _spawn(_run_intruder, fields["connect_url"], credentials, f"{_CJ_FQCN}.intruder_{_JOB_A[:8]}")
    assert (rc == ReturnCode.OK) is reaches_admin
    assert ("admin" in _drain(record_q)) is reaches_admin
