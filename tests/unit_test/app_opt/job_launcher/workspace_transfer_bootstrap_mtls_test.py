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

"""Live mTLS regression: the workspace-transfer bootstrap cell authenticates to its parent with a job-bound cert."""

import logging
import multiprocessing as mp
import os
import socket
import time
import traceback
import uuid

import pytest
from cryptography import x509

from nvflare.app_opt.job_launcher.workspace_cell_transfer import make_workspace_transfer_fqcn
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.message import Message
from nvflare.lighter.constants import CertExtensionOID
from nvflare.lighter.utils import Identity, generate_cert, generate_keys, serialize_cert, serialize_pri_key
from nvflare.private.fed.utils.job_cert_utils import JobCertIssuer

_CHANNEL = "ws_bootstrap_test"
_TOPIC = "echo"
_CONNECT_TIMEOUT = 10.0
_REJECT_WAIT = 3.0
_REQUEST_TIMEOUT = 3.0
_REJECTION_LOG_WAIT = 10.0
_JOB_ID = str(uuid.uuid4())
_OTHER_JOB_ID = str(uuid.uuid4())


class _RejectionRecorder(logging.Handler):
    """Forwards the parent's job-binding rejections to the test process."""

    def __init__(self, queue):
        super().__init__(level=logging.ERROR)
        self.queue = queue

    def emit(self, record):
        message = record.getMessage()
        if "bound to job" in message:
            self.queue.put(message)


def _write_pki(out_dir: str) -> dict:
    root_key, root_pub = generate_keys()
    root_cert = generate_cert(Identity("rootCA"), Identity("rootCA"), root_key, root_pub, ca=True)
    srv_key, srv_pub = generate_keys()
    srv_cert = generate_cert(Identity("server"), Identity("rootCA"), root_key, srv_pub, server_default_host="localhost")
    jca_key, jca_pub = generate_keys()
    marker = x509.UnrecognizedExtension(x509.ObjectIdentifier(CertExtensionOID.JOB_CA_MARKER), b"job_ca")
    jca_cert = generate_cert(
        Identity("job_ca"),
        Identity("rootCA"),
        root_key,
        jca_pub,
        ca=True,
        ca_path_length=0,
        extra_extensions=[(marker, False)],
    )
    issuer = JobCertIssuer(serialize_cert(jca_cert), jca_key)
    job_crt, job_key = issuer.issue("server", _JOB_ID)
    other_crt, other_key = issuer.issue("server", _OTHER_JOB_ID)
    files = {
        "rootCA.pem": serialize_cert(root_cert),
        "server.crt": serialize_cert(srv_cert),
        "server.key": serialize_pri_key(srv_key),
        "job.crt": job_crt,
        "job.key": job_key,
        "other_job.crt": other_crt,
        "other_job.key": other_key,
    }
    paths = {}
    for name, data in files.items():
        paths[name] = os.path.join(out_dir, name)
        with open(paths[name], "wb") as f:
            f.write(data)
    return paths


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_parent(root_url, pki, ready_q, stop_ev, reject_q):
    cell = None
    try:
        logging.getLogger().addHandler(_RejectionRecorder(reject_q))
        credentials = {
            DriverParams.CA_CERT.value: pki["rootCA.pem"],
            DriverParams.SERVER_CERT.value: pki["server.crt"],
            DriverParams.SERVER_KEY.value: pki["server.key"],
            DriverParams.CONNECTION_SECURITY.value: "mtls",
        }
        cell = Cell("server", root_url, secure=True, credentials=credentials, create_internal_listener=False)
        cell.register_request_cb(_CHANNEL, _TOPIC, lambda request: Message(payload=request.payload))
        cell.start()
        ready_q.put("ready")
        stop_ev.wait(120)
    except Exception:
        ready_q.put(traceback.format_exc())
    finally:
        if cell:
            cell.stop()


def _run_bootstrap(root_url, pki, cert_name, fqcn, wait, result_q):
    cell = None
    try:
        credentials = {
            DriverParams.CA_CERT.value: pki["rootCA.pem"],
            DriverParams.SERVER_CERT.value: pki[f"{cert_name}.crt"],
            DriverParams.SERVER_KEY.value: pki[f"{cert_name}.key"],
            DriverParams.CONNECTION_SECURITY.value: "mtls",
        }
        cell = Cell(
            fqcn,
            root_url,
            secure=True,
            credentials=credentials,
            create_internal_listener=False,
            parent_url=root_url,
            parent_resources={DriverParams.CONNECTION_SECURITY.value: "mtls"},
            auth_identity_map={"server": "server"},
        )
        cell.start()
        deadline = time.time() + wait
        while time.time() < deadline and not cell.is_cell_connected("server"):
            time.sleep(0.1)
        # the client-side connected flag is transient while the parent is still validating the handshake;
        # only a completed application request proves the parent accepted this cell
        reply = cell.send_request(_CHANNEL, _TOPIC, "server", Message(payload="hello"), timeout=_REQUEST_TIMEOUT)
        result_q.put({"rc": reply.get_header(MessageHeaderKey.RETURN_CODE)})
    except Exception:
        result_q.put({"error": traceback.format_exc()})
    finally:
        if cell:
            cell.stop()


@pytest.fixture(scope="module")
def parent(tmp_path_factory):
    ctx = mp.get_context("spawn")
    pki = _write_pki(str(tmp_path_factory.mktemp("pki")))
    root_url = f"stcp://localhost:{_free_port()}"
    ready_q, reject_q, stop_ev = ctx.Queue(), ctx.Queue(), ctx.Event()
    proc = ctx.Process(target=_run_parent, args=(root_url, pki, ready_q, stop_ev, reject_q))
    proc.start()
    try:
        status = ready_q.get(timeout=30)
        assert status == "ready", status
        yield root_url, pki, reject_q
    finally:
        stop_ev.set()
        proc.join(15)


def _bootstrap(parent, cert_name, wait):
    root_url, pki, _ = parent
    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    fqcn = make_workspace_transfer_fqcn("server", _JOB_ID)
    proc = ctx.Process(target=_run_bootstrap, args=(root_url, pki, cert_name, fqcn, wait, result_q))
    proc.start()
    try:
        result = result_q.get(timeout=wait + 30)
    finally:
        proc.join(15)
    assert "error" not in result, result.get("error")
    return result


def test_bootstrap_cell_authenticates_with_its_jobs_cert(parent):
    result = _bootstrap(parent, "job", _CONNECT_TIMEOUT)

    assert result["rc"] == ReturnCode.OK


def test_bootstrap_cell_rejected_with_another_jobs_cert(parent):
    fqcn = make_workspace_transfer_fqcn("server", _JOB_ID)
    result = _bootstrap(parent, "other_job", _REJECT_WAIT)

    assert result["rc"] != ReturnCode.OK
    reject_q = parent[2]
    deadline = time.time() + _REJECTION_LOG_WAIT
    while True:
        message = reject_q.get(timeout=max(0.1, deadline - time.time()))
        if fqcn in message:
            break
    assert f"bound to job '{_OTHER_JOB_ID}'" in message
