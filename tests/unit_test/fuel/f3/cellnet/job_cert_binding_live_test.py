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

"""Live mTLS regression: a site parent's internal listener binds job certificates to their job's FQCNs."""

import json
import multiprocessing as mp
import os
import socket
import time
import traceback
import uuid

import pytest
from cryptography import x509

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.message import Message
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.lighter.constants import CertExtensionOID
from nvflare.lighter.utils import Identity, generate_cert, generate_keys, serialize_cert, serialize_pri_key
from nvflare.private.fed.utils.job_cert_utils import JobCertIssuer

_CHANNEL = "job_binding_test"
_TOPIC = "echo"
_CONNECT_TIMEOUT = 10.0
_REJECT_WAIT = 3.0
_JOB_A = str(uuid.uuid4())
_JOB_B = str(uuid.uuid4())


def _write_pki(out_dir: str) -> dict:
    root_key, root_pub = generate_keys()
    root_cert = generate_cert(Identity("rootCA"), Identity("rootCA"), root_key, root_pub, ca=True)
    server_key, server_pub = generate_keys()
    server_cert = generate_cert(
        Identity("server"), Identity("rootCA"), root_key, server_pub, server_default_host="localhost"
    )
    site_key, site_pub = generate_keys()
    site_cert = generate_cert(
        Identity("site-1"), Identity("rootCA"), root_key, site_pub, server_default_host="localhost"
    )
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
    job_a_crt, job_a_key = issuer.issue("site-1", _JOB_A)
    job_b_crt, job_b_key = issuer.issue("site-1", _JOB_B)
    files = {
        "rootCA.pem": serialize_cert(root_cert),
        "server.crt": serialize_cert(server_cert),
        "server.key": serialize_pri_key(server_key),
        "site-1.crt": serialize_cert(site_cert),
        "site-1.key": serialize_pri_key(site_key),
        "job_a.crt": job_a_crt,
        "job_a.key": job_a_key,
        "job_b.crt": job_b_crt,
        "job_b.key": job_b_key,
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


def _run_server(root_url, pki, ready_q, stop_ev):
    cell = None
    try:
        credentials = {
            DriverParams.CA_CERT.value: pki["rootCA.pem"],
            DriverParams.SERVER_CERT.value: pki["server.crt"],
            DriverParams.SERVER_KEY.value: pki["server.key"],
            DriverParams.CONNECTION_SECURITY.value: "mtls",
        }
        cell = Cell("server", root_url, secure=True, credentials=credentials, create_internal_listener=False)
        cell.start()
        ready_q.put("ready")
        stop_ev.wait(120)
    except Exception:
        ready_q.put(traceback.format_exc())
    finally:
        if cell:
            cell.stop()


def _run_site_parent(root_url, pki, config_dir, ready_q, stop_ev):
    cell = None
    try:
        # the internal listener scheme and security come from comm_config.json, as in a provisioned kit
        ConfigService.initialize(section_files={}, config_path=[config_dir])
        credentials = {
            DriverParams.CA_CERT.value: pki["rootCA.pem"],
            DriverParams.CLIENT_CERT.value: pki["site-1.crt"],
            DriverParams.CLIENT_KEY.value: pki["site-1.key"],
            DriverParams.SERVER_CERT.value: pki["site-1.crt"],
            DriverParams.SERVER_KEY.value: pki["site-1.key"],
            DriverParams.CONNECTION_SECURITY.value: "mtls",
        }
        cell = Cell(
            "site-1",
            root_url,
            secure=True,
            credentials=credentials,
            create_internal_listener=True,
            auth_identity="site-1",
            auth_identity_map={"site-1": "site-1", "server": "server"},
        )
        cell.register_request_cb(_CHANNEL, _TOPIC, lambda request: Message(payload=request.payload))
        cell.start()
        deadline = time.time() + _CONNECT_TIMEOUT
        while time.time() < deadline and not cell.is_cell_connected("server"):
            time.sleep(0.1)
        ready_q.put(cell.get_internal_listener_url())
        stop_ev.wait(120)
    except Exception:
        ready_q.put(traceback.format_exc())
    finally:
        if cell:
            cell.stop()


def _run_job_cell(parent_url, pki, cert_name, fqcn, wait, result_q):
    cell = None
    try:
        credentials = {
            DriverParams.CA_CERT.value: pki["rootCA.pem"],
            DriverParams.CLIENT_CERT.value: pki[f"{cert_name}.crt"],
            DriverParams.CLIENT_KEY.value: pki[f"{cert_name}.key"],
            DriverParams.CONNECTION_SECURITY.value: "mtls",
        }
        cell = Cell(
            fqcn,
            parent_url,
            secure=True,
            credentials=credentials,
            create_internal_listener=False,
            parent_url=parent_url,
            parent_resources={DriverParams.CONNECTION_SECURITY.value: "mtls"},
            auth_identity_map={"site-1": "site-1"},
        )
        cell.start()
        deadline = time.time() + wait
        while time.time() < deadline and not cell.is_cell_connected("site-1"):
            time.sleep(0.1)
        connected = cell.is_cell_connected("site-1")
        rc = None
        if connected:
            reply = cell.send_request(_CHANNEL, _TOPIC, "site-1", Message(payload="hello"), timeout=5.0)
            rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
        result_q.put({"connected": connected, "rc": rc})
    except Exception:
        result_q.put({"error": traceback.format_exc()})
    finally:
        if cell:
            cell.stop()


@pytest.fixture(scope="module")
def site_parent(tmp_path_factory):
    ctx = mp.get_context("spawn")
    pki = _write_pki(str(tmp_path_factory.mktemp("pki")))
    config_dir = str(tmp_path_factory.mktemp("config"))
    with open(os.path.join(config_dir, "comm_config.json"), "w") as f:
        json.dump(
            {"internal": {"scheme": "stcp", "resources": {"host": "localhost", "connection_security": "mtls"}}}, f
        )
    root_url = f"stcp://localhost:{_free_port()}"
    stop_ev = ctx.Event()
    server_q, parent_q = ctx.Queue(), ctx.Queue()
    server = ctx.Process(target=_run_server, args=(root_url, pki, server_q, stop_ev))
    server.start()
    parent = None
    try:
        status = server_q.get(timeout=30)
        assert status == "ready", status
        parent = ctx.Process(target=_run_site_parent, args=(root_url, pki, config_dir, parent_q, stop_ev))
        parent.start()
        internal_url = parent_q.get(timeout=40)
        assert internal_url.startswith("stcp://"), internal_url
        yield internal_url, pki
    finally:
        stop_ev.set()
        if parent:
            parent.join(15)
        server.join(15)


def _job_cell(site_parent, cert_name, fqcn, wait):
    internal_url, pki = site_parent
    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    proc = ctx.Process(target=_run_job_cell, args=(internal_url, pki, cert_name, fqcn, wait, result_q))
    proc.start()
    try:
        result = result_q.get(timeout=wait + 30)
    finally:
        proc.join(15)
    assert "error" not in result, result.get("error")
    return result


def test_site_parent_accepts_job_cell_with_its_own_job_cert(site_parent):
    result = _job_cell(site_parent, "job_b", f"site-1.{_JOB_B}", _CONNECT_TIMEOUT)

    assert result["connected"] is True
    assert result["rc"] == ReturnCode.OK


def test_site_parent_rejects_another_jobs_cert_on_job_fqcn(site_parent):
    result = _job_cell(site_parent, "job_a", f"site-1.{_JOB_B}", _REJECT_WAIT)

    assert result["connected"] is False
