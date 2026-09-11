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
from types import SimpleNamespace
from unittest.mock import MagicMock

from nvflare.apis.fl_constant import ConnPropKey, SecureTrainConst
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.private.fed.client import fed_client_base
from nvflare.private.fed.client.fed_client_base import FederatedClientBase

_SITE_ARGS = {
    SecureTrainConst.SSL_ROOT_CERT: "rootCA.pem",
    SecureTrainConst.SSL_CERT: "client.crt",
    SecureTrainConst.PRIVATE_KEY: "client.key",
}
# in a job process the configer has already made the job credential the ssl_cert / ssl_private_key
_JOB_ARGS = {
    SecureTrainConst.SSL_ROOT_CERT: "rootCA.pem",
    SecureTrainConst.SSL_CERT: "job.crt",
    SecureTrainConst.PRIVATE_KEY: "job.key",
}


def _make_client():
    client = FederatedClientBase.__new__(FederatedClientBase)
    client._shutdown_lock = threading.Lock()
    client.communicator = SimpleNamespace(heartbeat_done=False)
    client.cell = MagicMock()
    client.engine = None
    client.client_name = "site-1"
    client.logger = MagicMock()
    client.terminate = MagicMock()
    client.logout_client = MagicMock()
    return client


def _create_cell_credentials(monkeypatch, job_id, client_args):
    captured = {}

    class _FakeCell:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def start(self):
            pass

        def stop(self):
            pass

    conn_props = {ConnPropKey.CP_CONN_PROPS: {ConnPropKey.FQCN: "site-1", ConnPropKey.URL: "tcp://cp:1"}}
    monkeypatch.setattr(fed_client_base, "Cell", _FakeCell)
    monkeypatch.setattr(fed_client_base, "NetAgent", lambda cell: MagicMock())
    monkeypatch.setattr(fed_client_base.mpm, "add_cleanup_cb", lambda cb: None)
    monkeypatch.setattr(
        fed_client_base, "get_scope_property", lambda name, key, default=None: conn_props.get(key, default)
    )

    client = _make_client()
    client.secure_train = True
    client.client_args = dict(client_args)
    client.args = SimpleNamespace(job_id=job_id)
    client.communicator = MagicMock()
    client.engine_create_timeout = 1.0
    client.cell_check_frequency = 0.001
    client.engine = MagicMock()
    client.client_runner = MagicMock()

    client._create_cell("localhost:8002", "grpc")
    return captured["credentials"]


def test_cp_cell_uses_site_credential(monkeypatch):
    credentials = _create_cell_credentials(monkeypatch, None, _SITE_ARGS)

    assert credentials[DriverParams.CLIENT_CERT.value] == "client.crt"
    assert credentials[DriverParams.CLIENT_KEY.value] == "client.key"
    assert DriverParams.SERVER_CERT.value not in credentials


def test_cj_cell_uses_job_credential_in_both_tls_roles(monkeypatch):
    credentials = _create_cell_credentials(monkeypatch, "job-1", _JOB_ARGS)

    assert credentials[DriverParams.CLIENT_CERT.value] == "job.crt"
    assert credentials[DriverParams.CLIENT_KEY.value] == "job.key"
    assert credentials[DriverParams.SERVER_CERT.value] == "job.crt"
    assert credentials[DriverParams.SERVER_KEY.value] == "job.key"


def test_send_request_before_shutdown_skips_after_close():
    client = _make_client()
    reply = MagicMock()
    client.cell.send_request.return_value = reply

    assert client.send_request_before_shutdown(topic="terminal_outcome") is reply

    client.close()

    assert client.communicator.heartbeat_done is True
    assert client.send_request_before_shutdown(topic="terminal_outcome") is None
    client.cell.send_request.assert_called_once_with(topic="terminal_outcome")
    client.logout_client.assert_called_once()


def test_close_waits_for_terminal_request_before_logout():
    client = _make_client()
    request_started = threading.Event()
    release_request = threading.Event()
    close_entered = threading.Event()

    def send_request(**_kwargs):
        request_started.set()
        assert release_request.wait(timeout=1.0)
        return MagicMock()

    client.cell.send_request.side_effect = send_request
    client.terminate.side_effect = close_entered.set

    request_thread = threading.Thread(target=client.send_request_before_shutdown, kwargs={"topic": "outcome"})
    request_thread.start()
    assert request_started.wait(timeout=1.0)

    close_thread = threading.Thread(target=client.close)
    close_thread.start()
    assert not close_entered.wait(timeout=0.1)
    client.logout_client.assert_not_called()

    release_request.set()
    request_thread.join(timeout=1.0)
    close_thread.join(timeout=1.0)

    assert not request_thread.is_alive()
    assert not close_thread.is_alive()
    assert client.communicator.heartbeat_done is True
    client.logout_client.assert_called_once()
