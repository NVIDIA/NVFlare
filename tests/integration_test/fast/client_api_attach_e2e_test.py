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

"""Fast integration coverage for Cell-based Client API modes with real trainers and Cells."""

import json
import os
import subprocess
import sys
import textwrap
import time
import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest

import nvflare
from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_constant import FLContextKey, FLMetaKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.apis.utils.decomposers import flare_decomposers
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.decomposers import common_decomposers
from nvflare.app_common.executors.client_api.attach_backend import AttachBackend
from nvflare.app_common.executors.client_api.backend_spec import ClientAPIBackendContext
from nvflare.app_common.executors.client_api.external_process_backend import ExternalProcessBackend
from nvflare.app_common.np.constants import NPConstants
from nvflare.client.cell.defs import CHANNEL, Topic
from nvflare.client.config import ExchangeFormat
from nvflare.fuel.data_event.utils import set_scope_property
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply, new_cell_message
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.streaming.download_service import DownloadService
from nvflare.fuel.f3.streaming.stream_const import StreamHeaderKey
from nvflare.fuel.utils.fobs import FOBSContextKey
from nvflare.fuel.utils.network_utils import get_open_ports
from nvflare.private.fed.authenticator import validate_auth_headers

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(nvflare.__file__)))
_TRAINER_SCRIPT = textwrap.dedent(
    """
    import sys

    if "--deny-network" in sys.argv:
        denied_events = {
            "socket.__new__",
            "socket.bind",
            "socket.connect",
            "socket.connect_ex",
            "socket.getaddrinfo",
            "socket.sendto",
        }

        def deny_network(event, args):
            if event in denied_events:
                raise RuntimeError(f"network access is forbidden for this trainer: {event}")

        sys.addaudithook(deny_network)

    import numpy as np

    import nvflare.client as flare
    from nvflare.app_common.np.constants import NPConstants

    config_file = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "--deny-network" else None
    flare.init(config_file=config_file)
    while flare.is_running():
        model = flare.receive()
        if model is None:
            break
        weights = np.asarray(model.params[NPConstants.NUMPY_KEY])
        flare.send(
            flare.FLModel(
                params={NPConstants.NUMPY_KEY: weights + 1},
                current_round=model.current_round,
            )
        )
    """
)


def _wait_for_listener(cell: Cell, timeout: float = 10.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        url = cell.get_internal_listener_url()
        if url:
            return url.replace("localhost", "127.0.0.1")
        time.sleep(0.05)
    raise AssertionError(f"Cell {cell.get_fqcn()} did not create an internal listener")


def _fl_ctx(cell: Cell, site_name: str, job_id: str, secure_mode: bool = False) -> FLContext:
    if secure_mode:
        set_scope_property(site_name, FLMetaKey.AUTH_TOKEN, "site-auth-token")
        set_scope_property(site_name, FLMetaKey.AUTH_TOKEN_SIGNATURE, "site-auth-signature")
    engine = MagicMock()
    engine.get_cell.return_value = cell
    engine.new_context.return_value.__enter__.return_value = FLContext()
    fl_ctx = FLContext()
    fl_ctx.put(ReservedKey.ENGINE, engine, private=True, sticky=False)
    fl_ctx.put(ReservedKey.RUN_NUM, job_id, private=False, sticky=False)
    fl_ctx.put(ReservedKey.IDENTITY_NAME, site_name, private=False, sticky=False)
    fl_ctx.put(FLContextKey.CURRENT_JOB_ID, job_id, private=False, sticky=False)
    fl_ctx.put(FLContextKey.SECURE_MODE, secure_mode, private=True, sticky=False)
    return fl_ctx


def _stop_process(process: subprocess.Popen) -> tuple[str, str]:
    try:
        return process.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            return process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.communicate(timeout=5)


def _add_server_auth_policy(cell: Cell, site_name: str, authenticated_origins: list) -> None:
    token_verifier = MagicMock()
    token_verifier.verify.return_value = True
    auth_logger = MagicMock()

    def validate_server_auth(message):
        if (
            message.get_header(StreamHeaderKey.CHANNEL) != CHANNEL
            or message.get_header(StreamHeaderKey.TOPIC) != Topic.RESULT_READY
        ):
            return None
        reply = validate_auth_headers(
            message=message,
            token_verifier=token_verifier,
            logger=auth_logger,
            client_fqcn_resolver=lambda _name, _token: site_name,
            local_cell_fqcn=cell.get_fqcn(),
        )
        if reply is None:
            authenticated_origins.append(message.get_header(MessageHeaderKey.ORIGIN))
        return reply

    cell.core_cell.add_incoming_filter(channel="*", topic="*", cb=validate_server_auth)


def _record_download_transactions(monkeypatch):
    created_transaction_cells = []
    real_new_transaction = DownloadService.new_transaction.__func__

    def record_new_transaction(cls, *args, **kwargs):
        cell = kwargs.get("cell") if "cell" in kwargs else (args[0] if args else None)
        created_transaction_cells.append(cell)
        return real_new_transaction(cls, *args, **kwargs)

    monkeypatch.setattr(DownloadService, "new_transaction", classmethod(record_new_transaction))
    return created_transaction_cells


@pytest.mark.timeout(60)
@pytest.mark.parametrize("transport", ["tcp", "grpc", "http", "shared-file"])
@pytest.mark.parametrize("startup_order", ["trainer-first", "cj-first"])
def test_external_trainer_attaches_and_completes_numpy_task(tmp_path, monkeypatch, transport, startup_order):
    flare_decomposers.register()
    common_decomposers.register()
    server_url = f"tcp://127.0.0.1:{get_open_ports(1)[0]}"
    suffix = uuid.uuid4().hex[:8]
    site_name = f"site-{suffix}"
    job_id = f"job-{suffix}"
    attach_id = f"trainer_{suffix}"
    cells = []
    backend = None
    trainer = None
    secure_job = transport == "shared-file"
    completed = False

    try:
        received = {}
        server = Cell(f"server-{suffix}", server_url, secure=False, credentials={})
        server.start()
        cells.append(server)

        site = Cell(
            site_name,
            server_url,
            secure=False,
            credentials={},
            create_internal_listener=True,
        )
        site.start()
        cells.append(site)
        site_listener = _wait_for_listener(site)
        site.register_request_cb(
            channel="attach_e2e",
            topic="result",
            cb=lambda request: received.update(result=request.payload) or make_reply(ReturnCode.OK),
        )

        if transport == "shared-file":
            attach_resources = {
                "root_dir": str(tmp_path / "attach"),
                DriverParams.CONNECTION_SECURITY.value: "clear",
                "poll_interval": 0.005,
                "max_poll_interval": 0.05,
                "lease_interval": 0.2,
                "lease_timeout": 5,
            }
            trainer_connection = {
                "rendezvous_dir": attach_resources["root_dir"],
            }
        else:
            attach_port = get_open_ports(1)[0]
            attach_resources = {
                "host": "127.0.0.1",
                "port": attach_port,
                DriverParams.CONNECTION_SECURITY.value: "clear",
            }
            trainer_connection = {
                "connect_url": f"{transport}://127.0.0.1:{attach_port}",
                "cj_fqcn": f"{site_name}.{job_id}",
                "connection_security": "clear",
            }

        cj = Cell(
            f"{site_name}.{job_id}",
            server_url,
            secure=False,
            credentials={},
            parent_url=site_listener,
            create_internal_listener=False,
        )
        cj.start()
        cells.append(cj)
        cj.core_cell.comm_configurator.config = {
            "client_api_attach": {
                "scheme": transport,
                "resources": attach_resources,
            }
        }

        profile = tmp_path / "attach_profile.json"
        profile.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "execution_mode": "attach",
                    "attach_id": attach_id,
                    "site_name": site_name,
                    "job_wait_timeout": 20.0,
                    **trainer_connection,
                }
            )
        )
        trainer_script = tmp_path / "trainer.py"
        trainer_script.write_text(_TRAINER_SCRIPT)
        env = os.environ.copy()
        env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")

        def start_trainer():
            return subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(trainer_script),
                    str(profile),
                    *(["--deny-network"] if transport == "shared-file" else []),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )

        backend = AttachBackend()
        # Protected shared-file exercises secure credential delegation without
        # granting the trainer any socket access. Clear network cases remain
        # explicit non-secure development routes.
        fl_ctx = _fl_ctx(cj, site_name, job_id, secure_mode=secure_job)
        context = ClientAPIBackendContext(
            executor=MagicMock(),
            attach_id=attach_id,
            attach_timeout=20.0,
            heartbeat_interval=0.5,
            heartbeat_timeout=5.0,
            task_wait_timeout=20.0,
            result_wait_timeout=20.0,
            allow_insecure_attach=transport != "shared-file",
            params_exchange_format=ExchangeFormat.NUMPY,
            server_expected_format=ExchangeFormat.NUMPY,
        )
        if startup_order == "trainer-first":
            trainer = start_trainer()
        backend.initialize(context, fl_ctx)
        if startup_order == "cj-first":
            # Make the first SESSION_OPEN fail before the trainer Cell exists.
            # Attach must keep retrying with the same session until the trainer
            # starts or attach_timeout expires.
            time.sleep(1.0)
            assert not backend._get_session().ready.is_set()
            assert backend._session_thread.is_alive()
            trainer = start_trainer()

        authenticated_origins = []
        if secure_job:
            # Wait until protected SESSION_OPEN has installed the trainer's
            # outgoing auth filters, then apply the server's auth policy to
            # subsequent trainer-originated requests arriving at the CJ.
            deadline = time.monotonic() + 20.0
            while not backend._get_session().ready.is_set() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert backend._get_session().ready.is_set()
            _add_server_auth_policy(cj, site_name, authenticated_origins)

        created_transaction_cells = _record_download_transactions(monkeypatch)

        # Cross the ViaDownloader threshold and run twice. The second round
        # proves round one's trainer-hosted source settled instead of leaving
        # the single-threaded trainer blocked in flare.send().
        initial = np.arange(1024 * 1024, dtype=np.float32).reshape(1024, 1024)
        for current_round in range(2):
            task = DXO(DataKind.WEIGHTS, {NPConstants.NUMPY_KEY: initial}).to_shareable()
            task.set_header(AppConstants.CURRENT_ROUND, current_round)
            task.set_header(FOBSContextKey.RECEIVER_IDS, [site.get_fqcn()])
            result = backend.execute("train", task, fl_ctx, Signal())

            # The CJ deliberately holds lazy references. Forward the result to
            # the ultimate site receiver. Cell messages physically traverse
            # the CJ, but the trainer must remain the DownloadService source.
            received.clear()
            created_transaction_cells.clear()
            reply = cj.send_request(
                channel="attach_e2e",
                topic="result",
                target=site.get_fqcn(),
                request=new_cell_message({}, result),
                timeout=20.0,
            )
            assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
            assert not any(
                cell is cj for cell in created_transaction_cells
            ), "pass-through forwarding created a second CJ DownloadService transaction"
            result_dxo = from_shareable(received["result"])
            np.testing.assert_array_equal(result_dxo.data[NPConstants.NUMPY_KEY], initial + 1)
            initial = result_dxo.data[NPConstants.NUMPY_KEY]
        if secure_job:
            trainer_fqcn = f"{site_name}.{job_id}.-client_api_{attach_id}"
            assert trainer_fqcn in authenticated_origins
        completed = True
    finally:
        if backend is not None:
            if not completed:
                session = backend._get_session()
                if session is not None:
                    session.result_source_live.clear()
            backend.finalize(_fl_ctx(cells[-1], site_name, job_id, secure_mode=secure_job))
        stdout, stderr = _stop_process(trainer) if trainer is not None else ("", "")
        for cell in reversed(cells):
            fqcn = cell.get_fqcn()
            cell.stop()
            CoreCell.ALL_CELLS.pop(fqcn, None)

    assert trainer.returncode == 0, f"trainer failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"


@pytest.mark.timeout(60)
def test_secure_external_process_delegates_auth_and_keeps_trainer_as_source(tmp_path, monkeypatch):
    flare_decomposers.register()
    common_decomposers.register()
    server_url = f"tcp://127.0.0.1:{get_open_ports(1)[0]}"
    suffix = uuid.uuid4().hex[:8]
    site_name = f"site-{suffix}"
    job_id = f"job-{suffix}"
    cells = []
    backend = None
    completed = False

    try:
        received = {}
        server = Cell(f"server-{suffix}", server_url, secure=False, credentials={})
        server.start()
        cells.append(server)

        site = Cell(site_name, server_url, secure=False, credentials={}, create_internal_listener=True)
        site.start()
        cells.append(site)
        site_listener = _wait_for_listener(site)
        site.register_request_cb(
            channel="external_e2e",
            topic="result",
            cb=lambda request: received.update(result=request.payload) or make_reply(ReturnCode.OK),
        )

        cj = Cell(
            f"{site_name}.{job_id}",
            server_url,
            secure=False,
            credentials={},
            parent_url=site_listener,
            create_internal_listener=False,
        )
        cj.start()
        cells.append(cj)

        trainer_script = tmp_path / "trainer.py"
        trainer_script.write_text(_TRAINER_SCRIPT)
        fl_ctx = _fl_ctx(cj, site_name, job_id, secure_mode=True)
        workspace = MagicMock()
        workspace.get_app_dir.return_value = str(tmp_path)
        workspace.get_app_custom_dir.return_value = str(tmp_path)
        fl_ctx.put(FLContextKey.WORKSPACE_OBJECT, workspace, private=True, sticky=False)
        context = ClientAPIBackendContext(
            executor=MagicMock(),
            command=f"{sys.executable} -u {trainer_script}",
            launch_once=True,
            launch_timeout=20.0,
            heartbeat_interval=0.5,
            heartbeat_timeout=5.0,
            task_wait_timeout=20.0,
            result_wait_timeout=20.0,
            shutdown_timeout=5.0,
            params_exchange_format=ExchangeFormat.NUMPY,
            server_expected_format=ExchangeFormat.NUMPY,
        )
        backend = ExternalProcessBackend()
        backend.initialize(context, fl_ctx)

        authenticated_origins = []
        _add_server_auth_policy(cj, site_name, authenticated_origins)
        created_transaction_cells = _record_download_transactions(monkeypatch)

        initial = np.arange(1024 * 1024, dtype=np.float32).reshape(1024, 1024)
        for current_round in range(2):
            task = DXO(DataKind.WEIGHTS, {NPConstants.NUMPY_KEY: initial}).to_shareable()
            task.set_header(AppConstants.CURRENT_ROUND, current_round)
            task.set_header(FOBSContextKey.RECEIVER_IDS, [site.get_fqcn()])
            result = backend.execute("train", task, fl_ctx, Signal())

            received.clear()
            created_transaction_cells.clear()
            reply = cj.send_request(
                channel="external_e2e",
                topic="result",
                target=site.get_fqcn(),
                request=new_cell_message({}, result),
                timeout=20.0,
            )
            assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
            assert not any(
                cell is cj for cell in created_transaction_cells
            ), "pass-through forwarding created a second CJ DownloadService transaction"
            result_dxo = from_shareable(received["result"])
            np.testing.assert_array_equal(result_dxo.data[NPConstants.NUMPY_KEY], initial + 1)
            initial = result_dxo.data[NPConstants.NUMPY_KEY]

        assert f"{site_name}.{job_id}.client_api_trainer_1" in authenticated_origins
        completed = True
    finally:
        if backend is not None:
            if not completed and backend._active_launch is not None:
                backend._active_launch.result_source_live.clear()
            backend.finalize(fl_ctx)
        for cell in reversed(cells):
            fqcn = cell.get_fqcn()
            cell.stop()
            CoreCell.ALL_CELLS.pop(fqcn, None)
