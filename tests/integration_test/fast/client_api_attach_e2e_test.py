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

"""Fast integration coverage for Attach with a real external trainer process and real Cells."""

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
from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.apis.utils.decomposers import flare_decomposers
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.decomposers import common_decomposers
from nvflare.app_common.executors.client_api.attach_backend import AttachBackend
from nvflare.app_common.executors.client_api.backend_spec import ClientAPIBackendContext
from nvflare.app_common.np.constants import NPConstants
from nvflare.client.config import ExchangeFormat
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply, new_cell_message
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.utils.network_utils import get_open_ports

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

    flare.init(config_file=sys.argv[1])
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


@pytest.mark.timeout(60)
@pytest.mark.parametrize("transport", ["tcp", "grpc", "http", "shared-file"])
@pytest.mark.parametrize("startup_order", ["trainer-first", "cj-first"])
def test_external_trainer_attaches_and_completes_numpy_task(tmp_path, transport, startup_order):
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
        # Exercise the relayed-job result path independently of the dedicated
        # Attach transport's own security. This is especially load-bearing for
        # a clear shared-file trainer route in an otherwise secure job.
        fl_ctx = _fl_ctx(cj, site_name, job_id, secure_mode=True)
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

        # Cross the ViaDownloader threshold and run twice. The second round
        # proves round one's trainer-hosted source settled instead of leaving
        # the single-threaded trainer blocked in flare.send().
        initial = np.arange(1024 * 1024, dtype=np.float32).reshape(1024, 1024)
        for current_round in range(2):
            task = DXO(DataKind.WEIGHTS, {NPConstants.NUMPY_KEY: initial}).to_shareable()
            task.set_header(AppConstants.CURRENT_ROUND, current_round)
            result = backend.execute("train", task, fl_ctx, Signal())

            # The CJ deliberately holds lazy references. Forward the result to
            # its CP so the CP downloads through the secure-job relay from the
            # external trainer.
            received.clear()
            reply = cj.send_request(
                channel="attach_e2e",
                topic="result",
                target=site.get_fqcn(),
                request=new_cell_message({}, result),
                timeout=20.0,
            )
            assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
            result_dxo = from_shareable(received["result"])
            np.testing.assert_array_equal(result_dxo.data[NPConstants.NUMPY_KEY], initial + 1)
            initial = result_dxo.data[NPConstants.NUMPY_KEY]
    finally:
        if backend is not None:
            backend.finalize(_fl_ctx(cells[-1], site_name, job_id, secure_mode=True))
        stdout, stderr = _stop_process(trainer) if trainer is not None else ("", "")
        for cell in reversed(cells):
            fqcn = cell.get_fqcn()
            cell.stop()
            CoreCell.ALL_CELLS.pop(fqcn, None)

    assert trainer.returncode == 0, f"trainer failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"
