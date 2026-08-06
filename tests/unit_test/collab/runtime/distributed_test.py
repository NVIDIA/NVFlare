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

import importlib.util
import json
import logging
import os
import platform
import queue
import socket
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.apis.workspace import Workspace
from nvflare.collab import collab
from nvflare.collab.api.app import ClientApp
from nvflare.collab.runtime import distributed_worker as distributed_worker_module
from nvflare.collab.runtime.defs import CallReplyKey, DistributedKey, DistributedTopic, ObjectCallKey
from nvflare.collab.runtime.distributed import DistributedClientSession
from nvflare.collab.runtime.distributed_worker import DistributedWorker, _RelayCell
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.network_utils import get_open_ports
from nvflare.private.fed.utils.fed_utils import fobs_initialize

torch = None
if importlib.util.find_spec("torch"):
    import torch

gloo_available = bool(torch and torch.distributed.is_available() and torch.distributed.is_gloo_available())
running_under_xdist = bool(os.environ.get("PYTEST_XDIST_WORKER"))
GLOO_INIT_ERROR_MARKERS = (
    "Cannot resolve 127.0.0.1 to a (local) address",
    "Unable to resolve hostname",
    "unsupported gloo device",
    "uv_bind: operation not permitted",
)


class _DistributedTestClient:
    @collab.publish
    def train(self, value, fail_rank=None):
        rank = torch.distributed.get_rank()
        if rank == fail_rank:
            raise RuntimeError(f"rank {rank} failed")
        return value + rank


def _set_gloo_loopback_if_needed():
    if os.environ.get("GLOO_SOCKET_IFNAME"):
        return
    try:
        interface_names = {name for _, name in socket.if_nameindex()}
        if platform.system() == "Linux" and "lo" in interface_names:
            os.environ["GLOO_SOCKET_IFNAME"] = "lo"
        elif platform.system() == "Darwin" and "lo0" in interface_names:
            os.environ["GLOO_SOCKET_IFNAME"] = "lo0"
    except (AttributeError, OSError):
        pass


def _two_rank_worker(rank, init_path, result_queue):
    import torch.distributed as dist

    try:
        _set_gloo_loopback_if_needed()
        dist.init_process_group("gloo", init_method=f"file://{init_path}", rank=rank, world_size=2)
        worker = DistributedWorker.__new__(DistributedWorker)
        worker.rank = rank
        worker.world_size = 2
        worker.dist = dist
        worker.logger = MagicMock()
        worker.app = ClientApp(_DistributedTestClient())
        worker.app.name = "site-1"

        results = []
        for fail_rank in (None, 1):
            request = new_cell_message(
                {},
                {
                    ObjectCallKey.CALLER: "server",
                    ObjectCallKey.TARGET_NAME: "site-1.client",
                    ObjectCallKey.METHOD_NAME: "train",
                    ObjectCallKey.ARGS: [],
                    ObjectCallKey.KWARGS: {"value": 10, "fail_rank": fail_rank},
                },
            )
            if rank == 0:
                worker._broadcast("invoke", request)
            else:
                command = [None]
                dist.broadcast_object_list(command, src=0)
                request = command[0][DistributedKey.PAYLOAD]
            reply = worker._invoke_all(request)
            results.append(
                {
                    "rc": reply.get_header(MessageHeaderKey.RETURN_CODE, ReturnCode.OK),
                    "result": reply.payload.get(CallReplyKey.RESULT) if isinstance(reply.payload, dict) else None,
                    "error": reply.payload.get(CallReplyKey.ERROR) if isinstance(reply.payload, dict) else None,
                }
            )
        result_queue.put({"rank": rank, "ok": True, "results": results})
    except Exception as ex:
        result_queue.put({"rank": rank, "ok": False, "error": repr(ex)})
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_session_appends_worker_module_to_launcher_command():
    workspace = MagicMock()
    workspace.get_root_dir.return_value = "/workspace"
    session = DistributedClientSession(
        command="python3 -m torch.distributed.run --nproc_per_node=2",
        startup_timeout=10.0,
        shutdown_timeout=10.0,
        logger=MagicMock(),
    )
    session.bootstrap_path = "/workspace/job/.collab.fobs"

    argv = session._worker_command(workspace, "site-1", "job")

    assert argv[:5] == ["python3", "-m", "torch.distributed.run", "--nproc_per_node=2", "-m"]
    assert argv[5] == "nvflare.collab.runtime.distributed_worker"
    assert "--collab-bootstrap" in argv
    assert argv[-2:] == ["--startup-timeout", "10.0"]


def test_session_bootstrap_is_owner_only_and_fobs_round_trips(tmp_path):
    path = tmp_path / "bootstrap.fobs"

    DistributedClientSession._write_bootstrap(str(path), {"proof": "secret"})

    assert fobs.loadf(str(path)) == {"proof": "secret"}
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600


def test_nonzero_rank_cannot_make_outbound_collab_call():
    relay = _RelayCell(
        cell=None,
        parent_fqcn="site-1.job",
        outbound_topic="session:outbound",
        session_id="session",
        rank=1,
        secure_supported=False,
        abort_signal=Signal(),
    )

    with pytest.raises(RuntimeError, match="require global rank 0"):
        relay.send_request(
            channel="collab",
            target="server.job",
            topic="call",
            request=MagicMock(),
            timeout=1.0,
        )


def test_rank_zero_relay_uses_private_collab_session():
    cell = MagicMock()
    expected = new_cell_message({}, {CallReplyKey.RESULT: "ok"})
    cell.send_request.return_value = expected
    abort_signal = Signal()
    relay = _RelayCell(
        cell=cell,
        parent_fqcn="site-1.job",
        outbound_topic="session:outbound",
        session_id="session",
        rank=0,
        secure_supported=True,
        abort_signal=abort_signal,
    )
    target_request = new_cell_message({}, {"value": 1})

    reply = relay.send_request(
        channel="collab",
        target="server.job",
        topic="call",
        request=target_request,
        timeout=3.0,
        secure=True,
    )

    assert reply is expected
    call = cell.send_request.call_args.kwargs
    assert call["channel"] == "collab_distributed"
    assert call["topic"] == "session:outbound"
    assert call["target"] == "site-1.job"
    assert call["abort_signal"] is abort_signal
    assert call["request"].payload[DistributedKey.SESSION_ID] == "session"
    assert call["request"].payload[DistributedKey.TARGET] == "server.job"
    assert call["request"].payload[DistributedKey.SECURE] is True


def test_parent_accepts_only_authenticated_worker_hello():
    session = DistributedClientSession("python3 -u", 10.0, 10.0, MagicMock())
    session.worker_fqcn = "site-1.job.worker"

    forged = session._handle_hello(
        new_cell_message(
            {MessageHeaderKey.ORIGIN: session.worker_fqcn},
            {
                DistributedKey.PROTOCOL_VERSION: 1,
                DistributedKey.AUTH_TOKEN: "wrong-token",
            },
        )
    )
    accepted = session._handle_hello(
        new_cell_message(
            {MessageHeaderKey.ORIGIN: session.worker_fqcn},
            {
                DistributedKey.PROTOCOL_VERSION: 1,
                DistributedKey.AUTH_TOKEN: session.auth_token,
            },
        )
    )

    assert forged.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST
    assert accepted.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
    assert accepted.payload[DistributedKey.SESSION_ID] == session.session_id


def test_first_rank_error_is_selected_deterministically():
    statuses = [
        {"rank": 0, DistributedKey.OK: True},
        {"rank": 1, DistributedKey.OK: False, DistributedKey.ERROR: "rank one failed"},
        {"rank": 2, DistributedKey.OK: False, DistributedKey.ERROR: "rank two failed"},
    ]

    assert DistributedWorker._first_error(statuses) == statuses[1]


def test_worker_component_builder_keeps_authorization_enabled(tmp_path, monkeypatch):
    client_config = tmp_path / "config_fed_client.json"
    client_config.write_text(
        json.dumps(
            {
                "components": [
                    {
                        "id": "client",
                        "path": "nvflare.collab.api.module_wrapper.ModuleWrapper",
                        "args": {"module": "client_module"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    worker = DistributedWorker.__new__(DistributedWorker)
    worker.workspace = MagicMock()
    worker.context = MagicMock()
    worker.components = {}
    builder = MagicMock()
    builder.build_component.return_value = object()
    builder_type = MagicMock(return_value=builder)
    monkeypatch.setattr(distributed_worker_module, "WorkerComponentBuilder", builder_type)

    worker._build_components(
        {
            "client_config": str(client_config),
            "client_obj_id": "client",
            "collab_obj_ids": [],
        }
    )

    builder_type.assert_called_once_with(fl_ctx=worker.context, workspace=worker.workspace)
    assert worker.components["client"] is builder.build_component.return_value


def test_nonzero_rank_skips_broadcast_loop_after_failed_initialization():
    worker = DistributedWorker.__new__(DistributedWorker)
    worker.initialized = False
    worker.dist = None

    worker._run_nonzero_rank()


def test_session_forwards_invocation_timeout_to_worker_cell():
    session = DistributedClientSession(
        command="python3 -u",
        startup_timeout=10.0,
        shutdown_timeout=10.0,
        logger=MagicMock(),
    )
    expected_reply = new_cell_message({}, {CallReplyKey.RESULT: "done"})
    session.cell = MagicMock()
    session.cell.send_request.return_value = expected_reply
    session.worker_fqcn = "site-1.job.worker"
    session.session_id = "session"
    session.started = True
    session.abort_signal = Signal()
    request = new_cell_message({}, {ObjectCallKey.TIMEOUT: 3.5})

    reply = session.invoke(request)

    assert reply.headers == expected_reply.headers
    assert reply.payload == expected_reply.payload
    request_args = session.cell.send_request.call_args.kwargs
    assert request_args["target"] == "site-1.job.worker"
    assert request_args["topic"] == session._topic(DistributedTopic.INVOKE)
    assert request_args["timeout"] == 3.5


def test_transport_timeout_fails_session_and_rejects_subsequent_call():
    session = DistributedClientSession("python3 -u", 10.0, 10.0, MagicMock())
    session.cell = MagicMock()
    session.cell.send_request.return_value = new_cell_message(
        {MessageHeaderKey.RETURN_CODE: ReturnCode.TIMEOUT},
        None,
    )
    session.worker_fqcn = "site-1.job.worker"
    session.session_id = "session"
    session.started = True
    session.abort_signal = Signal()
    request = new_cell_message({}, {ObjectCallKey.TIMEOUT: 0.1})

    first = session.invoke(request)
    second = session.invoke(request)

    assert first.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.TIMEOUT
    assert second.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.PROCESS_EXCEPTION
    assert "interrupted invocation" in second.payload[CallReplyKey.ERROR]
    session.cell.send_request.assert_called_once()


def test_session_finalizes_with_fresh_signal_after_run_signal_is_triggered():
    session = DistributedClientSession(
        command="python3 -u",
        startup_timeout=10.0,
        shutdown_timeout=10.0,
        logger=MagicMock(),
    )
    session.cell = MagicMock()
    session.cell.send_request.return_value = new_cell_message({}, None)
    session.worker_fqcn = "site-1.job.worker"
    session.session_id = "session"
    session.started = True
    session.abort_signal = Signal()
    session.abort_signal.trigger("end run")

    session.stop(finalize=True)

    finalize_signal = session.cell.send_request.call_args_list[0].kwargs["abort_signal"]
    assert finalize_signal is not session.abort_signal
    assert not finalize_signal.triggered
    assert session.cell.send_request.call_count == 2


def test_session_reports_nonzero_worker_exit_after_finalization():
    session = DistributedClientSession(
        command="python3 -u",
        startup_timeout=10.0,
        shutdown_timeout=10.0,
        logger=MagicMock(),
    )
    session.started = True
    session.process = MagicMock()
    session._send_worker_request = MagicMock(return_value=new_cell_message({}, None))
    session._close_worker = MagicMock()
    session._wait_or_terminate_worker = MagicMock(return_value=1)

    with pytest.raises(RuntimeError, match="worker launcher exited with code 1 after finalization"):
        session.stop(finalize=True)

    assert session.closed is True
    assert session.session_id is None


@pytest.mark.timeout(30)
def test_single_process_session_runs_reconstructed_app_end_to_end(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[4]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(repo_root), os.environ.get("PYTHONPATH", "")]))
    workspace_root = tmp_path / "workspace"
    (workspace_root / "startup").mkdir(parents=True)
    (workspace_root / "local").mkdir()
    app_dir = workspace_root / "job" / "app_site-1"
    config_dir = app_dir / "config"
    custom_dir = app_dir / "custom"
    config_dir.mkdir(parents=True)
    custom_dir.mkdir()
    (workspace_root / "job" / "meta.json").write_text('{"byoc": true}', encoding="utf-8")
    (custom_dir / "distributed_e2e_client.py").write_text(
        "from pathlib import Path\n"
        "from nvflare.collab import collab\n"
        "@collab.init\n"
        "def initialize():\n"
        "    Path(collab.workspace.get_run_dir(collab.fl_ctx.get_job_id()), 'initialized').write_text('yes')\n"
        "@collab.publish\n"
        "def train(value):\n"
        "    return value * 2\n"
        "@collab.final\n"
        "def finalize():\n"
        "    Path(collab.workspace.get_run_dir(collab.fl_ctx.get_job_id()), 'finalized').write_text('yes')\n",
        encoding="utf-8",
    )
    client_config = {
        "format_version": 2,
        "executors": [],
        "components": [
            {
                "id": "_client",
                "path": "nvflare.collab.api.module_wrapper.ModuleWrapper",
                "args": {"module": "distributed_e2e_client"},
            }
        ],
        "task_data_filters": [],
        "task_result_filters": [],
    }
    (config_dir / "config_fed_client.json").write_text(json.dumps(client_config), encoding="utf-8")

    fobs_initialize()
    port = get_open_ports(1)[0]
    root_cell = Cell("server", f"tcp://localhost:{port}", secure=False, credentials={})
    parent_cell = Cell("site-1.job", f"tcp://localhost:{port}", secure=False, credentials={})
    parent_cell.core_cell.start()
    workspace = Workspace(str(workspace_root), "site-1")
    engine = MagicMock()
    engine.get_cell.return_value = parent_cell
    engine.get_workspace.return_value = workspace
    fl_ctx = FLContext()
    fl_ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=False)
    fl_ctx.set_prop(ReservedKey.RUN_NUM, "job", private=False, sticky=False)
    fl_ctx.set_prop(ReservedKey.IDENTITY_NAME, "site-1", private=False, sticky=False)
    fl_ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace, private=True, sticky=False)
    fl_ctx.set_prop(FLContextKey.CURRENT_JOB_ID, "job", private=False, sticky=False)
    fl_ctx.set_prop(FLContextKey.SECURE_MODE, False, private=True, sticky=False)
    test_logger = logging.getLogger("distributed_session_e2e")
    test_logger.setLevel(logging.INFO)
    session = DistributedClientSession(
        command=f"{sys.executable} -u",
        startup_timeout=15.0,
        shutdown_timeout=10.0,
        logger=test_logger,
    )
    run_abort_signal = Signal()
    try:
        session.start(
            fl_ctx=fl_ctx,
            client_obj_id="_client",
            collab_obj_ids=[],
            props={},
            server_spec={"name": "server", "fqn": "server", "target": "server", "interface": {"": {}}},
            client_specs=[{"name": "site-1", "fqn": "site-1", "target": "server", "interface": {"": {}}}],
            abort_signal=run_abort_signal,
        )
        request = new_cell_message(
            {},
            {
                ObjectCallKey.CALLER: "server",
                ObjectCallKey.TARGET_NAME: "site-1.client",
                ObjectCallKey.METHOD_NAME: "train",
                ObjectCallKey.ARGS: [],
                ObjectCallKey.KWARGS: {"value": 7},
                ObjectCallKey.TIMEOUT: 10.0,
            },
        )

        reply = session.invoke(request)

        assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
        assert reply.payload[CallReplyKey.RESULT] == 14
        assert (workspace_root / "job" / "initialized").read_text(encoding="utf-8") == "yes"
        run_abort_signal.trigger("end run")
        session.stop(finalize=True)
        assert (workspace_root / "job" / "finalized").read_text(encoding="utf-8") == "yes"
    finally:
        session.stop(finalize=False)
        parent_cell.core_cell.stop()
        CoreCell.ALL_CELLS.pop(parent_cell.get_fqcn(), None)
        CoreCell.ALL_CELLS.pop(root_cell.get_fqcn(), None)


@pytest.mark.skipif(not gloo_available, reason="torch.distributed gloo is required")
@pytest.mark.skipif(running_under_xdist, reason="nested torch multiprocessing is unstable under xdist")
def test_two_rank_gloo_runs_every_rank_and_propagates_rank_failure(tmp_path):
    context = torch.multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    init_path = tmp_path / "collab-gloo.init"
    processes = [context.Process(target=_two_rank_worker, args=(rank, init_path, result_queue)) for rank in range(2)]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
    if platform.system() == "Darwin" and any(process.exitcode != 0 for process in processes):
        pytest.skip("torch.distributed gloo could not bind its loopback transport on this macOS runner")
    assert all(process.exitcode == 0 for process in processes)

    results = []
    for _ in processes:
        try:
            results.append(result_queue.get(timeout=5))
        except queue.Empty:
            pytest.fail("timed out waiting for distributed Collab worker")
    results.sort(key=lambda item: item["rank"])

    if results and all(
        not item["ok"] and any(marker in item.get("error", "") for marker in GLOO_INIT_ERROR_MARKERS)
        for item in results
    ):
        pytest.skip(f"torch.distributed gloo could not initialize on this runner: {results}")
    assert all(item["ok"] for item in results), results
    assert [item["results"][0]["result"] for item in results] == [10, 11]
    assert all(item["results"][1]["rc"] == ReturnCode.PROCESS_EXCEPTION for item in results)
    assert all("rank 1" in item["results"][1]["error"] for item in results)
