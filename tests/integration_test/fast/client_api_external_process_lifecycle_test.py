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

"""Real-process integration coverage for accepted Client API result-source teardown."""

import os
import signal
import sys
import textwrap
import threading
import time
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_constant import FLContextKey, FLMetaKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.apis.utils.decomposers import flare_decomposers
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.decomposers import common_decomposers
from nvflare.app_common.executors.client_api.backend_spec import ClientAPIBackendContext
from nvflare.app_common.executors.client_api.external_process_backend import ExternalProcessBackend
from nvflare.app_common.np.constants import NPConstants
from nvflare.client.config import ExchangeFormat
from nvflare.fuel.common.exit_codes import ProcessExitCode
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.streaming.download_service import OBJ_DOWNLOADER_CHANNEL, OBJ_DOWNLOADER_TOPIC, DownloadService
from nvflare.fuel.utils.fobs import FOBSContextKey
from nvflare.fuel.utils.fobs.decomposers.via_downloader import (
    contains_lazy_download_ref,
    materialize_lazy_download_refs,
)
from nvflare.fuel.utils.network_utils import get_open_ports

_TRAINER_SCRIPT = textwrap.dedent(
    """
    import numpy as np

    import nvflare.client as flare
    from nvflare.app_common.np.constants import NPConstants

    flare.init()
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


def _wait_until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


def _wait_for_listener(cell: Cell, timeout=10.0):
    assert _wait_until(cell.get_internal_listener_url, timeout), f"Cell {cell.get_fqcn()} has no listener"
    return cell.get_internal_listener_url().replace("localhost", "127.0.0.1")


@pytest.fixture
def external_process_env(tmp_path, monkeypatch):
    flare_decomposers.register()
    common_decomposers.register()
    # Keep the payload modest while guaranteeing enough pulls for the test to
    # pause after the producer has served (and therefore acquired) the receiver.
    monkeypatch.setenv("NVFLARE_NP_DOWNLOAD_CHUNK_SIZE", str(64 * 1024))

    suffix = uuid.uuid4().hex[:8]
    site_name = f"site-{suffix}"
    job_id = f"job-{suffix}"
    server_url = f"tcp://127.0.0.1:{get_open_ports(1)[0]}"
    cells = []
    backend = None
    run_abort_signal = Signal()

    server = Cell(f"server-{suffix}", server_url, secure=False, credentials={})
    server.start()
    cells.append(server)

    site = Cell(site_name, server_url, secure=False, credentials={}, create_internal_listener=True)
    site.start()
    cells.append(site)
    site_listener = _wait_for_listener(site)
    DownloadService.initialize(site)

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

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    trainer_script = tmp_path / "trainer.py"
    trainer_script.write_text(_TRAINER_SCRIPT)

    executor = MagicMock()
    engine = MagicMock()
    engine.get_cell.return_value = cj
    engine.new_context.return_value.__enter__.return_value = FLContext()
    workspace = MagicMock()
    workspace.get_run_dir.return_value = str(run_dir)
    workspace.get_app_dir.return_value = str(tmp_path)
    workspace.get_app_custom_dir.return_value = str(tmp_path)

    fl_ctx = FLContext()
    fl_ctx.put(ReservedKey.ENGINE, engine, private=True, sticky=False)
    fl_ctx.put(ReservedKey.RUN_NUM, job_id, private=False, sticky=False)
    fl_ctx.put(ReservedKey.IDENTITY_NAME, site_name, private=False, sticky=False)
    fl_ctx.put(ReservedKey.RUN_ABORT_SIGNAL, run_abort_signal, private=True, sticky=False)
    fl_ctx.put(FLContextKey.CURRENT_JOB_ID, job_id, private=False, sticky=False)
    fl_ctx.put(FLContextKey.WORKSPACE_OBJECT, workspace, private=True, sticky=False)

    context = ClientAPIBackendContext(
        executor=executor,
        command=f"{sys.executable} -u {trainer_script}",
        launch_once=False,
        launch_timeout=10.0,
        heartbeat_interval=0.1,
        heartbeat_timeout=2.0,
        task_wait_timeout=10.0,
        result_wait_timeout=10.0,
        shutdown_timeout=1.0,
        stop_grace_period=0.2,
        params_exchange_format=ExchangeFormat.NUMPY,
        server_expected_format=ExchangeFormat.NUMPY,
    )
    backend = ExternalProcessBackend()
    backend.initialize(context, fl_ctx)

    env = SimpleNamespace(
        backend=backend,
        cells=cells,
        cj=cj,
        executor=executor,
        fl_ctx=fl_ctx,
        job_id=job_id,
        run_abort_signal=run_abort_signal,
        run_dir=run_dir,
        site=site,
    )
    try:
        yield env
    finally:
        if backend is not None:
            with backend._launch_lock:
                trainer = backend._active_launch
            if trainer is not None and backend._process_group_alive(trainer):
                run_abort_signal.trigger("test cleanup")
                backend.abort(fl_ctx)
            backend.finalize(fl_ctx)
        DownloadService.shutdown()
        for cell in reversed(cells):
            fqcn = cell.get_fqcn()
            cell.stop()
            CoreCell.ALL_CELLS.pop(fqcn, None)


def _accepted_lazy_result(env):
    initial = np.arange(2 * 1024 * 1024, dtype=np.float32)
    task = DXO(DataKind.WEIGHTS, {NPConstants.NUMPY_KEY: initial}).to_shareable()
    task.set_header(AppConstants.CURRENT_ROUND, 0)
    task.set_header(FOBSContextKey.RECEIVER_IDS, [env.site.get_fqcn()])

    result = env.backend.execute("train", task, env.fl_ctx, Signal())
    trainer = env.backend._active_launch

    assert contains_lazy_download_ref(result)
    assert trainer is not None
    assert trainer.result_accepted.is_set()
    assert trainer.result_source_live.is_set()
    assert trainer.reaper_thread is not None and trainer.reaper_thread.is_alive()
    assert env.backend._process_group_alive(trainer)
    return result, trainer


def _start_paused_download(env, result, monkeypatch):
    first_chunk_served = threading.Event()
    release_first_chunk = threading.Event()
    download_done = threading.Event()
    download_abort = Signal()
    download_errors = []
    real_send_request = env.site.send_request
    request_lock = threading.Lock()
    request_count = 0

    def pause_after_first_chunk(*args, **kwargs):
        nonlocal request_count
        reply = real_send_request(*args, **kwargs)
        if kwargs.get("channel") == OBJ_DOWNLOADER_CHANNEL and kwargs.get("topic") == OBJ_DOWNLOADER_TOPIC:
            with request_lock:
                request_count += 1
                first = request_count == 1
            if first:
                first_chunk_served.set()
                if not release_first_chunk.wait(10.0):
                    raise TimeoutError("test did not release the first result chunk")
        return reply

    monkeypatch.setattr(env.site, "send_request", pause_after_first_chunk)

    def materialize():
        try:
            materialize_lazy_download_refs(result, env.site, abort_signal=download_abort)
        except BaseException as ex:
            download_errors.append(ex)
        finally:
            download_done.set()

    thread = threading.Thread(target=materialize, name="accepted_result_receiver")
    thread.start()
    assert first_chunk_served.wait(10.0), "receiver never acquired the accepted result source"
    return SimpleNamespace(
        abort=download_abort,
        done=download_done,
        errors=download_errors,
        release=release_first_chunk,
        thread=thread,
    )


@pytest.mark.timeout(30)
def test_run_abort_reaps_real_trainer_during_accepted_result_download(external_process_env, monkeypatch, caplog):
    env = external_process_env
    result, trainer = _accepted_lazy_result(env)
    download = _start_paused_download(env, result, monkeypatch)

    started = time.monotonic()
    env.run_abort_signal.trigger("job aborted")
    env.backend.abort(env.fl_ctx)
    download.abort.trigger("job aborted")
    download.release.set()
    env.backend.finalize(env.fl_ctx)
    elapsed = time.monotonic() - started

    download.thread.join(timeout=5.0)
    assert download.done.is_set(), "accepted-result receiver remained blocked after abort"
    assert download.errors, "aborted materialization unexpectedly completed successfully"
    assert elapsed < 5.0
    assert trainer.process.poll() is not None
    assert not env.backend._process_group_alive(trainer)
    assert not trainer.result_source_live.is_set()
    assert env.backend._result_reapers == set()
    assert env.backend._active_launch is None
    assert "DOWNLOAD_RETRY" not in caplog.text
    env.executor.system_panic.assert_not_called()


@pytest.mark.timeout(30)
@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process-group SIGKILL")
def test_real_trainer_death_interrupts_exact_accepted_result_download(external_process_env, monkeypatch, caplog):
    env = external_process_env
    result, trainer = _accepted_lazy_result(env)
    download = _start_paused_download(env, result, monkeypatch)

    started = time.monotonic()
    os.killpg(trainer.pgid, signal.SIGKILL)
    assert _wait_until(lambda: trainer.result_failure_notified, timeout=5.0)
    assert trainer.result_failure_delivery_done.wait(5.0)
    download.release.set()
    download.thread.join(timeout=5.0)
    elapsed = time.monotonic() - started

    assert download.done.is_set(), "source-loss notification did not release the exact receiver"
    assert download.errors, "materialization unexpectedly completed after its source died"
    assert elapsed < 10.0
    assert _wait_until(lambda: env.executor.system_panic.call_count == 1, timeout=5.0)
    assert env.run_abort_signal.triggered
    assert trainer.process.poll() == -signal.SIGKILL
    assert not env.backend._process_group_alive(trainer)
    assert not trainer.result_source_live.is_set()
    assert "DOWNLOAD_RETRY" not in caplog.text
    assert (env.run_dir / FLMetaKey.PROCESS_RC_FILE).read_text() == str(ProcessExitCode.EXCEPTION)

    env.backend.finalize(env.fl_ctx)
