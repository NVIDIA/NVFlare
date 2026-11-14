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

"""CP-owned launch, callback and settings isolation without external runtimes."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_launcher_spec import JOB_LAUNCH_OPTIONS, JobHandleSpec, JobLaunchOptions, publish_job_handle
from nvflare.private.fed.client.task_runtime import (
    TaskExecutionBinding,
    _launch_job_with_task_workers,
    _TaskExecutionState,
    shutdown_task_execution,
)


def runtime_fixture(job_id="job-1"):
    owner = SimpleNamespace(attach_cj=Mock(), release=Mock(), finish_cj_launch=Mock(), thread=Mock())
    owner.thread.is_alive.return_value = False
    service = SimpleNamespace(add_job=Mock(return_value=owner), shutdown=Mock())
    service_class = Mock(get_or_create=Mock(return_value=service))
    cell = SimpleNamespace(get_fqcn=lambda: "site-1")
    context = FLContext()
    context.set_prop(ReservedKey.IDENTITY_NAME, "site-1")
    engine = SimpleNamespace(get_cell=lambda: cell, _task_execution_state=_TaskExecutionState())
    context.set_prop(ReservedKey.ENGINE, engine)
    options = JobLaunchOptions(environment={"WORKER_KIND": job_id}, apply_study_runtime=False)
    runtime = TaskExecutionBinding(object(), service_class, "TASK_OWNER", "TASK_TOKEN", {"job_id": job_id}, options, 5)
    return SimpleNamespace(**locals())


@pytest.mark.parametrize("failure", ["none", "create", "publish", "after-publish"])
def test_launch_restores_parent_settings_and_cleans_acquired_runtime(failure):
    p = runtime_fixture()
    previous = JobLaunchOptions(environment={"PARENT_ONLY": "secret"})
    p.context.set_prop(JOB_LAUNCH_OPTIONS, previous, private=True, sticky=False)
    callback = Mock(side_effect=RuntimeError("publish") if failure == "publish" else None)
    p.context.set_prop(FLContextKey.JOB_HANDLE_CALLBACK, callback, private=True, sticky=False)
    original_options = p.context.get_prop_detail(JOB_LAUNCH_OPTIONS)
    original_callback = p.context.get_prop_detail(FLContextKey.JOB_HANDLE_CALLBACK)
    native = Mock(spec=JobHandleSpec)

    def create(meta, context):
        options = context.get_prop(JOB_LAUNCH_OPTIONS)
        assert options.environment["TASK_OWNER"] == "site-1"
        assert len(options.environment["TASK_TOKEN"]) == 64
        assert "PARENT_ONLY" not in options.environment
        if failure == "create":
            raise RuntimeError("create")
        publish_job_handle(native, context)
        if failure == "after-publish":
            raise RuntimeError("after-publish")
        return native

    launcher = Mock(launch_job=create)
    if failure == "none":
        handle = _launch_job_with_task_workers(p.runtime, launcher, p.context)
        assert callback.call_args.args == (handle,)
        assert handle.handle is native
        assert not handle.can_restart()
        p.owner.release.assert_not_called()
        native.terminate.assert_not_called()
    else:
        with pytest.raises(RuntimeError, match=failure):
            _launch_job_with_task_workers(p.runtime, launcher, p.context)
        p.owner.release.assert_called()
        if failure != "create":
            native.terminate.assert_called_once()
            native.wait.assert_not_called()
            p.owner.thread.join.assert_called_with(5)
    assert p.context.get_prop_detail(JOB_LAUNCH_OPTIONS) == original_options
    assert p.context.get_prop_detail(FLContextKey.JOB_HANDLE_CALLBACK) == original_callback
    assert "TASK_TOKEN" not in p.options.environment
    shutdown_task_execution(p.engine)
    p.service.shutdown.assert_called_once_with(timeout=5)


def test_early_abort_uses_owned_handle_before_launcher_waits():
    p = runtime_fixture()
    native = Mock(spec=JobHandleSpec)
    p.context.set_prop(FLContextKey.JOB_HANDLE_CALLBACK, lambda handle: handle.terminate(), sticky=False)

    def create(meta, context):
        publish_job_handle(native, context)
        p.owner.release.assert_called_once()
        native.terminate.assert_called_once()
        return native

    handle = _launch_job_with_task_workers(p.runtime, Mock(launch_job=create), p.context)
    assert handle.handle is native
    p.owner.attach_cj.assert_called_once_with(native)


def test_two_jobs_share_launcher_without_sharing_capabilities_or_settings():
    cases = [runtime_fixture("job-1"), runtime_fixture("job-2")]
    barrier = threading.Barrier(2)

    def create(meta, context):
        options = context.get_prop(JOB_LAUNCH_OPTIONS)
        environment = dict(options.environment)
        barrier.wait(timeout=5)
        assert context.get_prop(JOB_LAUNCH_OPTIONS).environment == environment
        assert environment["WORKER_KIND"] == meta["job_id"]
        native = Mock(spec=JobHandleSpec)
        publish_job_handle(native, context)
        return native

    launcher = Mock(launch_job=create)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_launch_job_with_task_workers, p.runtime, launcher, p.context) for p in cases]
        handles = [f.result(timeout=10) for f in futures]
    tokens = [p.service.add_job.call_args.args[-1] for p in cases]
    assert len(set(tokens)) == 2
    for p, handle in zip(cases, handles):
        assert p.context.get_prop(JOB_LAUNCH_OPTIONS) is None
        assert p.context.get_prop(FLContextKey.JOB_HANDLE_CALLBACK) is None
        handle.terminate()
        native = handle.handle
        native.terminate.assert_called_once()
        shutdown_task_execution(p.engine)


def test_shutdown_fences_a_later_launch_before_acquiring_owner_or_coordinator():
    p = runtime_fixture()
    shutdown_task_execution(p.engine)
    launcher = Mock()
    with pytest.raises(RuntimeError, match="shutting down"):
        _launch_job_with_task_workers(p.runtime, launcher, p.context)
    p.service.add_job.assert_not_called()
    launcher.launch_job.assert_not_called()


def test_shutdown_uses_site_timeout_and_continues_after_a_service_failure(caplog):
    p = runtime_fixture()
    _launch_job_with_task_workers(p.runtime, Mock(launch_job=Mock(spec=JobHandleSpec)), p.context)
    p.service.shutdown.side_effect = RuntimeError("cleanup unavailable")
    second = Mock()
    p.engine._task_execution_state.services[object()] = (second, 7.5)
    shutdown_task_execution(p.engine)
    p.service.shutdown.assert_called_once_with(timeout=5)
    second.shutdown.assert_called_once_with(timeout=7.5)
    assert "continuing CP shutdown" in caplog.text


def test_system_end_still_reaches_components_if_runtime_shutdown_raises(monkeypatch):
    from nvflare.apis.event_type import EventType
    from nvflare.private.fed.client import client_engine as engine_module

    engine = object.__new__(engine_module.ClientEngine)
    engine.cell = object()
    engine.logger = Mock()
    engine.fl_components = [Mock()]
    dispatch = Mock()
    monkeypatch.setattr(engine_module, "shutdown_task_execution", Mock(side_effect=RuntimeError("cleanup unavailable")))
    monkeypatch.setattr(engine_module, "fire_event", dispatch)
    context = FLContext()
    engine.fire_event(EventType.SYSTEM_END, context)
    dispatch.assert_called_once_with(event=EventType.SYSTEM_END, handlers=engine.fl_components, ctx=context)
    engine.logger.error.assert_called_once()


@pytest.mark.parametrize("late_creation", [False, True])
def test_docker_stop_failure_retains_owner_until_confirmed_exit(late_creation):
    from docker.errors import APIError

    from nvflare.apis.job_launcher_spec import JobReturnCode
    from nvflare.app_opt.job_launcher.docker_launcher import DockerJobHandle
    from nvflare.app_opt.job_launcher.slurm.task_service import SlurmTaskService
    from tests.timing_utils import ThreadGroup, join_thread

    p = runtime_fixture()
    p.runtime.service_class = SlurmTaskService
    p.runtime.backend = SimpleNamespace(poll_interval=0.01)
    p.runtime.cleanup_timeout = 0.03
    p.cell.register_request_cb = Mock()
    client = Mock()
    container = client.containers.get.return_value
    container.status = "running"
    container.stop.side_effect = APIError("daemon temporarily unavailable")
    native = DockerJobHandle("container-1", "coordinator", client, watch_exit=False)
    native.wait = Mock(side_effect=AssertionError("unbounded wait must not run during launch-failure cleanup"))
    entered, finish = threading.Event(), threading.Event()

    def create(meta, context):
        entered.set()
        if late_creation:
            finish.wait()
        publish_job_handle(native, context)
        return native

    def launch_late():
        with pytest.raises(RuntimeError, match="shut down during"):
            _launch_job_with_task_workers(p.runtime, Mock(launch_job=create), p.context)

    with ThreadGroup(finish) as threads:
        try:
            if late_creation:
                worker = threads.start(launch_late)
                assert entered.wait(5)
            else:
                _launch_job_with_task_workers(p.runtime, Mock(launch_job=create), p.context)
            service = p.cell._slurm_task_service
            owner = service.jobs["job-1"]
            shutdown_task_execution(p.engine)
            finish.set()
            if late_creation:
                join_thread(worker)
            assert owner.thread.is_alive() and service.jobs["job-1"] is owner
            assert owner.cj_handle is native
            assert native.poll() == JobReturnCode.UNKNOWN
            native.wait.assert_not_called()
            container.stop.assert_called()
            # A later successful retry is needed before ownership can disappear.
            container.stop.side_effect = None
            owner.wake.set()
            join_thread(owner.thread)
            assert native.poll() == JobReturnCode.ABORTED
            assert not service.jobs
        finally:
            container.stop.side_effect = None
            finish.set()
            service = getattr(p.cell, "_slurm_task_service", None)
            if service:
                assert service.shutdown(timeout=5) == ()


@pytest.mark.parametrize("late_creation", [False, True])
def test_exited_process_leader_does_not_release_owned_children(monkeypatch, late_creation):
    import signal

    from nvflare.apis.job_launcher_spec import JobReturnCode
    from nvflare.app_common.job_launcher import process_launcher as process_module
    from nvflare.app_common.job_launcher.process_launcher import ProcessHandle
    from nvflare.app_opt.job_launcher.slurm.task_service import SlurmTaskService
    from tests.timing_utils import ThreadGroup, join_thread

    p = runtime_fixture()
    p.runtime.service_class = SlurmTaskService
    p.runtime.backend = SimpleNamespace(poll_interval=0.01)
    p.runtime.cleanup_timeout = 0.03
    p.cell.register_request_cb = Mock()
    adapter = Mock(pid=12345)
    adapter.poll.return_value = 0  # The coordinator exited; its group still has a child.
    native = ProcessHandle(process_adapter=adapter, owns_process_group=True)
    entered, finish, group_gone, kill_requested = (threading.Event() for _ in range(4))

    def killpg(pid, sig):
        assert pid == adapter.pid
        if group_gone.is_set():
            raise ProcessLookupError()
        if sig == signal.SIGKILL:
            kill_requested.set()

    monkeypatch.setattr(process_module.os, "killpg", killpg)

    def create(meta, context):
        entered.set()
        if late_creation:
            finish.wait()
        publish_job_handle(native, context)
        return native

    def launch_late():
        with pytest.raises(RuntimeError, match="shut down during"):
            _launch_job_with_task_workers(p.runtime, Mock(launch_job=create), p.context)

    with ThreadGroup(finish) as threads:
        try:
            if late_creation:
                worker = threads.start(launch_late)
                assert entered.wait(5)
            else:
                _launch_job_with_task_workers(p.runtime, Mock(launch_job=create), p.context)
            service = p.cell._slurm_task_service
            owner = service.jobs["job-1"]
            shutdown_task_execution(p.engine)
            finish.set()
            if late_creation:
                join_thread(worker)
            assert kill_requested.wait(5), "cleanup skipped the exited coordinator's child group"
            assert native.poll() == JobReturnCode.SUCCESS
            assert not native.is_cleanup_complete()
            assert owner.thread.is_alive() and service.jobs["job-1"] is owner
            # Even after SIGKILL, CP retains ownership until group absence is observed.
            group_gone.set()
            owner.wake.set()
            join_thread(owner.thread)
            assert native.is_cleanup_complete() and not service.jobs
            adapter.wait.assert_not_called()
        finally:
            group_gone.set()
            finish.set()
            service = getattr(p.cell, "_slurm_task_service", None)
            if service:
                assert service.shutdown(timeout=5) == ()


def test_wrapped_cleanup_requires_physical_execution_and_owned_workers_to_settle():
    from nvflare.private.fed.client.task_runtime import JobHandleWithTaskWorkers

    p = runtime_fixture()
    native = Mock(spec=JobHandleSpec)
    native.is_cleanup_complete.return_value = False
    wrapped = JobHandleWithTaskWorkers(native, p.owner)
    assert not wrapped.is_cleanup_complete()
    p.owner.release.assert_not_called()
    native.is_cleanup_complete.return_value = True
    p.owner.thread.is_alive.return_value = True
    assert not wrapped.is_cleanup_complete()
    p.owner.release.assert_called_once()
    p.owner.thread.is_alive.return_value = False
    assert wrapped.is_cleanup_complete()
