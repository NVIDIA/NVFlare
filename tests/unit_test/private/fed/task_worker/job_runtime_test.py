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

"""Parent-owned launch, callback and settings isolation without external runtimes."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_launcher_spec import JOB_LAUNCH_OPTIONS, JobHandleSpec, JobLaunchOptions, publish_job_handle
from nvflare.private.fed.task_worker.job_runtime import TaskJobRuntime, shutdown_task_jobs


def runtime_fixture(job_id="job-1"):
    owner = SimpleNamespace(attach_cj=Mock(), release=Mock(), thread=Mock())
    owner.thread.is_alive.return_value = False
    service = SimpleNamespace(add_job=Mock(return_value=owner), shutdown=Mock())
    service_class = Mock(get_or_create=Mock(return_value=service))
    cell = SimpleNamespace(get_fqcn=lambda: "site-1")
    context = FLContext()
    context.set_prop(ReservedKey.IDENTITY_NAME, "site-1")
    context.set_prop(ReservedKey.ENGINE, SimpleNamespace(get_cell=lambda: cell))
    options = JobLaunchOptions(environment={"WORKER_KIND": job_id}, apply_study_runtime=False)
    runtime = TaskJobRuntime(object(), service_class, "TASK_OWNER", "TASK_TOKEN", {"job_id": job_id}, options, 5)
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
        handle = p.runtime.launch(launcher, p.context)
        assert callback.call_args.args == (handle,)
        assert handle.handle is native
        assert not handle.can_restart()
        p.owner.release.assert_not_called()
        native.terminate.assert_not_called()
    else:
        with pytest.raises(RuntimeError, match=failure):
            p.runtime.launch(launcher, p.context)
        p.owner.release.assert_called()
        if failure != "create":
            native.terminate.assert_called_once()
            native.wait.assert_called_once()
    assert p.context.get_prop_detail(JOB_LAUNCH_OPTIONS) == original_options
    assert p.context.get_prop_detail(FLContextKey.JOB_HANDLE_CALLBACK) == original_callback
    assert "TASK_TOKEN" not in p.options.environment
    shutdown_task_jobs(p.cell)
    p.service.shutdown.assert_called_once_with(timeout=30.0)


def test_early_abort_uses_owned_handle_before_launcher_waits():
    p = runtime_fixture()
    native = Mock(spec=JobHandleSpec)
    p.context.set_prop(FLContextKey.JOB_HANDLE_CALLBACK, lambda handle: handle.terminate(), sticky=False)

    def create(meta, context):
        publish_job_handle(native, context)
        p.owner.release.assert_called_once()
        native.terminate.assert_called_once()
        return native

    handle = p.runtime.launch(Mock(launch_job=create), p.context)
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
        futures = [pool.submit(p.runtime.launch, launcher, p.context) for p in cases]
        handles = [f.result(timeout=10) for f in futures]
    tokens = [p.service.add_job.call_args.args[-1] for p in cases]
    assert len(set(tokens)) == 2
    for p, handle in zip(cases, handles):
        assert p.context.get_prop(JOB_LAUNCH_OPTIONS) is None
        assert p.context.get_prop(FLContextKey.JOB_HANDLE_CALLBACK) is None
        handle.terminate()
        native = handle.handle
        native.terminate.assert_called_once()
        shutdown_task_jobs(p.cell)


def test_shutdown_fences_a_later_launch_before_acquiring_owner_or_coordinator():
    p = runtime_fixture()
    shutdown_task_jobs(p.cell)
    launcher = Mock()
    with pytest.raises(RuntimeError, match="shutting down"):
        p.runtime.launch(launcher, p.context)
    p.service.add_job.assert_not_called()
    launcher.launch_job.assert_not_called()
