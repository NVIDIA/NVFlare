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


"""Shared setup for policy unit tests and real process integration scenarios."""

import json
import time

import psutil

from nvflare.apis.job_launcher_spec import JobProcessEnv
from nvflare.private.fed.task_worker import TaskAttemptIdentity, WorkerBootstrap, write_bootstrap

_PROBE_MODULE = """
import json
import os

from nvflare.apis.executor import Executor
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.statistics_spec import Statistics


class ProbeExecutor(Executor):
    def __init__(self, marker_path):
        super().__init__()
        self.marker_path = marker_path

    def initialize(self, fl_ctx):
        with open(self.marker_path, "a") as stream:
            stream.write("initialize\\n")

    def finalize(self, fl_ctx):
        with open(self.marker_path, "a") as stream:
            stream.write("finalize\\n")

    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        try:
            fl_ctx.get_engine().get_cell()
        except RuntimeError as e:
            unsupported = str(e)
        else:
            unsupported = "missing explicit failure"
        return Shareable({
            "value": shareable["value"] + 4,
            "pid": os.getpid(),
            "site": fl_ctx.get_identity_name(),
            "job_id": fl_ctx.get_job_id(),
            "task_id": fl_ctx.get_prop("__task_id__"),
            "context": fl_ctx.get_prop("explicit_value"),
            "unsupported": unsupported,
            "credentials": [name for name in %r if name in os.environ],
        })


class RaiseExecutor(ProbeExecutor):
    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        raise ValueError("deliberate worker failure")


class FailLargeValueExecutor(ProbeExecutor):
    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        if shareable["value"] > 5:
            raise ValueError("deliberate worker failure")
        return super().execute(task_name, shareable, fl_ctx, abort_signal)


class FailingFinalizeStatistics(Statistics):
    def features(self):
        return {}

    def count(self, dataset_name, feature_name):
        return 8

    def finalize(self, fl_ctx):
        raise ValueError("statistics cleanup failed")


class ReferenceExecutor(Executor):
    def __init__(self, value, options):
        super().__init__()
        self.value = value
        self.options = options

    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        return Shareable({"value": self.value + shareable["value"], "options": self.options})
""" % (
    JobProcessEnv.ALL,
)


def _workspace(tmp_path):
    root = tmp_path / "workspace"
    (root / "startup").mkdir(parents=True)
    (root / "local").mkdir()
    app_root = root / "job-1" / "app_site-1"
    custom_dir = app_root / "custom"
    custom_dir.mkdir(parents=True)
    (app_root / "config").mkdir()
    (custom_dir / "worker_components.py").write_text(_PROBE_MODULE)
    (root / "job-1" / "meta.json").write_text(json.dumps({"byoc": True}))
    return root


def _identity(attempt_id, task_id="task-1", task_name="train"):
    return TaskAttemptIdentity(
        job_id="job-1",
        site_name="site-1",
        task_id=task_id,
        task_name=task_name,
        attempt_id=attempt_id,
    )


def _stage(store, identity, workspace_root, executor, data, components=(), context_properties=None, session_dir=None):
    if not executor:
        executor = {"path": "nvflare.app_common.np.np_trainer.NPTrainer"}
    store.create_attempt(identity)
    store.write_input(identity, data)
    bootstrap = WorkerBootstrap(
        identity=identity,
        artifact_root=store.root_dir,
        workspace_root=str(workspace_root),
        executor=executor,
        components=components,
        context_properties=context_properties or {},
        max_payload_bytes=store.max_payload_bytes,
        session_dir=session_dir,
    )
    path = store.bootstrap_path(identity)
    write_bootstrap(path, bootstrap)
    return path


def _wait_until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("bounded process probe timed out")
        time.sleep(0.05)


def _live(process):
    try:
        return process.is_running() and process.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
    except psutil.NoSuchProcess:
        return False
