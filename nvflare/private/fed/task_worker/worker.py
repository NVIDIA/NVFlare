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

"""Execute one staged assignment without a federation Cell or live CJ engine."""

import argparse
import copy
import os
import resource
import sys
import time
from typing import Mapping

from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import EventScope, FLContextKey
from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.apis.job_launcher_spec import pop_credential_env
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.workspace import Workspace
from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor
from nvflare.app_common.executors.multi_process_executor import WorkerComponentBuilder
from nvflare.private.event import fire_event
from nvflare.private.fed.utils.fed_utils import fobs_initialize, get_job_meta_from_workspace
from nvflare.security.logging import secure_format_exception

from .artifacts import FileTaskArtifactStore, TaskCompletion
from .client_api import ClientAPITaskResult, execute_client_api_task
from .protocol import WorkerBootstrap, read_bootstrap

_PROCESS_TYPE = "client_task_worker"
_PROTECTED_CONTEXT_KEYS = {
    FLContextKey.APP_ROOT,
    FLContextKey.CLIENT_NAME,
    FLContextKey.CURRENT_JOB_ID,
    FLContextKey.CURRENT_RUN,
    FLContextKey.JOB_META,
    FLContextKey.PROCESS_TYPE,
    FLContextKey.TASK_DATA,
    FLContextKey.TASK_ID,
    FLContextKey.TASK_NAME,
    FLContextKey.TASK_RESULT,
    FLContextKey.WORKSPACE_OBJECT,
    FLContextKey.WORKSPACE_ROOT,
}


class UnsupportedTaskWorkerService(RuntimeError):
    """An Executor requested a service that this worker runtime cannot provide."""


class TaskWorkerEngine:
    """Small task-local Engine surface for ordinary compute Executors.

    This intentionally has no Cell, federation messaging, streaming, controller,
    or task-acquisition service. Applications that need one receive an explicit
    error and are not yet supported by the task execution lifetime.
    """

    def __init__(self, workspace: Workspace, components: Mapping[str, object]):
        self._workspace = workspace
        self._components = dict(components)
        self._handlers = []
        self._context_manager = None

    def get_component(self, component_id: str):
        return self._components.get(component_id)

    def get_all_components(self) -> dict:
        return dict(self._components)

    def get_workspace(self) -> Workspace:
        return self._workspace

    def new_context(self) -> FLContext:
        if self._context_manager is None:
            raise RuntimeError("task worker context manager is not initialized")
        return self._context_manager.new_context()

    def fire_event(self, event_type: str, fl_ctx: FLContext):
        if fl_ctx.get_prop(FLContextKey.EVENT_SCOPE) == EventScope.FEDERATION:
            self._unsupported("federated events")
        fire_event(event=event_type, handlers=self._handlers, ctx=fl_ctx)

    @staticmethod
    def _unsupported(service: str):
        raise UnsupportedTaskWorkerService(f"task worker runtime does not provide {service}")

    def get_cell(self):
        self._unsupported("a federation Cell")

    def register_aux_message_handler(self, *args, **kwargs):
        self._unsupported("auxiliary message handlers")

    def send_aux_request(self, *args, **kwargs):
        self._unsupported("auxiliary federation requests")

    def multicast_aux_requests(self, *args, **kwargs):
        self._unsupported("multicast auxiliary federation requests")

    def fire_and_forget_aux_request(self, *args, **kwargs):
        self._unsupported("asynchronous auxiliary federation requests")

    def dispatch(self, *args, **kwargs):
        self._unsupported("auxiliary dispatch")

    def stream_objects(self, *args, **kwargs):
        self._unsupported("federation object streaming")

    def get_task_assignment(self, *args, **kwargs):
        self._unsupported("task acquisition")

    def send_task_result(self, *args, **kwargs):
        self._unsupported("remote result publication")

    def validate_targets(self, *args, **kwargs):
        self._unsupported("federation target validation")

    def get_widget(self, *args, **kwargs):
        self._unsupported("resident job widgets")

    def build_component(self, *args, **kwargs):
        self._unsupported("runtime component construction")

    def abort_app(self, *args, **kwargs):
        self._unsupported("resident application control")


def _read_job_meta(workspace: Workspace, job_id: str) -> dict:
    try:
        meta = get_job_meta_from_workspace(workspace, job_id)
    except FileNotFoundError:
        return {}
    if not isinstance(meta, dict):
        raise RuntimeError(f"job metadata must be a dict but got {type(meta)}")
    return meta


def _new_context(bootstrap: WorkerBootstrap, workspace: Workspace):
    identity = bootstrap.identity
    app_root = workspace.get_app_dir(identity.job_id)
    components = {}
    engine = TaskWorkerEngine(workspace, components)
    manager = FLContextManager(engine=engine, identity_name=identity.site_name, job_id=identity.job_id)
    engine._context_manager = manager
    fl_ctx = manager.new_context()
    built_ins = {
        FLContextKey.APP_ROOT: app_root,
        FLContextKey.CLIENT_NAME: identity.site_name,
        FLContextKey.CURRENT_JOB_ID: identity.job_id,
        FLContextKey.JOB_META: _read_job_meta(workspace, identity.job_id),
        FLContextKey.PROCESS_TYPE: _PROCESS_TYPE,
        FLContextKey.TASK_ID: identity.task_id,
        FLContextKey.TASK_NAME: identity.task_name,
        FLContextKey.WORKSPACE_OBJECT: workspace,
        FLContextKey.WORKSPACE_ROOT: workspace.get_root_dir(),
    }
    for name, value in built_ins.items():
        fl_ctx.set_prop(name, value, private=True, sticky=name not in (FLContextKey.TASK_ID, FLContextKey.TASK_NAME))
    for name, prop in bootstrap.context_properties.items():
        if name in _PROTECTED_CONTEXT_KEYS:
            raise ValueError(f"worker bootstrap cannot override framework context property {name!r}")
        fl_ctx.set_prop(name, prop.value, private=prop.private, sticky=prop.sticky)
    return engine, fl_ctx


def _build_compute_graph(bootstrap: WorkerBootstrap, workspace: Workspace, fl_ctx: FLContext, engine):
    builder = WorkerComponentBuilder(fl_ctx=fl_ctx, workspace=workspace)
    components = {}
    for index, original in enumerate(bootstrap.components):
        config = copy.deepcopy(dict(original))
        component_id = config.get("id")
        node = builder.make_component_node(config, index + 1)
        components[component_id] = builder.build_component(config, node)

    executor_config = copy.deepcopy(dict(bootstrap.executor))
    executor = builder.build_component(executor_config, builder.make_component_node(executor_config))
    if not isinstance(executor, Executor):
        raise TypeError(f"worker executor config built {type(executor)} instead of Executor")
    engine._components = components
    # START_RUN/END_RUN are attempt-scoped initialization adapters here. Only
    # the explicitly selected compute graph receives them; job-scoped CJ
    # handlers are neither copied nor replayed.
    engine._handlers = [component for component in components.values() if isinstance(component, FLComponent)]
    if not isinstance(executor, ClientAPIExecutor):
        engine._handlers.append(executor)
    return executor


def _raise_event_errors(fl_ctx: FLContext, event_type: str):
    exceptions = fl_ctx.get_prop(FLContextKey.EXCEPTIONS)
    if not exceptions:
        return
    fl_ctx.remove_prop(FLContextKey.EXCEPTIONS, force_removal=True)
    names = ", ".join(sorted(str(name) for name in exceptions))
    first = next(iter(exceptions.values()))
    raise RuntimeError(
        f"task worker {event_type} handler failed ({names}): {secure_format_exception(first)}"
    ) from first


def _fire_checked(engine: TaskWorkerEngine, event_type: str, fl_ctx: FLContext):
    fl_ctx.remove_prop(FLContextKey.EXCEPTIONS, force_removal=True)
    engine.fire_event(event_type, fl_ctx)
    _raise_event_errors(fl_ctx, event_type)


def _restore_peer_context(data: Shareable, fl_ctx: FLContext):
    peer_props = data.get_peer_props()
    if isinstance(peer_props, dict):
        peer_ctx = FLContext()
        peer_ctx.set_public_props(peer_props)
        fl_ctx.set_peer_context(peer_ctx)


def _execute(bootstrap: WorkerBootstrap, data: Shareable, workspace: Workspace, store: FileTaskArtifactStore):
    engine, fl_ctx = _new_context(bootstrap, workspace)
    executor = _build_compute_graph(bootstrap, workspace, fl_ctx, engine)
    abort_signal = Signal()
    _restore_peer_context(data, fl_ctx)

    started = False
    try:
        started = True
        _fire_checked(engine, EventType.START_RUN, fl_ctx)
        fl_ctx.set_prop(FLContextKey.TASK_DATA, data, private=True, sticky=False)
        _fire_checked(engine, EventType.BEFORE_TASK_EXECUTION, fl_ctx)
        if isinstance(executor, ClientAPIExecutor):
            result = execute_client_api_task(executor, data, fl_ctx, store, bootstrap.identity)
            return result
        result = executor.execute(bootstrap.identity.task_name, data, fl_ctx, abort_signal)
        if not isinstance(result, Shareable):
            raise TypeError(f"Executor returned {type(result)} instead of Shareable")
        fl_ctx.set_prop(FLContextKey.TASK_RESULT, result, private=True, sticky=False)
        _fire_checked(engine, EventType.AFTER_TASK_EXECUTION, fl_ctx)
        return result
    finally:
        if started:
            _fire_checked(engine, EventType.END_RUN, fl_ctx)


def run_worker(bootstrap_path: str) -> TaskCompletion:
    """Run one bootstrap and commit its result after successful finalization."""

    # Task workers have no federation authority. Strip every CJ bootstrap
    # credential before importing job custom code, even if a launcher was
    # accidentally given a broader environment than its explicit request.
    pop_credential_env()
    started_at = time.time()
    bootstrap = read_bootstrap(bootstrap_path)
    identity = bootstrap.identity
    store = FileTaskArtifactStore(bootstrap.artifact_root)
    workspace = Workspace(bootstrap.workspace_root, site_name=identity.site_name)
    old_sys_path = sys.path.copy()
    try:
        custom_dir = workspace.get_app_custom_dir(identity.job_id)
        if os.path.isdir(custom_dir):
            sys.path.insert(0, custom_dir)
        fobs_initialize(workspace=workspace, job_id=identity.job_id)
        data = store.read_input(identity)
        result = _execute(bootstrap, data, workspace, store)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        completed_at = time.time()
        diagnostics = {
            "user_cpu_seconds": usage.ru_utime,
            "system_cpu_seconds": usage.ru_stime,
            "max_rss_native_units": usage.ru_maxrss,
            "max_rss_unit": "bytes" if sys.platform == "darwin" else "kibibytes",
        }
        if isinstance(result, ClientAPITaskResult):
            if result.analytics:
                diagnostics["analytics"] = store.write_analytics(identity, result.analytics).to_dict()
            commit = store.commit_staged_result
            payload = result.reference
        else:
            commit = store.commit_result
            payload = result
        return commit(
            identity,
            payload,
            worker_pid=os.getpid(),
            worker_ppid=os.getppid(),
            started_at=started_at,
            completed_at=completed_at,
            diagnostics=diagnostics,
        )
    except BaseException as e:
        try:
            store.record_failure(
                identity,
                worker_pid=os.getpid(),
                worker_ppid=os.getppid(),
                started_at=started_at,
                failed_at=time.time(),
                error_type=type(e).__name__,
                message=secure_format_exception(e),
            )
        except Exception:
            pass
        raise
    finally:
        sys.path[:] = old_sys_path


def main():
    parser = argparse.ArgumentParser(description="Execute one staged NVFlare task assignment")
    parser.add_argument("--bootstrap", required=True, help="absolute path to the worker bootstrap JSON")
    args = parser.parse_args()
    run_worker(args.bootstrap)


if __name__ == "__main__":
    main()
