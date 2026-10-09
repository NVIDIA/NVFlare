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

"""Execute one staged assignment without a federation Cell or live job engine."""

import logging
import os
import resource
import sys
import time

from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import EventScope, FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_launcher_spec import pop_credential_env
from nvflare.apis.shareable import Shareable
from nvflare.apis.workspace import Workspace
from nvflare.fuel.sec.audit import AuditService
from nvflare.fuel.utils.log_utils import configure_logging
from nvflare.private.fed.utils.fed_utils import fobs_initialize, get_job_meta_from_workspace
from nvflare.private.fed.utils.worker_component_builder import WorkerComponentBuilder
from nvflare.security.logging import secure_format_exception
from nvflare.utils.job_launcher_utils import refresh_custom_dir_import_path

from .artifacts import FileTaskArtifactStore, TaskCompletion
from .protocol import WorkerBootstrap, _thaw_json, read_bootstrap
from .runtime import TaskRuntime

_PROTECTED_CONTEXT_KEYS = {
    ReservedKey.ENGINE,
    ReservedKey.MANAGER,
    ReservedKey.IDENTITY_NAME,
    ReservedKey.PEER_CTX,
    FLContextKey.APP_ROOT,
    FLContextKey.CLIENT_NAME,
    FLContextKey.CURRENT_JOB_ID,
    FLContextKey.CURRENT_RUN,
    FLContextKey.JOB_META,
    FLContextKey.PROCESS_TYPE,
    FLContextKey.RUN_ABORT_SIGNAL,
    FLContextKey.TASK_DATA,
    FLContextKey.TASK_ID,
    FLContextKey.TASK_ATTEMPT_ID,
    FLContextKey.TASK_ATTEMPT_REQUIRED,
    FLContextKey.TASK_NAME,
    FLContextKey.TASK_RESULT,
    FLContextKey.WORKSPACE_OBJECT,
    FLContextKey.WORKSPACE_ROOT,
}


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
    runtime = TaskRuntime(workspace, identity.site_name, identity.job_id)
    fl_ctx = runtime.new_context()
    built_ins = {
        FLContextKey.APP_ROOT: app_root,
        FLContextKey.CURRENT_JOB_ID: identity.job_id,
        FLContextKey.JOB_META: _read_job_meta(workspace, identity.job_id),
        FLContextKey.TASK_ID: identity.task_id,
        FLContextKey.TASK_ATTEMPT_ID: identity.attempt_id,
        FLContextKey.TASK_ATTEMPT_REQUIRED: True,
        FLContextKey.TASK_NAME: identity.task_name,
        FLContextKey.WORKSPACE_OBJECT: workspace,
        FLContextKey.WORKSPACE_ROOT: workspace.get_root_dir(),
    }
    for name, value in built_ins.items():
        fl_ctx.set_prop(
            name,
            value,
            private=True,
            sticky=name not in (FLContextKey.TASK_ID, FLContextKey.TASK_NAME, FLContextKey.TASK_ATTEMPT_ID),
        )
    protected = _PROTECTED_CONTEXT_KEYS
    for name, prop in bootstrap.context_properties.items():
        if name in protected:
            raise ValueError(f"worker bootstrap cannot override framework context property {name!r}")
        fl_ctx.set_prop(name, _thaw_json(prop.value), private=prop.private, sticky=prop.sticky)
    return runtime, fl_ctx


def _authorize_compute_graph(bootstrap, workspace, fl_ctx):
    builder = WorkerComponentBuilder(fl_ctx=fl_ctx, workspace=workspace)
    config = bootstrap.to_dict()
    for index, component in enumerate(config["components"]):
        builder.authorize_tree(component, builder.make_component_node(component, index + 1))
    builder.authorize_tree(config["executor"])
    return builder


def _build_compute_graph(bootstrap: WorkerBootstrap, workspace: Workspace, fl_ctx: FLContext, runtime: TaskRuntime):
    builder = _authorize_compute_graph(bootstrap, workspace, fl_ctx)
    components = {}
    for index, original in enumerate(bootstrap.components):
        config = _thaw_json(original)
        component_id = config.get("id")
        node = builder.make_component_node(config, index + 1)
        components[component_id] = builder.build_component(config, node)

    executor_config = _thaw_json(bootstrap.executor)
    executor = builder.build_component(executor_config, builder.make_component_node(executor_config))
    if not isinstance(executor, Executor):
        raise TypeError(f"worker executor config built {type(executor)} instead of Executor")
    # START_RUN/END_RUN are attempt-scoped initialization adapters here. Only
    # the explicitly selected compute graph receives them; job-scoped
    # handlers are neither copied nor replayed.
    runtime.set_compute_graph(components, executor)
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


def _fire_checked(runtime: TaskRuntime, event_type: str, fl_ctx: FLContext):
    fl_ctx.remove_prop(FLContextKey.EXCEPTIONS, force_removal=True)
    fl_ctx.remove_prop(FLContextKey.EVENT_DATA, force_removal=True)
    fl_ctx.set_prop(FLContextKey.EVENT_SCOPE, EventScope.LOCAL, private=True, sticky=False)
    runtime.fire_event(event_type, fl_ctx)
    _raise_event_errors(fl_ctx, event_type)
    runtime.raise_if_failed()


def _restore_peer_context(data: Shareable, fl_ctx: FLContext):
    peer_props = data.get_peer_props()
    if isinstance(peer_props, dict):
        peer_ctx = FLContext()
        peer_ctx.set_public_props(peer_props)
        fl_ctx.set_peer_context(peer_ctx)


def _execute(
    bootstrap: WorkerBootstrap,
    data: Shareable,
    workspace: Workspace,
    store: FileTaskArtifactStore,
):
    runtime, fl_ctx = _new_context(bootstrap, workspace)
    executor = _build_compute_graph(bootstrap, workspace, fl_ctx, runtime)
    _restore_peer_context(data, fl_ctx)

    started = False
    first_error = None
    try:
        started = True
        _fire_checked(runtime, EventType.ABOUT_TO_START_RUN, fl_ctx)
        _fire_checked(runtime, EventType.START_RUN, fl_ctx)
        fl_ctx.set_prop(FLContextKey.TASK_DATA, data, private=True, sticky=False)
        _fire_checked(runtime, EventType.BEFORE_TASK_EXECUTION, fl_ctx)
        result = executor.execute(bootstrap.identity.task_name, data, fl_ctx, runtime.abort_signal)
        runtime.raise_if_failed()
        if not isinstance(result, Shareable):
            raise TypeError(f"Executor returned {type(result)} instead of Shareable")
        fl_ctx.set_prop(FLContextKey.TASK_RESULT, result, private=True, sticky=False)
        _fire_checked(runtime, EventType.AFTER_TASK_EXECUTION, fl_ctx)
    except BaseException as e:
        first_error = e
    finally:
        if started:
            for event_type in (EventType.ABOUT_TO_END_RUN, EventType.END_RUN):
                try:
                    _fire_checked(runtime, event_type, fl_ctx)
                except BaseException as e:
                    # Finalizers must all run, but cannot replace the failure
                    # that caused cleanup, or an earlier finalizer failure.
                    if first_error is None:
                        first_error = e
    if first_error is not None:
        raise first_error
    result = fl_ctx.get_prop(FLContextKey.TASK_RESULT)
    if not isinstance(result, Shareable):
        raise TypeError("task hooks must leave a Shareable result")
    runtime.raise_if_failed()
    return result, runtime


def run_worker(bootstrap_path: str) -> TaskCompletion:
    """Run one bootstrap and commit its result after successful finalization."""

    # Strip CJ bootstrap credentials from the environment before importing job
    # custom code. This is not filesystem isolation: Process workers share the
    # site's UID and workspace, which can contain credential files.
    pop_credential_env()
    sys.argv[:] = ["nvflare-task-worker"]
    started_at = time.time()
    started_monotonic = time.monotonic()
    bootstrap = read_bootstrap(bootstrap_path)
    identity = bootstrap.identity
    store = FileTaskArtifactStore(bootstrap.artifact_root, max_payload_bytes=bootstrap.max_payload_bytes)
    workspace = Workspace(bootstrap.workspace_root, site_name=identity.site_name)
    old_sys_path = sys.path.copy()
    # Failure to acquire the claim must not modify an existing attempt's records.
    store.claim_worker(identity)
    owns_auditor = False
    try:
        # Record policy decisions even before application logging or imports.
        if AuditService.get_auditor() is None:
            AuditService.initialize(workspace.get_audit_file_path(identity.job_id))
            owns_auditor = True
        # Policy preflight precedes logging factories, custom decomposers and payload decoding.
        _runtime, fl_ctx = _new_context(bootstrap, workspace)
        _authorize_compute_graph(bootstrap, workspace, fl_ctx)
        if workspace.get_log_config_file_path():
            configure_logging(workspace, identity.job_id, file_prefix=f"task_{identity.attempt_id}")
        else:
            logging.basicConfig(level=logging.INFO)
        refresh_custom_dir_import_path(workspace.get_app_custom_dir(identity.job_id))
        fobs_initialize(workspace=workspace, job_id=identity.job_id)
        data = store.read_input(identity)
        result, runtime = _execute(bootstrap, data, workspace, store)
        reference = store.stage_result(identity, result)
        # Serialization can invoke application decomposers. Fail early here;
        # the publication guard makes the final decision after verification.
        runtime.raise_if_failed()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        completed_at = time.time()
        diagnostics = {
            "elapsed_seconds": time.monotonic() - started_monotonic,
            "user_cpu_seconds": usage.ru_utime,
            "system_cpu_seconds": usage.ru_stime,
            "max_rss_native_units": usage.ru_maxrss,
            "max_rss_unit": "bytes" if sys.platform == "darwin" else "kibibytes",
        }
        return store.commit_staged_result(
            identity,
            reference,
            worker_pid=os.getpid(),
            worker_ppid=os.getppid(),
            started_at=started_at,
            completed_at=completed_at,
            diagnostics=diagnostics,
            publication_guard=runtime.completion_guard,
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
        if owns_auditor:
            AuditService.close()


def main():
    from nvflare.private.fed.app.client.task_worker_process import main as process_main

    process_main()


if __name__ == "__main__":
    main()
