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

from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReservedKey
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


def _new_context(bootstrap: WorkerBootstrap, workspace: Workspace, protected_context_keys=()):
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
    protected = _PROTECTED_CONTEXT_KEYS | set(protected_context_keys)
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
    runtime.set_compute_graph(components, executor, instances=builder.components)
    return executor


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
    *,
    context_binding=None,
    protected_context_keys=(),
    analytics=None,
):
    runtime, fl_ctx = _new_context(bootstrap, workspace, protected_context_keys)
    adapt_executor = None
    if context_binding is not None:
        adapt_executor = context_binding(fl_ctx, bootstrap.identity, store, analytics)
    executor = _build_compute_graph(bootstrap, workspace, fl_ctx, runtime)
    if adapt_executor is not None:
        runner = adapt_executor(executor)
        runtime.set_compute_graph(
            runtime.get_all_components(),
            runner,
            instances=[runner if c is executor else c for c in runtime.compute_components],
        )
        executor = runner
    _restore_peer_context(data, fl_ctx)

    initialized = []
    try:
        for component in runtime.compute_components:
            # Include partially initialized objects in cleanup. Hooks must be
            # idempotent and tolerate partial setup.
            initialized.append(component)
            initialize = getattr(component, "initialize", None)
            if initialize is not None:
                with runtime.operation():
                    initialize(fl_ctx)
            runtime.raise_if_failed()
        fl_ctx.set_prop(FLContextKey.TASK_DATA, data, private=True, sticky=False)
        with runtime.operation():
            result = executor.execute(bootstrap.identity.task_name, data, fl_ctx, runtime.abort_signal)
        runtime.raise_if_failed()
        if not isinstance(result, Shareable):
            raise TypeError(f"Executor returned {type(result)} instead of Shareable")
        fl_ctx.set_prop(FLContextKey.TASK_RESULT, result, private=True, sticky=False)
    except BaseException as error:
        runtime.fail(error)
    finally:
        for component in reversed(initialized):
            try:
                finalize = getattr(component, "finalize", None)
                if finalize is not None:
                    with runtime.operation():
                        finalize(fl_ctx)
            except BaseException as error:
                runtime.fail(error)
    runtime.raise_if_failed()
    result = fl_ctx.get_prop(FLContextKey.TASK_RESULT)
    if not isinstance(result, Shareable):
        raise TypeError("executor finalization must leave a Shareable result")
    runtime.raise_if_failed()
    return result, runtime


def run_worker(bootstrap_path: str, *, context_binding=None, protected_context_keys=()) -> TaskCompletion:
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
        _runtime, fl_ctx = _new_context(bootstrap, workspace, protected_context_keys)
        builder = _authorize_compute_graph(bootstrap, workspace, fl_ctx)
        if workspace.get_log_config_file_path():
            configure_logging(workspace, identity.job_id, file_prefix=f"task_{identity.attempt_id}")
        else:
            logging.basicConfig(level=logging.INFO)
        refresh_custom_dir_import_path(workspace.get_app_custom_dir(identity.job_id))
        fobs_initialize(workspace=workspace, job_id=identity.job_id)
        # The same serialization registrations are needed on both sides of
        # the payload handoff, before decoding input or building the trainer.
        for index, original in enumerate(bootstrap.components):
            spec = _thaw_json(original)
            if builder.get_class_path(spec) == "nvflare.app_common.widgets.decomposer_reg.DecomposerRegister":
                builder.build_component(spec, builder.make_component_node(spec, index + 1))
        data = store.read_input(identity)
        analytics = []
        if context_binding is None:
            result, runtime = _execute(bootstrap, data, workspace, store)
        else:
            result, runtime = _execute(
                bootstrap,
                data,
                workspace,
                store,
                context_binding=context_binding,
                protected_context_keys=protected_context_keys,
                analytics=analytics,
            )
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
        if analytics:
            diagnostics["analytics"] = store.write_analytics(identity, analytics).to_dict()
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
