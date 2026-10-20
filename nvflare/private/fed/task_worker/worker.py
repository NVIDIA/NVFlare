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

"""Execute staged assignments without a federation Cell or live job engine."""

import logging
import os
import resource
import sys
import tempfile
import time

from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import UnsafeComponentError
from nvflare.apis.job_launcher_spec import pop_credential_env
from nvflare.apis.shareable import Shareable
from nvflare.apis.workspace import Workspace
from nvflare.app_common.abstract.statistics_spec import Statistics
from nvflare.app_common.widgets.component_path_authorizer import ComponentPathAuthorizer
from nvflare.client.config import ExchangeFormat
from nvflare.client.decomposers import register_framework_decomposers
from nvflare.fuel.common.excepts import ComponentNotAuthorized
from nvflare.fuel.sec.audit import AuditService
from nvflare.fuel.utils.class_utils import ModuleScanner, get_class_path_from_config, instantiate_class
from nvflare.fuel.utils.json_scanner import Node
from nvflare.fuel.utils.log_utils import configure_logging
from nvflare.job_config.task_execution import (
    CLIENT_API_EXECUTOR_PATH,
    DECOMPOSER_REGISTER_PATH,
    STATISTICS_EXECUTOR_PATH,
    task_constructor_args,
)
from nvflare.private.fed.utils.fed_utils import fobs_initialize, get_job_meta_from_workspace
from nvflare.security.logging import secure_format_exception
from nvflare.utils.job_launcher_utils import refresh_custom_dir_import_path

from .artifacts import ArtifactReference, FileTaskArtifactStore, TaskCompletion, _read_json
from .protocol import SESSION_END_MARKER, TaskAttemptIdentity, WorkerBootstrap, _thaw_json, read_bootstrap
from .runtime import TaskRuntime

_SESSION_POLL_INTERVAL = 0.05

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


def _new_context(bootstrap: WorkerBootstrap, workspace: Workspace, protected_context_keys=(), *, components=None):
    identity = bootstrap.identity
    app_root = workspace.get_app_dir(identity.job_id)
    runtime = TaskRuntime(workspace, identity.site_name, identity.job_id, components=components)
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


def _class_path(config):
    scanner = ModuleScanner(["nvflare"], ["apis", "app_common", "widgets"], True)
    return get_class_path_from_config(config, resolve_name=scanner.get_module_name)


def _authorize_bootstrap(bootstrap, workspace, fl_ctx):
    """Check site policy before logging factories, codecs or executor imports."""
    authorizer = ComponentPathAuthorizer()
    config = bootstrap.to_dict()
    executor = config["executor"]
    generator_id = (
        task_constructor_args(executor).get("generator_id")
        if _class_path(executor) == STATISTICS_EXECUTOR_PATH
        else None
    )
    if _class_path(executor) == STATISTICS_EXECUTOR_PATH and (
        not isinstance(generator_id, str) or generator_id not in {spec["id"] for spec in config["components"]}
    ):
        raise ValueError("StatisticsExecutor requires a configured statistics generator_id")
    specs = [("executor", executor)]
    specs.extend((f"components.#{index}", spec) for index, spec in enumerate(config["components"], 1))
    for path, spec in specs:
        # Normalize aliases without importing the selected class.
        class_path = _class_path(spec)
        node = Node(spec)
        node.paths = path.split(".")
        try:
            authorizer.authorize_component_config(
                {**spec, "path": class_path}, node, fl_ctx=fl_ctx, workspace=workspace
            )
        except UnsafeComponentError as error:
            raise ComponentNotAuthorized(f"component not authorized: {error}") from error
        task_constructor_args(spec)
        if path != "executor" and class_path != DECOMPOSER_REGISTER_PATH and spec.get("id") != generator_id:
            raise ValueError(
                "task workers accept only decomposer registration and the statistics generator in components"
            )


def _build_executor(config):
    config = _thaw_json(config)
    executor = (
        None
        if config.get("disabled") is True
        else instantiate_class(_class_path(config), task_constructor_args(config))
    )
    if not isinstance(executor, Executor):
        raise TypeError(f"worker executor config built {type(executor)} instead of Executor")
    return executor


def _restore_peer_context(data: Shareable, fl_ctx: FLContext):
    peer_props = data.get_peer_props()
    if isinstance(peer_props, dict):
        peer_ctx = FLContext()
        peer_ctx.set_public_props(peer_props)
        fl_ctx.set_peer_context(peer_ctx)


class AttemptBinding:
    """Per-attempt handles that adapters read at call time, so one worker can serve several attempts."""

    def __init__(self, identity):
        self.identity = identity
        self.analytics = None


def _set_attempt_context(fl_ctx, identity):
    for name, value in (
        (FLContextKey.TASK_ID, identity.task_id),
        (FLContextKey.TASK_ATTEMPT_ID, identity.attempt_id),
        (FLContextKey.TASK_NAME, identity.task_name),
    ):
        fl_ctx.set_prop(name, value, private=True, sticky=False)


def _attempt_identities(bootstrap: WorkerBootstrap):
    yield bootstrap.identity
    if bootstrap.session_dir is None:
        return
    sequence = 2
    while True:
        next_path = os.path.join(bootstrap.session_dir, f"{sequence:06d}.json")
        if os.path.lexists(next_path):
            yield TaskAttemptIdentity.from_dict(_read_json(next_path))
            sequence += 1
        elif os.path.lexists(os.path.join(bootstrap.session_dir, SESSION_END_MARKER)):
            return
        else:
            time.sleep(_SESSION_POLL_INTERVAL)


def _commit_attempt(store, identity, result, analytics, runtime, started_at, started_monotonic, *, final):
    reference = result if isinstance(result, ArtifactReference) else store.stage_result(identity, result)
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
    checkpoint = store.stage_checkpoint(identity)
    if checkpoint is not None:
        diagnostics["checkpoint"] = checkpoint.to_dict()
    if analytics is not None:
        analytics_reference = store.write_analytics(identity, analytics)
        if analytics_reference is not None:
            diagnostics["analytics"] = analytics_reference.to_dict()
    return store.commit_staged_result(
        identity,
        reference,
        worker_pid=os.getpid(),
        worker_ppid=os.getppid(),
        started_at=started_at,
        completed_at=completed_at,
        diagnostics=diagnostics,
        publication_guard=runtime.completion_guard if final else runtime.attempt_guard,
    )


def _execute(
    bootstrap: WorkerBootstrap,
    workspace: Workspace,
    store: FileTaskArtifactStore,
    binding: AttemptBinding,
    *,
    context_binding=None,
    protected_context_keys=(),
    spool_analytics=False,
    started=None,
):
    """Run every attempt of this worker on one executor; return the last completion.

    A one-shot worker (no session directory) commits after finalization, as #5405 does.
    A session worker commits each attempt as it finishes and finalizes once at the end.
    """
    components = {}
    for original in bootstrap.components:
        spec = _thaw_json(original)
        if _class_path(spec) != DECOMPOSER_REGISTER_PATH:
            component = (
                None
                if spec.get("disabled") is True
                else instantiate_class(_class_path(spec), task_constructor_args(spec))
            )
            if not isinstance(component, Statistics):
                raise TypeError(f"statistics generator {spec['id']!r} must implement Statistics")
            components[spec["id"]] = component
    runtime, fl_ctx = _new_context(bootstrap, workspace, protected_context_keys, components=components or None)
    session = bootstrap.session_dir is not None
    adapt_executor = None
    if context_binding is not None:
        adapt_executor = context_binding(fl_ctx, binding, store, session)
    executor = _build_executor(bootstrap.executor)
    if adapt_executor is not None:
        executor = adapt_executor(executor)

    completion = None
    pending = None
    try:
        initialize = getattr(executor, "initialize", None)
        if initialize is not None:
            with runtime.operation():
                initialize(fl_ctx)
        runtime.raise_if_failed()
        for identity in _attempt_identities(bootstrap):
            # The first attempt's start is the worker's start, preflight included, as in #5405.
            started_at, started_monotonic = (
                started if started and identity == bootstrap.identity else (time.time(), time.monotonic())
            )
            if identity != bootstrap.identity:
                store.claim_worker(identity)
            binding.identity = identity
            binding.analytics = tempfile.TemporaryFile(dir=store.attempt_dir(identity)) if spool_analytics else None
            try:
                _set_attempt_context(fl_ctx, identity)
                data = store.read_input(identity)
                _restore_peer_context(data, fl_ctx)
                fl_ctx.set_prop(FLContextKey.TASK_DATA, data, private=True, sticky=False)
                with runtime.operation():
                    result = executor.execute(identity.task_name, data, fl_ctx, runtime.abort_signal)
                runtime.raise_if_failed()
                if not isinstance(result, (Shareable, ArtifactReference)):
                    raise TypeError(f"Executor returned {type(result)} instead of Shareable")
                fl_ctx.set_prop(FLContextKey.TASK_RESULT, result, private=True, sticky=False)
                if session:
                    completion = _commit_attempt(
                        store, identity, result, binding.analytics, runtime, started_at, started_monotonic, final=False
                    )
                else:
                    pending = (identity, started_at, started_monotonic, binding.analytics)
            finally:
                if session and binding.analytics is not None:
                    binding.analytics.close()
                    binding.analytics = None
    except BaseException as error:
        runtime.fail(error)
    finally:
        try:
            finalize = getattr(executor, "finalize", None)
            if finalize is not None:
                with runtime.operation():
                    finalize(fl_ctx)
        except BaseException as error:
            runtime.fail(error)
    runtime.raise_if_failed()
    if pending is not None:
        identity, started_at, started_monotonic, analytics = pending
        result = fl_ctx.get_prop(FLContextKey.TASK_RESULT)
        if not isinstance(result, (Shareable, ArtifactReference)):
            raise TypeError("executor finalization must leave a Shareable or staged result reference")
        try:
            completion = _commit_attempt(
                store, identity, result, analytics, runtime, started_at, started_monotonic, final=True
            )
        finally:
            if analytics is not None:
                analytics.close()
    runtime.raise_if_failed()
    return completion


def run_worker(bootstrap_path: str, *, context_binding=None, protected_context_keys=()) -> TaskCompletion:
    """Run one bootstrap: a single attempt, or every attempt of its session."""

    # Strip CJ bootstrap credentials from the environment before importing job
    # custom code. This is not filesystem isolation: Process workers share the
    # site's UID and workspace, which can contain credential files.
    pop_credential_env()
    sys.argv[:] = ["nvflare-task-worker"]
    started_at = time.time()
    started_monotonic = time.monotonic()
    bootstrap = read_bootstrap(bootstrap_path)
    identity = bootstrap.identity
    binding = AttemptBinding(identity)
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
        _authorize_bootstrap(bootstrap, workspace, fl_ctx)
        if workspace.get_log_config_file_path():
            configure_logging(workspace, identity.job_id, file_prefix=f"task_{identity.attempt_id}")
        else:
            logging.basicConfig(level=logging.INFO)
        refresh_custom_dir_import_path(workspace.get_app_custom_dir(identity.job_id))
        fobs_initialize(workspace=workspace, job_id=identity.job_id)
        if _class_path(bootstrap.executor) == CLIENT_API_EXECUTOR_PATH:
            args = task_constructor_args(_thaw_json(bootstrap.executor))
            register_framework_decomposers(
                args.get("params_exchange_format", ExchangeFormat.RAW),
                args.get("server_expected_format", ExchangeFormat.NUMPY),
            )
        # The same serialization registrations are needed on both sides of
        # the payload handoff, before decoding input or building the trainer.
        for original in bootstrap.components:
            spec = _thaw_json(original)
            if _class_path(spec) == DECOMPOSER_REGISTER_PATH and spec.get("disabled") is not True:
                instantiate_class(_class_path(spec), task_constructor_args(spec))
        return _execute(
            bootstrap,
            workspace,
            store,
            binding,
            context_binding=context_binding,
            protected_context_keys=protected_context_keys,
            spool_analytics=context_binding is not None,
            started=(started_at, started_monotonic),
        )
    except BaseException as e:
        try:
            store.record_failure(
                binding.identity,
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
