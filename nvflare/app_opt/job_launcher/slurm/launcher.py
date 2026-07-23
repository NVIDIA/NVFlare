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
"""Public NVFlare integration surface for the Slurm job launcher."""

from __future__ import annotations

import os
import shlex
from abc import abstractmethod
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, JobConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.job_launcher_spec import JobHandleSpec, JobLauncherSpec, JobProcessArgs, add_launcher
from nvflare.apis.utils.format_check import check_job_id
from nvflare.apis.workspace import Workspace
from nvflare.app_opt.job_launcher.slurm.config import (
    _JOB_SLURM_KEYS,
    CONTAINER_RESOLV_CONF,
    SLURM_CHILD_PROCESS_ENV,
    BindMount,
    JobResources,
    LaunchPlan,
    SlurmConfig,
    SlurmLauncherError,
    _mapping_or_empty,
    _paths_overlap,
    _require_int,
    _require_positive_number,
    _require_string,
    _validate_mount_destination,
    _validate_mount_source,
    normalize_slurm_image,
    normalize_slurm_launcher_settings,
    normalize_slurm_workspace_path,
)
from nvflare.app_opt.job_launcher.slurm.manager import SlurmJobManager
from nvflare.app_opt.job_launcher.study_runtime import (
    StudyRuntime,
    load_study_runtime_file,
    resolve_study_runtime,
    study_runtime_file_path,
)
from nvflare.utils.job_launcher_utils import (
    get_client_job_args,
    get_credential_env,
    get_job_launcher_spec,
    get_server_job_args,
)


def _validate_run_dir(workspace_path: str, run_dir: str) -> str:
    if os.path.islink(run_dir) or not os.path.isdir(run_dir):
        raise SlurmLauncherError(f"job run directory must be an existing non-symlink directory: {run_dir}")
    workspace_real = os.path.realpath(workspace_path)
    run_real = os.path.realpath(run_dir)
    if os.path.dirname(run_real) != workspace_real:
        raise SlurmLauncherError(f"job run directory must be an immediate child of workspace_path: {run_dir}")
    return run_real


def _resolve_resources(
    job_meta: dict,
    site_name: str,
    sandbox: str,
    site_pending_timeout: float,
    spec: dict,
) -> JobResources:
    spec = _mapping_or_empty(spec, f"Slurm spec for site '{site_name}'")
    unknown = set(spec) - _JOB_SLURM_KEYS
    if unknown:
        raise SlurmLauncherError(f"unsupported job-owned Slurm key(s): {sorted(unknown)}")

    nodes = _require_int(spec.get("nodes", 1), "nodes")
    if nodes > 1 and sandbox != "none":
        raise SlurmLauncherError("multi-node Slurm jobs require effective sandbox 'none'")

    def optional_int(name: str) -> Optional[int]:
        value = spec.get(name)
        return None if value is None else _require_int(value, name)

    gpus_per_node = optional_int("gpus_per_node")
    resources = _mapping_or_empty(job_meta.get(JobMetaKey.RESOURCE_SPEC.value), "resource_spec")
    site = _mapping_or_empty(resources.get(site_name), f"resource_spec for site '{site_name}'")
    portable_total = site.get("num_of_gpus")
    if portable_total is not None:
        portable_total = _require_int(portable_total, "num_of_gpus", 0)
    if gpus_per_node is not None and portable_total is not None and portable_total != nodes * gpus_per_node:
        raise SlurmLauncherError("num_of_gpus must equal nodes * gpus_per_node when both are specified")
    if nodes > 1 and portable_total and gpus_per_node is None:
        raise SlurmLauncherError("multi-node GPU jobs require explicit gpus_per_node")
    if nodes == 1 and gpus_per_node is None and portable_total:
        gpus_per_node = portable_total

    cpus_per_node = optional_int("cpus_per_node")
    mem_per_node = optional_int("mem_per_node")
    time_limit = spec.get("time")
    if time_limit is not None:
        time_limit = _require_string(time_limit, "time")
    pending_timeout = _require_positive_number(spec.get("pending_timeout", site_pending_timeout), "pending_timeout")
    if pending_timeout > site_pending_timeout:
        raise SlurmLauncherError("job pending_timeout may only reduce the site value")
    return JobResources(
        nodes=nodes,
        gpus_per_node=gpus_per_node,
        cpus_per_node=cpus_per_node,
        mem_per_node=mem_per_node,
        time_limit=time_limit,
        pending_timeout=pending_timeout,
    )


def _module_args(job_args: dict, arg_names: list[str]) -> tuple:
    result = []
    for name in arg_names:
        entry = job_args.get(name)
        if not entry:
            continue
        flag, value = entry
        result.append(str(flag))
        if name == JobProcessArgs.OPTIONS:
            if not isinstance(value, str):
                raise SlurmLauncherError("JOB_PROCESS_ARGS options must be a string")
            try:
                result.extend(shlex.split(value))
            except ValueError as e:
                raise SlurmLauncherError("malformed JOB_PROCESS_ARGS options") from e
        else:
            result.append(str(value))
    return tuple(result)


def _resolve_parent_host(configured_host: Optional[str]) -> str:
    if configured_host is not None:
        return _require_string(configured_host, "parent_host")
    if "SLURM_JOB_ID" in os.environ:
        return _require_string(os.environ.get("SLURMD_NODENAME"), "SLURMD_NODENAME")
    raise SlurmLauncherError("parent_host is required when the CP does not run inside a Slurm allocation")


def _rewrite_parent_url(job_args: dict, parent_host: Optional[str], internal_port: int) -> dict:
    copied = dict(job_args)
    entry = copied.get(JobProcessArgs.PARENT_URL)
    if not isinstance(entry, (tuple, list)) or len(entry) != 2:
        raise SlurmLauncherError(f"missing or malformed {JobProcessArgs.PARENT_URL} in JOB_PROCESS_ARGS")
    flag, raw_url = entry
    try:
        parsed = urlsplit(str(raw_url))
        port = parsed.port
        host = parsed.hostname
    except ValueError as e:
        raise SlurmLauncherError("malformed parent URL in JOB_PROCESS_ARGS") from e
    if parsed.scheme != "tcp" or port != internal_port or not host:
        raise SlurmLauncherError(
            f"parent URL must use tcp and configured internal_port {internal_port}, got {raw_url!r}"
        )
    host = _resolve_parent_host(parent_host)
    rendered_host = host if host.startswith("[") and host.endswith("]") else f"[{host}]" if ":" in host else host
    netloc = f"{rendered_host}:{internal_port}"
    rewritten = urlunsplit(("tcp", netloc, parsed.path, parsed.query, parsed.fragment))
    copied[JobProcessArgs.PARENT_URL] = (flag, rewritten)
    return copied


class SlurmJobLauncher(JobLauncherSpec):
    """Common lifecycle and launch-plan construction for client and server jobs."""

    EXE_MODULE: Optional[str] = None

    def __init__(
        self,
        *,
        workspace_path: str,
        sandbox: str,
        python_path: str,
        executables: dict,
        image: Optional[str] = None,
        internal_port: int = 8102,
        sbatch_directives: Optional[dict] = None,
        setup: str = "",
        forward_env: Optional[list] = None,
        parent_host: Optional[str] = None,
        poll_interval: float = 10.0,
        pending_timeout: float = 600.0,
    ):
        super().__init__()
        if not self.EXE_MODULE:
            raise TypeError("SlurmJobLauncher must be instantiated through a concrete subclass")
        self._child_process = os.environ.get(SLURM_CHILD_PROCESS_ENV) == "1"
        self.config = SlurmConfig(
            workspace_path=normalize_slurm_workspace_path(workspace_path),
            **normalize_slurm_launcher_settings(
                sandbox=sandbox,
                python_path=python_path,
                executables=executables,
                image=image,
                internal_port=internal_port,
                sbatch_directives=sbatch_directives,
                setup=setup,
                forward_env=forward_env,
                parent_host=parent_host,
                poll_interval=poll_interval,
                pending_timeout=pending_timeout,
            ),
        )
        self.manager = None
        if not self._child_process:
            self.manager = SlurmJobManager(
                config=self.config,
                logger=self.logger,
            )

    @abstractmethod
    def get_module_args(self, job_args: dict) -> tuple:
        """Select and structure the worker arguments for this launcher type."""

    def _load_study_runtime(self, study: Optional[str]) -> StudyRuntime:
        runtime_file = study_runtime_file_path(self.config.workspace_path)
        legacy_file = os.path.join(self.config.workspace_path, "local", "study_data.yaml")
        runtime_exists = os.path.exists(runtime_file)
        if os.path.exists(legacy_file):
            message = (
                f"study runtime file '{runtime_file}' cannot be combined with legacy '{legacy_file}'"
                if runtime_exists
                else f"legacy study data file '{legacy_file}' is unsupported by Slurm; migrate to study_runtime.yaml"
            )
            raise SlurmLauncherError(message)
        if not runtime_exists:
            return StudyRuntime(study=study or "")
        try:
            runtime_map = load_study_runtime_file(runtime_file, launcher_mode="slurm", logger=self.logger)
            return resolve_study_runtime(runtime_map, study, runtime_file, logger=self.logger)
        except ValueError as e:
            raise SlurmLauncherError(str(e)) from e

    def _effective_study_values(
        self, runtime: StudyRuntime, job_image: Optional[str] = None
    ) -> tuple[str, Optional[str], str, dict]:
        slurm = runtime.slurm
        sandbox = slurm.get("sandbox", self.config.sandbox)
        if job_image is not None:
            job_image = _require_string(job_image, "job-owned Slurm image")
        if sandbox == "none" and (job_image or runtime.container_image or runtime.datasets or runtime.secret_mounts):
            raise SlurmLauncherError(
                "effective sandbox 'none' does not support job/study container, datasets, or secret_mounts"
            )
        image = (
            None
            if sandbox == "none"
            else normalize_slurm_image(
                job_image or runtime.container_image or self.config.image,
                sandbox,
                require_file=True,
                label="effective image",
            )
        )
        setup = slurm.get("setup", self.config.setup)
        directives = dict(self.config.sbatch_directives)
        directives.update({key: slurm[key] for key in ("partition", "account", "qos") if key in slurm})
        return sandbox, image, setup, directives

    def _study_environment(self, runtime: StudyRuntime) -> tuple[dict, dict]:
        environment = dict(runtime.env)
        secret_environment = {}
        for reference in runtime.secret_env:
            if reference.source not in os.environ:
                raise SlurmLauncherError(
                    f"study secret_env '{reference.name}' requires parent environment variable '{reference.source}'"
                )
            secret_environment[reference.name] = _require_string(
                os.environ[reference.source],
                f"study secret_env '{reference.name}' value",
                allow_empty=True,
            )
        duplicated = (set(environment) | set(secret_environment)) & set(self.config.forward_env)
        if duplicated:
            raise SlurmLauncherError(
                f"study env, secret_env, and forward_env contain duplicate name(s): {sorted(duplicated)}"
            )
        return environment, secret_environment

    def _study_mounts(self, runtime: StudyRuntime) -> tuple:
        workspace = self.config.workspace_path
        owned_destinations = (workspace, CONTAINER_RESOLV_CONF)
        mounts = []
        specs = [
            (dataset, f"dataset '{dataset.dataset}'", "ro" if dataset.read_only else "rw")
            for dataset in runtime.datasets
        ] + [(secret, f"secret mount '{secret.name}'", "ro") for secret in runtime.secret_mounts]
        for item, label, mode in specs:
            destination = _validate_mount_destination(item.mount_path, f"{label} destination")
            owned_overlap = any(_paths_overlap(destination, owned) for owned in owned_destinations)
            other_overlap = any(_paths_overlap(destination, mount.destination) for mount in mounts)
            if owned_overlap or other_overlap:
                target = "a launcher-owned path" if owned_overlap else "another study mount"
                raise SlurmLauncherError(f"{label} destination overlaps {target}: {destination}")
            source = _validate_mount_source(
                item.source,
                workspace,
                f"{label} source",
            )
            mounts.append(BindMount(source, destination, mode))
        return tuple(mounts)

    @staticmethod
    def _python_environment(workspace: Workspace, job_id: str) -> str:
        custom_paths = (workspace.get_app_custom_dir(job_id), workspace.get_site_custom_dir())
        return os.pathsep.join(dict.fromkeys(path for path in custom_paths if path and os.path.isdir(path)))

    def _build_launch_plan(self, job_meta: dict, fl_ctx: FLContext) -> LaunchPlan:
        if not isinstance(job_meta, dict):
            raise SlurmLauncherError("job_meta must be a mapping")
        job_id = job_meta.get(JobConstants.JOB_ID)
        try:
            check_job_id(job_id)
        except ValueError as e:
            raise SlurmLauncherError("invalid job ID") from e
        if "\n" in job_id or "\r" in job_id:
            raise SlurmLauncherError("invalid job ID")
        site_name = _require_string(fl_ctx.get_identity_name(), "site identity")
        workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        if not isinstance(workspace, Workspace):
            raise SlurmLauncherError(f"missing {FLContextKey.WORKSPACE_OBJECT} in FLContext")
        if os.path.realpath(workspace.get_root_dir()) != self.config.workspace_path:
            raise SlurmLauncherError("FLContext workspace does not match configured Slurm workspace_path")
        run_dir = _validate_run_dir(self.config.workspace_path, workspace.get_run_dir(job_id))

        raw_job_args = fl_ctx.get_prop(FLContextKey.JOB_PROCESS_ARGS)
        if not isinstance(raw_job_args, dict) or not raw_job_args:
            raise SlurmLauncherError(f"missing {FLContextKey.JOB_PROCESS_ARGS} in FLContext")
        connection_entry = raw_job_args.get(JobProcessArgs.PARENT_CONN_SEC, (None, "clear"))
        if not isinstance(connection_entry, (tuple, list)) or len(connection_entry) != 2:
            raise SlurmLauncherError(f"malformed {JobProcessArgs.PARENT_CONN_SEC} in JOB_PROCESS_ARGS")
        process_connection_security = _require_string(connection_entry[1], f"{JobProcessArgs.PARENT_CONN_SEC} value")
        if process_connection_security != "clear":
            raise SlurmLauncherError("Slurm job launch requires clear parent connection security")
        job_args = _rewrite_parent_url(
            raw_job_args,
            parent_host=self.config.parent_host,
            internal_port=self.config.internal_port,
        )

        study = job_meta.get(JobMetaKey.STUDY.value)
        if study is not None:
            study = _require_string(study, "study name")
        runtime = self._load_study_runtime(study)
        job_spec = _mapping_or_empty(
            get_job_launcher_spec(job_meta, site_name, "slurm"), f"Slurm spec for site '{site_name}'"
        )
        # AppDeployer classifies a selected Slurm image as BYOC and authorizes it
        # before the locally deployed job reaches this launcher.
        job_image = job_spec.get("image")
        sandbox, image, setup, directives = self._effective_study_values(runtime, job_image)
        resources = _resolve_resources(
            job_meta,
            site_name,
            sandbox,
            self.config.pending_timeout,
            spec=job_spec,
        )
        study_env, secret_env = self._study_environment(runtime)
        secret_env.update(get_credential_env(job_args))
        mounts = self._study_mounts(runtime) if sandbox != "none" else ()
        return LaunchPlan(
            job_id=job_id,
            run_dir=run_dir,
            exe_module=self.EXE_MODULE,
            module_args=self.get_module_args(job_args),
            resources=resources,
            directives=directives,
            sandbox=sandbox,
            image=image,
            setup=setup,
            study_env=study_env,
            study_secret_env=secret_env,
            mounts=mounts,
            python_path=self.config.python_path,
            python_env=self._python_environment(workspace, job_id),
            forward_env=self.config.forward_env,
        )

    def launch_job(self, job_meta: dict, fl_ctx: FLContext) -> JobHandleSpec:
        if self.manager is None:
            raise SlurmLauncherError("nested Slurm job launch is unavailable inside a Slurm child process")
        plan = self._build_launch_plan(job_meta, fl_ctx)
        return self.manager.launch(plan)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if self.manager is None:
            return
        if event_type == EventType.SYSTEM_BOOTSTRAP:
            self.manager.initialize()
        elif event_type == EventType.BEFORE_JOB_LAUNCH:
            add_launcher(self, fl_ctx)
        elif event_type == EventType.SYSTEM_END:
            self.manager.shutdown()


class ClientSlurmJobLauncher(SlurmJobLauncher):
    EXE_MODULE = "nvflare.private.fed.app.client.worker_process"

    def get_module_args(self, job_args: dict) -> tuple:
        return _module_args(job_args, get_client_job_args(include_exe_module=False, include_set_options=True))


class ServerSlurmJobLauncher(SlurmJobLauncher):
    EXE_MODULE = "nvflare.private.fed.app.server.runner_process"

    def get_module_args(self, job_args: dict) -> tuple:
        return _module_args(job_args, get_server_job_args(include_exe_module=False, include_set_options=True))
