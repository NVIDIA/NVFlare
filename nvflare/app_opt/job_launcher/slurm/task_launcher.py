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
"""Single-node CPU and GPU Slurm tasks on a coherent shared filesystem.

This adapter implements TaskLauncherSpec. Trusted CJ selection and materializing
the backend worker command are separate integration prerequisites; see the
deployment documentation. It never interprets job metadata as site policy.
"""

import hashlib
import ipaddress
import json
import math
import os
import re
import shlex
import socket
import tempfile
import threading
import time
from dataclasses import dataclass, replace

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.job_launcher_spec import JobProcessEnv
from nvflare.apis.task_launcher_spec import (
    TaskExecutionPhase,
    TaskExecutionStatus,
    TaskHandleSpec,
    TaskLauncherError,
    TaskLaunchError,
    TaskLauncherSpec,
    TaskLaunchRequest,
    TaskResourceRequest,
    TaskSettlementError,
    UnsupportedTaskResourceError,
)
from nvflare.app_opt.job_launcher.slurm.batch import _submission_argv
from nvflare.app_opt.job_launcher.slurm.config import (
    BATCH_FILE,
    SLURM_CHILD_PROCESS_ENV,
    JobResources,
    LaunchPlan,
    LookupStatus,
    SlurmConfig,
    _require_positive_number,
    _require_string,
    normalize_slurm_directives,
    normalize_slurm_executables,
    normalize_slurm_workspace_path,
)
from nvflare.app_opt.job_launcher.slurm.manager import (
    SlurmJobHandle,
    SlurmJobManager,
    _ensure_dir,
    _is_terminal,
    _write_exclusive,
)
from nvflare.app_opt.job_launcher.slurm.scheduler_client import _command_diagnostic
from nvflare.app_opt.job_launcher.slurm.task_scheduler import SlurmTaskCliAdapter, submission_was_rejected
from nvflare.client.api_spec import CLIENT_API_TYPE_KEY
from nvflare.client.cell.bootstrap import BOOTSTRAP_FILE_ENV_VAR
from nvflare.private.fed.task_worker.protocol import CELL_SESSION_CREDENTIALS, _open_regular, read_bootstrap
from nvflare.private.fed.utils.job_cert_utils import job_cert_paths

WORKER_MODULE = "nvflare.app_opt.job_launcher.slurm.task_worker"


@dataclass(frozen=True)
class _JobBinding:
    job_id: str
    site_name: str
    run_dir: str
    resources: JobResources
    config: SlurmConfig
    python_path: str
    environment: tuple


_QUEUED = {"PENDING", "CONFIGURING"}
_GPU_ENV = ("CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "ROCR_VISIBLE_DEVICES")
_ALLOCATION_ENV = (
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_NODELIST",
    "SLURM_JOB_NUM_NODES",
    "SLURMD_NODENAME",
    "SLURM_CPUS_PER_TASK",
    "SLURM_MEM_PER_NODE",
    "SLURM_GPUS_ON_NODE",
    "SLURM_JOB_GPUS",
    "SLURM_STEP_GPUS",
    "SLURM_NTASKS",
    "SLURM_PROCID",
    "SLURM_LOCALID",
)
_FORBIDDEN_ENV = (
    set(JobProcessEnv.ALL)
    | set(_GPU_ENV)
    | {
        "HIP_VISIBLE_DEVICES",
        "NVFLARE_SLURM_TASK_OWNER",
        "NVFLARE_SLURM_TASK_TOKEN",
        "NVIDIA_VISIBLE_DEVICES",
        CLIENT_API_TYPE_KEY,
        BOOTSTRAP_FILE_ENV_VAR,
        "NVFLARE_CLIENT_API_CONFIG",
        "NVFLARE_CLIENT_API_CONFIG_PATH",
        "NVFLARE_CLIENT_API_SERVICE_CONF",
    }
)


def _attempt_key(request):
    return hashlib.sha256(json.dumps(request.identity, separators=(",", ":")).encode("utf-8")).hexdigest()


def _check_filesystem(root):
    """Exercise local durability primitives, not cross-host coherence."""
    with tempfile.TemporaryDirectory(prefix=".fs-probe-", dir=root) as probe:
        source, target = os.path.join(probe, "source"), os.path.join(probe, "target")
        _write_exclusive(source, b"committed", 0o600)
        with open(source, "rb") as stream:
            os.fsync(stream.fileno())
        os.link(source, target, follow_symlinks=False)
        try:
            os.link(source, target, follow_symlinks=False)
        except FileExistsError:
            pass
        else:
            raise TaskLauncherError("shared filesystem does not enforce exclusive hardlink publication")
        replacement = os.path.join(probe, "replacement")
        _write_exclusive(replacement, b"promoted", 0o600)
        with open(replacement, "rb") as stream:
            os.fsync(stream.fileno())
        with open(target, "rb") as active_reader:
            os.replace(replacement, target)
            if active_reader.read() != b"committed":
                raise TaskLauncherError("shared filesystem replacement changed an active reader")
        with open(target, "rb") as stream:
            if stream.read() != b"promoted":
                raise TaskLauncherError("shared filesystem replacement is not visible")
        directory_fd = os.open(probe, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def _batch_script(request, setup):
    # No inherited CJ environment, credential file, workspace teardown, or
    # shell interpolation of command arguments. Scheduler GPU visibility is
    # captured in the compute allocation, never from the submitting CJ.
    names = sorted(request.environment)
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        '[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]] || exit 101',
    ]
    lines.extend(f"export {name}={shlex.quote(request.environment[name])}" for name in names)
    lines.extend([setup, "set +x", "_nvfl_env=()"])
    lines.extend(f'_nvfl_env+=("{name}=${{{name}}}")' for name in names)
    for name in _ALLOCATION_ENV:
        lines.append(f'if [[ ${{{name}+x}} ]]; then _nvfl_env+=("{name}=${{{name}}}"); fi')
    if request.resources.gpu_count:
        lines.append(
            '[[ -n "${CUDA_VISIBLE_DEVICES:-}${ROCR_VISIBLE_DEVICES:-}" ]] || '
            '{ echo "Slurm did not supply GPU visibility" >&2; exit 102; }'
        )
        for name in _GPU_ENV:
            lines.append(f'if [[ ${{{name}+x}} ]]; then _nvfl_env+=("{name}=${{{name}}}"); fi')
    else:
        lines.append(
            "_nvfl_env+=(CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES=none)"
        )
    lines.append('exec /usr/bin/env -i "${_nvfl_env[@]}" ' + shlex.join(request.argv))
    return "\n".join(lines) + "\n"


class SlurmTaskManager(SlurmJobManager):
    """Reuse workspace, submission serialization and file ownership mechanisms.

    Task attempts retain control files even after settlement. Resident job
    cleanup and its weaker accounting fallback never drive task settlement.
    """

    def initialize(self):
        if getattr(self, "_task_ready", False):
            return
        super().initialize()
        with self._lock:
            try:
                self.jobs_dir = os.path.join(self.config.workspace_path, ".nvflare_slurm", "tasks")
                _ensure_dir(self.jobs_dir)
                _check_filesystem(self.jobs_dir)
            except BaseException:
                self._initialized = False
                raise
            self._task_ready = True

    def _marker(self, key):
        return f"nvfl-task:{key}"

    def launch_task(self, request, resources=None):
        self._require_initialized()
        key = _attempt_key(request)
        with self._submission_gate:
            if self._reject_launches:
                raise TaskLauncherError("Slurm task launcher is shutting down")
            # The exclusive directory fences identities across launcher objects
            # and across restarts. No physical retry/adoption is implied.
            _, job_dir = self._prepare_job_dir(key, False)
            name = f"nvfl-task-{key[:48]}"
            handle = SlurmTaskHandle(self, request, key, name, job_dir)
            _write_exclusive(
                os.path.join(job_dir, "identity.json"),
                json.dumps(dict(zip(("job_id", "site_name", "task_id", "attempt_id"), request.identity))).encode(),
                0o600,
            )
            _write_exclusive(
                os.path.join(job_dir, BATCH_FILE), _batch_script(request, self.config.setup).encode(), 0o700
            )
            resources = resources or JobResources(
                cpus_per_node=int(request.resources.cpu_cores or 1),
                mem_per_node=request.resources.memory_mb,
                gpus_per_node=request.resources.gpu_count,
            )
            plan = LaunchPlan(
                job_id=key,
                site_name=request.site_name,
                run_dir=request.cwd,
                exe_module="",
                module_args=(),
                resources=resources,
                directives=self.config.sbatch_directives,
                sandbox="none",
                image=None,
                setup="",
                study_env={},
                study_secret_env={},
                mounts=(),
                python_path="",
                python_env="",
                forward_env=(),
                clear_gres=True,
            )
            argv = _submission_argv(plan, job_dir, name, self._marker(key), self.config)
            with self._lock:
                self._handles[key] = handle
            submission = None
            try:
                submission = self.adapter.submit(argv, self.config.submit_timeout)
                handle.job_id, handle.cluster = submission.job_id, submission.cluster
                handle._record("submitted")
                if not submission.command.available or not submission.job_id or submission.cluster:
                    diagnostic = _command_diagnostic(submission.command)
                    handle.failure_reason = f"Slurm task submission failed: {diagnostic}"
                    if submission_was_rejected(submission):
                        handle._status(
                            TaskExecutionPhase.TERMINAL,
                            settled=True,
                            exit_code=submission.command.returncode,
                            failure=handle.failure_reason,
                        )
                    elif submission.cluster and submission.job_id:
                        handle.request_cancel()
                        # The exact returned identity is used, as in the resident
                        # launcher. Acknowledgement never proves settlement.
                        response = self.adapter.cancel_suffix(
                            submission.job_id, submission.cluster, timeout=self.config.cancel_timeout
                        )
                        handle._cancel_acknowledged = response.status == LookupStatus.FOUND
                        handle._cancel_sent_at = self._monotonic()
                        handle._record("cancel_ack", status=response.status.value)
                    raise TaskLauncherError(handle.failure_reason)
            except BaseException as e:
                if not handle.failure_reason:
                    detail = _command_diagnostic(submission.command) if submission else f"{type(e).__name__}: {e}"
                    handle.failure_reason = f"Slurm task submission unresolved: {detail}"
                # sbatch may have submitted work before losing its response.
                # Never delete scripts or permit identity reuse after that call.
                raise TaskLaunchError(handle.failure_reason, handle) from e
            return handle


class SlurmTaskHandle(SlurmJobHandle, TaskHandleSpec):
    def __init__(self, manager, request, key, name, job_dir):
        super().__init__(
            manager, key, name, job_dir, None, manager.config.pending_timeout, manager.config.poll_interval
        )
        self._request = request
        self.operation_lock = threading.RLock()
        self.cluster = None
        self._cleanup_adapter = None
        self.failure_reason = None
        self._settled = None
        self._last_status = TaskExecutionStatus(TaskExecutionPhase.QUEUED)
        self._submitted_at = manager._monotonic()
        self._last_evidence = None
        self._last_queue = None
        self._cancel_sent_at = float("-inf")
        self._cancel_acknowledged = False
        self._next_query_at = float("-inf")

    @property
    def request(self):
        return self._request

    @property
    def execution_id(self):
        physical = self.job_id or "unresolved"
        return f"slurm:{self.cluster or 'local'}:{physical}:{self.nvflare_job_id}"

    def _record(self, event, **facts):
        record = {"event": event, "execution_id": self.execution_id, "time": time.time(), **facts}
        with open(os.path.join(self.job_dir, "scheduler.jsonl"), "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _status(self, phase, *, settled=False, exit_code=None, termination_signal=None, failure=None):
        if self.failure_reason and failure and failure != self.failure_reason:
            failure = f"{self.failure_reason}; {failure}"
        status = TaskExecutionStatus(
            phase,
            exit_code=exit_code,
            termination_signal=termination_signal,
            cancel_requested=self.cancel_requested,
            settled=settled,
            failure_reason=failure or self.failure_reason,
        )
        if status != self._last_status:
            self._record(
                "status",
                phase=phase.value,
                settled=settled,
                failure_reason=status.failure_reason,
                exit_code=exit_code,
                termination_signal=termination_signal,
                cancel_requested=self.cancel_requested,
            )
        self._last_status = status
        if settled:
            self._settled = status
            with self.manager._lock:
                if self.manager._handles.get(self.nvflare_job_id) is self:
                    self.manager._handles.pop(self.nvflare_job_id)
        return status

    def poll(self):
        with self.operation_lock:
            if self._settled is not None:
                return replace(self._settled, cancel_requested=self.cancel_requested)
            if self.manager._monotonic() < self._next_query_at:
                return replace(self._last_status, cancel_requested=self.cancel_requested)
            self._next_query_at = self.manager._monotonic() + self.poll_interval
            try:
                from nvflare.app_opt.job_launcher.slurm.task_reconcile import operator_release_reason

                try:
                    reason = operator_release_reason(self)
                except (OSError, ValueError, TypeError) as error:
                    # A stale or malformed attestation never blocks native
                    # cancellation and scheduler-confirmed settlement.
                    self._record("operator_release_rejected", error=str(error))
                    reason = None
                if reason:
                    self.cancel_requested = True
                    self._record("operator_release", reason=reason)
                    return self._status(
                        TaskExecutionPhase.TERMINAL,
                        settled=True,
                        exit_code=1,
                        failure=f"Operator reconciled allocation: {reason}",
                    )
                if not self.job_id:
                    result = self.manager.adapter.task_by_marker(
                        self.job_name,
                        self.manager._marker(self.nvflare_job_id),
                        timeout=self.manager.config.query_timeout,
                    )
                    if result.status != LookupStatus.FOUND:
                        return self._status(
                            TaskExecutionPhase.TERMINAL, failure="Slurm allocation identity is unresolved"
                        )
                    if len(result.records) != 1 or not re.fullmatch(r"[1-9][0-9]*", result.records[0].job_id):
                        raise TaskLauncherError("Slurm task marker lookup is ambiguous")
                    self.job_id = result.records[0].job_id
                    self.cancel_requested = True
                    self._record("resolved_for_cleanup")
                return self._observe()
            except Exception as e:
                # A failed/invalid query is never evidence of released resources.
                return self._status(
                    self._last_status.phase, failure=f"Slurm task observation failed: {type(e).__name__}"
                )

    def _observe(self):
        config = self.manager.config
        if self.cluster:
            if self._cleanup_adapter is None:
                self._cleanup_adapter = self.manager.adapter.for_cluster(self.cluster)
            adapter = self._cleanup_adapter
        else:
            adapter = self.manager.adapter
        active = adapter.active_by_id(
            self.job_id, self.job_name, self.manager._marker(self.nvflare_job_id), timeout=config.query_timeout
        )
        queue = (active.status.value, tuple(r.state for r in active.records))
        if queue != self._last_queue:
            self._record("queue", status=queue[0], states=queue[1])
            self._last_queue = queue
        if active.status == LookupStatus.UNAVAILABLE:
            return self._status(self._last_status.phase, failure="Slurm queue is unavailable")
        if active.status == LookupStatus.FOUND:
            record = active.records[0]
            phase = TaskExecutionPhase.QUEUED if record.state in _QUEUED else TaskExecutionPhase.RUNNING
            if _is_terminal(record.state):
                phase = TaskExecutionPhase.TERMINAL
            if (
                phase == TaskExecutionPhase.QUEUED
                and self.manager._monotonic() - self._submitted_at >= self.pending_timeout
            ):
                self.cancel_requested = True
                self.failure_reason = "Slurm task exceeded pending_timeout"
            if (
                self.cancel_requested
                and not self._cancel_acknowledged
                and self.manager._monotonic() - self._cancel_sent_at >= self.poll_interval
            ):
                self._cancel_sent_at = self.manager._monotonic()
                response = adapter.cancel(self.job_id, timeout=config.cancel_timeout)
                self._cancel_acknowledged = response.status == LookupStatus.FOUND
                self._record("cancel_ack", status=response.status.value)
            return self._status(phase)
        accounting = adapter.task_accounting(
            self.job_id, self.job_name, self.manager._marker(self.nvflare_job_id), timeout=config.query_timeout
        )
        if accounting.status != LookupStatus.FOUND:
            return self._status(self._last_status.phase, failure="Slurm task accounting is missing or unavailable")
        records = accounting.records
        evidence = tuple(
            (r.job_id, r.state, r.exit_status, r.exit_signal, r.derived_exit_status, r.derived_exit_signal)
            for r in records
        )
        if evidence != self._last_evidence:
            self._record("accounting", records=evidence)
            self._last_evidence = evidence
        allocation = next(r for r in records if r.job_id == self.job_id)
        if not _is_terminal(allocation.state) or any(not _is_terminal(r.state) for r in records):
            return self._status(TaskExecutionPhase.TERMINAL, failure="Slurm allocation/steps have not settled")
        # A completed allocation must include its batch step. A queued cancel
        # or launch failure can legitimately have no steps at all.
        if allocation.state == "COMPLETED" and not any(r.job_id == f"{self.job_id}.batch" for r in records):
            return self._status(TaskExecutionPhase.TERMINAL, failure="Slurm batch accounting is incomplete")
        # Recheck the live controller after accounting to reject completion
        # races (COMPLETING, epilog cleanup, requeue, or a still-active step).
        active = adapter.active_by_id(
            self.job_id, self.job_name, self.manager._marker(self.nvflare_job_id), timeout=config.query_timeout
        )
        if active.status != LookupStatus.NOT_FOUND:
            return self._status(TaskExecutionPhase.TERMINAL, failure="Slurm release is not confirmed by the controller")
        bad = next(
            (
                r
                for r in records
                if r.state != "COMPLETED"
                or r.exit_status
                or r.exit_signal
                or r.derived_exit_status
                or r.derived_exit_signal
            ),
            None,
        )
        return self._status(
            TaskExecutionPhase.TERMINAL,
            settled=True,
            exit_code=(bad.exit_status or bad.derived_exit_status or 1) if bad else 0,
            termination_signal=(bad.exit_signal or bad.derived_exit_signal or None) if bad else None,
            failure=f"Slurm task {bad.job_id}: {bad.state}" if bad else None,
        )

    def wait_for_settlement(self, timeout=None):
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout < 0
        ):
            raise ValueError("timeout must be finite and nonnegative or None")
        deadline = None if timeout is None else self.manager._monotonic() + timeout
        interval = min(self.poll_interval, 1.0)
        while True:
            # An explicit settlement wait must not consume its entire budget
            # behind a prior poll's throttle while terminal accounting catches up.
            with self.operation_lock:
                self._next_query_at = float("-inf")
                status = self.poll()
            if status.settled:
                return status
            remaining = None if deadline is None else deadline - self.manager._monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError(f"Slurm task settlement is unconfirmed: {self.execution_id}")
            time.sleep(interval if remaining is None else min(interval, remaining))
            interval = min(self.poll_interval, interval * 2.0)

    def request_cancel(self):
        with self.operation_lock:
            if not self.cancel_requested:
                self.cancel_requested = True
                self._next_query_at = float("-inf")

    def cancel(self):
        self.request_cancel()
        try:
            return self.wait_for_settlement(self.manager.config.cancel_timeout)
        except TimeoutError as e:
            raise TaskSettlementError("Slurm cancellation did not confirm allocation release", self.poll()) from e

    def terminate(self):
        return self.cancel()


class _SlurmTaskBackend(TaskLauncherSpec):
    """Trusted site adapter; construction does not contact or submit to Slurm.

    CPU and GPU tasks use native scheduler admission and the same allocation
    settlement contract. The CPU CJ does not reserve the task's GPUs.

    ``shared_filesystem`` and ``cgroup_enforced`` are explicit operator profile
    attestations, not remote discovery. Initialization probes local filesystem
    primitives and accounting; deployed qualification must verify both hosts.
    """

    launch_mode = "slurm"

    def __init__(
        self,
        *,
        workspace_path,
        shared_filesystem=False,
        cgroup_enforced=False,
        sbatch_directives=None,
        setup="",
        executables=None,
        submit_timeout=30.0,
        query_timeout=10.0,
        cancel_timeout=120.0,
        poll_interval=10.0,
        pending_timeout=600.0,
        python_path="/usr/bin/python3",
        cpu_cores=1,
        memory_mb=1024,
        gpu_count=0,
        environment=None,
        worker_listener_host=None,
    ):
        super().__init__()
        if shared_filesystem is not True or cgroup_enforced is not True:
            raise ValueError("Slurm tasks require coherent shared paths and enforced Slurm cgroup tracking")
        config = SlurmConfig(
            workspace_path=normalize_slurm_workspace_path(workspace_path),
            sandbox="none",
            python_path="/usr/bin/env",
            executables=normalize_slurm_executables(executables or {}),
            sbatch_directives=normalize_slurm_directives(sbatch_directives),
            setup=_require_string(setup, "setup", True),
            **{
                name: _require_positive_number(value, name)
                for name, value in (
                    ("submit_timeout", submit_timeout),
                    ("query_timeout", query_timeout),
                    ("cancel_timeout", cancel_timeout),
                    ("poll_interval", poll_interval),
                    ("pending_timeout", pending_timeout),
                )
            },
        )
        self.manager = SlurmTaskManager(config, self.logger)
        if not isinstance(python_path, str) or not os.path.isabs(python_path) or "\x00" in python_path:
            raise ValueError("Slurm task python_path must be an absolute path shared with compute hosts")
        self.python_path = python_path
        self.worker_listener_host = (
            None if worker_listener_host is None else _require_string(worker_listener_host, "worker_listener_host")
        )
        self.resources = TaskResourceRequest(cpu_cores, memory_mb, gpu_count)
        if self.resources.cpu_cores is None or self.resources.memory_mb is None:
            raise ValueError("Slurm task defaults require CPU and memory requests")
        if self.resources.cpu_cores is not None and not float(self.resources.cpu_cores).is_integer():
            raise ValueError("Slurm task default CPU request must be whole cores")
        self.environment = {"PATH": "/usr/local/bin:/usr/bin:/bin", **dict(environment or {})}
        TaskLaunchRequest("site", "site", "policy", "policy", (python_path,), environment=self.environment)
        self._check_environment(self.environment)
        self._site_config = config
        self._binding = None
        self._binding_lock = threading.Lock()
        self._initialization_lock = threading.Lock()

    def get_worker_listener_host(self):
        """Advertise this CJ's host to workers in independent Slurm allocations."""
        host = self.worker_listener_host or socket.getfqdn()
        if not isinstance(host, str) or not host or host == "0" or host != host.strip():
            raise TaskLauncherError("Slurm worker_listener_host must name a worker-reachable CJ host")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", host):
                raise TaskLauncherError("Slurm worker_listener_host must be a host name or IP address, without a port")
            loopback = host.rstrip(".").lower() == "localhost" or host.lower().endswith(".localhost")
        else:
            if address.is_unspecified:
                raise TaskLauncherError("Slurm worker_listener_host cannot advertise a wildcard address")
            loopback = address.is_loopback
        if loopback and self.worker_listener_host is None:
            raise TaskLauncherError(
                "Slurm could not discover a worker-reachable CJ host; configure worker_listener_host"
            )
        return host

    def validate_site_config(self, site_config):
        from nvflare.app_opt.job_launcher.slurm.task_policy import validate_slurm_site

        validate_slurm_site(self, site_config)

    def runtime_policy(self, meta, *, runtime=None):
        from nvflare.app_opt.job_launcher.slurm.launcher import effective_slurm_study_values, load_slurm_study_runtime

        config = self._site_config
        if runtime is None:
            runtime = load_slurm_study_runtime(config, meta.get("study"), self.logger)
        if runtime.secret_env or runtime.secret_mounts or runtime.datasets or runtime.container_image:
            raise ValueError("Slurm task shared-path profile does not forward study secrets or container mounts")
        sandbox, image, python_path, setup, directives = effective_slurm_study_values(
            replace(config, python_path=self.python_path), runtime
        )
        if sandbox != "none" or image:
            raise ValueError("Slurm task shared-path profile requires effective sandbox none")
        environment = {**self.environment, **runtime.env}
        self._check_environment(environment)
        return replace(config, setup=setup, sbatch_directives=directives), python_path, environment

    def handle_event(self, event_type, fl_ctx):
        if event_type != EventType.START_RUN:
            return
        self.bind_job(
            fl_ctx.get_job_id(),
            fl_ctx.get_identity_name(),
            fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT) or fl_ctx.get_workspace(),
            fl_ctx.get_prop(FLContextKey.JOB_META, {}) or {},
        )

    def bind_job(self, job_id, site_name, workspace, meta, *, runtime=None):
        from nvflare.app_opt.job_launcher.slurm.task_policy import task_resources

        if (
            not job_id
            or not site_name
            or os.path.realpath(workspace.get_root_dir()) != self._site_config.workspace_path
        ):
            raise ValueError("Slurm task START_RUN requires the owning CJ's job/site/shared workspace")
        resources = task_resources(self, meta, site_name)
        config, python_path, environment = self.runtime_policy(meta, runtime=runtime)
        custom_paths = (workspace.get_app_custom_dir(job_id), workspace.get_site_custom_dir())
        environment["PYTHONPATH"] = os.pathsep.join(
            dict.fromkeys(
                [path for path in custom_paths if path and os.path.isdir(path)]
                + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])
            )
        )
        binding = _JobBinding(
            job_id,
            site_name,
            os.path.realpath(workspace.get_run_dir(job_id)),
            resources,
            replace(config, pending_timeout=resources.pending_timeout),
            python_path,
            tuple(sorted(environment.items())),
        )
        with self._binding_lock:
            if self._binding is not None and self._binding != binding:
                raise ValueError("Slurm task launcher is already bound to a different CJ or resource policy")
            self._binding = binding
            self.manager.config = binding.config

    @staticmethod
    def _check_environment(environment):
        for name in environment:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise TaskLauncherError("Slurm task environment name must be a shell identifier")
            if name in _FORBIDDEN_ENV or name.startswith(("SLURM_", "SBATCH_", "SQUEUE_", "SACCT_", "SCANCEL_")):
                raise TaskLauncherError(f"Slurm task environment cannot override scheduler/bootstrap setting {name}")

    def _materialize_staged(self, request):
        binding = self._binding
        if binding is None:
            raise TaskLauncherError("staged Slurm tasks require the owning START_RUN resource binding")
        if (request.job_id, request.site_name) != (binding.job_id, binding.site_name):
            raise TaskLauncherError("Slurm task request does not belong to the bound CJ")
        path = request.argv[4]
        root = self._site_config.workspace_path
        if not os.path.isabs(path) or os.path.commonpath((root, os.path.realpath(path))) != root:
            raise TaskLauncherError("Slurm bootstrap must be inside the shared workspace")
        bootstrap = read_bootstrap(path)
        identity = bootstrap.identity
        if (identity.job_id, identity.site_name, identity.task_id, identity.attempt_id) != request.identity:
            raise TaskLauncherError("Slurm bootstrap identity differs from the staged request")
        session = bootstrap.cell_session
        if session is None:
            raise TaskLauncherError("Slurm staged task requires its owning CJ's live Cell session")
        credentials = session.get("credentials", {})
        if credentials or session.get("connection_security") == "mtls":
            if set(credentials) != set(CELL_SESSION_CREDENTIALS):
                raise TaskLauncherError("Slurm Cell session must provide the CA and both job credential pairs")
            certificate, key = job_cert_paths(binding.run_dir)
            for name, credential_path in credentials.items():
                if os.path.islink(credential_path):
                    raise TaskLauncherError("Slurm Cell credential path cannot be a symlink file")
                credential_path = os.path.realpath(credential_path)
                if os.path.commonpath((root, credential_path)) != root:
                    raise TaskLauncherError("Slurm Cell credentials must be reachable in the shared workspace")
                if name != "ca_cert" and credential_path != (key if name.endswith("_key") else certificate):
                    raise TaskLauncherError("Slurm Cell credentials must use this job's certificate and key")
                try:
                    with _open_regular(credential_path):
                        pass
                except (OSError, ValueError) as error:
                    raise TaskLauncherError("Slurm Cell credential path is not a shared regular file") from error
        artifact_root = os.path.realpath(bootstrap.artifact_root)
        if (
            os.path.realpath(bootstrap.workspace_root) != root
            or os.path.commonpath((root, artifact_root)) != root
            or os.stat(artifact_root).st_dev != os.stat(root).st_dev
        ):
            raise TaskLauncherError("Slurm artifacts/checkpoints must share the canonical workspace filesystem")
        from nvflare.app_opt.job_launcher.slurm.task_policy import resource_request

        if not request.resources.is_empty() and request.resources != resource_request(binding.resources):
            raise TaskLauncherError("Slurm staged resources differ from the owning START_RUN binding")
        return replace(
            request,
            argv=(binding.python_path, "-m", WORKER_MODULE, "--bootstrap", path),
            environment=dict(binding.environment),
            resources=resource_request(binding.resources),
            cwd=os.path.realpath(request.cwd) if request.cwd else None,
        )

    def launch_task(self, request: TaskLaunchRequest):
        if not isinstance(request, TaskLaunchRequest):
            raise TypeError("request must be TaskLaunchRequest")
        staged = (
            len(request.argv) == 7
            and request.argv[1:4] == ("-m", "nvflare.private.fed.app.client.task_worker_process", "--bootstrap")
            and request.argv[5] == "--parent_pid"
            and request.argv[6].isdigit()
        )
        if staged:
            request = self._materialize_staged(request)
        else:
            request = replace(
                request,
                resources=TaskResourceRequest(
                    request.resources.cpu_cores or self.resources.cpu_cores,
                    request.resources.memory_mb or self.resources.memory_mb,
                    request.resources.gpu_count,
                ),
            )
        # Native allocation ownership is required. A process launcher fallback
        # or a visible GPU on the CJ would not satisfy this contract.
        if request.cwd is None or os.path.realpath(request.cwd) != request.cwd or not os.path.isdir(request.cwd):
            raise TaskLauncherError("Slurm task cwd must be a canonical shared directory")
        if os.path.commonpath((self.manager.config.workspace_path, request.cwd)) != self.manager.config.workspace_path:
            raise TaskLauncherError("Slurm task cwd must be inside the site workspace")
        if not os.path.isabs(request.argv[0]):
            raise TaskLauncherError("Slurm task command must use an absolute executable shared with compute hosts")
        if "nvflare.private.fed.app.client.task_worker_process" in request.argv:
            raise TaskLauncherError(
                "Slurm tasks require a backend worker command without the Process parent-PID contract"
            )
        if request.resources.cpu_cores is not None and not float(request.resources.cpu_cores).is_integer():
            raise UnsupportedTaskResourceError("Slurm task CPU requests must be whole cores")
        self._check_environment(request.environment)
        with self._initialization_lock:
            if not self.manager._initialized:
                self.manager.initialize()
                # Keep command implementation shared; only task accounting differs.
                self.manager.adapter = SlurmTaskCliAdapter(self.manager.config.executables, os.getuid(), self.logger)
        return self.manager.launch_task(request, self._binding.resources if staged else None)

    @property
    def poll_interval(self):
        return self.manager.config.poll_interval

    max_attempts = 10000


class SlurmTaskLauncher(_SlurmTaskBackend):
    """CJ proxy; CP retains every accepted allocation through CJ loss."""

    cleanup_owned_by_parent = True

    def handle_event(self, event_type, fl_ctx):
        from nvflare.app_opt.job_launcher.slurm.task_service import SlurmTaskClient

        if event_type == EventType.START_RUN:
            if os.environ.get(SLURM_CHILD_PROCESS_ENV) != "1" or not os.environ.get("SLURM_JOB_ID", "").isdigit():
                raise TaskLauncherError("Slurm task proxy requires a Slurm Client Job allocation")
            super().handle_event(event_type, fl_ctx)
            self._client = SlurmTaskClient(
                fl_ctx.get_engine().get_cell(),
                fl_ctx.get_job_id(),
                fl_ctx.get_identity_name(),
                self.manager.config.query_timeout,
                self.poll_interval,
                self.manager.config.cancel_timeout,
            )
        elif event_type == EventType.END_RUN:
            client = getattr(self, "_client", None)
            if client is not None:
                client.close()

    def launch_task(self, request):
        client = getattr(self, "_client", None)
        if client is None:
            raise TaskLauncherError("Slurm task proxy has no START_RUN binding")
        return client.launch(request)
