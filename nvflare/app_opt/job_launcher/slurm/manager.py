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
"""Slurm submission, monitoring, and cancellation."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import stat
import threading
import time
from dataclasses import replace
from typing import Callable, Optional

from nvflare.apis.fl_exception import UnsafeComponentError
from nvflare.apis.job_launcher_spec import JobHandleSpec, JobReturnCode
from nvflare.app_opt.job_launcher.slurm.batch import _render_batch_script, _render_secret_file, _submission_argv
from nvflare.app_opt.job_launcher.slurm.config import (
    _APPLICATION_TERMINAL_STATES,
    _INFRASTRUCTURE_TERMINAL_STATES,
    _PENDING_STATES,
    BATCH_FILE,
    CONTAINER_RESOLV_CONF,
    CONTROL_DIR,
    SANDBOX_ROOT,
    SECRET_FILE,
    BindMount,
    LaunchPlan,
    LookupStatus,
    SlurmConfig,
    SlurmLauncherError,
    SlurmProtocolError,
    SlurmRecord,
    resolve_slurm_parent_executables,
)
from nvflare.app_opt.job_launcher.slurm.scheduler_client import _command_diagnostic, _SlurmCliAdapter
from nvflare.fuel.common.exit_codes import ProcessExitCode

_HEALTHY_MISSES = 5
_ACCOUNTING_RETRY_INTERVAL = 6.0
_COMMAND_TIMEOUT = 10.0
_MIN_SLURM_VERSION = (23, 2)
_SLURM_VERSION_RE = re.compile(r"^slurm\s+(\d+)\.(\d+)", re.IGNORECASE)
_JOB_NAME_SITE_LENGTH = 32


def _job_key(job_id: str) -> str:
    return hashlib.sha256(job_id.encode("utf-8")).hexdigest()


def _ensure_dir(path: str) -> None:
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        if os.path.islink(path) or not os.path.isdir(path):
            raise UnsafeComponentError(f"expected Slurm control directory: {path}")


def _write_exclusive(path: str, data: bytes, mode: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, mode)
    try:
        with os.fdopen(fd, "wb") as stream:
            os.fchmod(stream.fileno(), mode)
            stream.write(data)
    except BaseException:
        try:
            os.unlink(path)
        finally:
            raise


def _cleanup_job_dir(job_dir: str) -> None:
    try:
        sandbox_root = os.path.join(job_dir, SANDBOX_ROOT)
        if os.path.isdir(sandbox_root) and not os.path.islink(sandbox_root):
            pending = [sandbox_root]
            while pending:
                path = pending.pop()
                os.chmod(path, 0o700)
                with os.scandir(path) as entries:
                    pending.extend(entry.path for entry in entries if entry.is_dir(follow_symlinks=False))
        shutil.rmtree(job_dir)
    except FileNotFoundError:
        pass


def _job_name(site_name: str, job_key: str) -> str:
    return f"nvfl-{site_name[:_JOB_NAME_SITE_LENGTH]}-{job_key[:8]}"


def _is_terminal(state: str) -> bool:
    return state in _APPLICATION_TERMINAL_STATES or state in _INFRASTRUCTURE_TERMINAL_STATES or state == "CANCELLED"


class SlurmJobManager:
    """Own scheduler access and live job handles for one NVFlare parent."""

    def __init__(
        self,
        config: SlurmConfig,
        logger: logging.Logger,
        adapter: Optional[_SlurmCliAdapter] = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ):
        self.config = config
        self.logger = logger
        self.adapter = adapter
        self._monotonic = monotonic_clock
        self.jobs_dir = None
        self._lock = threading.Lock()
        self._submission_gate = threading.Lock()
        self._handles = {}
        self._reject_launches = False
        self._initialized = False

    def _validate_workspace(self) -> str:
        workspace = self.config.workspace_path
        try:
            info = os.lstat(workspace)
        except OSError as e:
            raise UnsafeComponentError(f"cannot inspect workspace_path: {workspace}") from e
        if not stat.S_ISDIR(info.st_mode) or os.path.realpath(workspace) != workspace:
            raise UnsafeComponentError(f"workspace_path must be a canonical real directory: {workspace}")
        if info.st_uid != os.geteuid():
            raise UnsafeComponentError(f"workspace_path must be owned by current uid {os.geteuid()}: {workspace}")
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise UnsafeComponentError(f"workspace_path must not grant group/world permissions: {workspace}")
        for name in ("startup", "local"):
            path = os.path.join(workspace, name)
            if os.path.islink(path) or not os.path.isdir(path):
                raise UnsafeComponentError(f"runtime {name} must be a real directory: {path}")
        return workspace

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            workspace = self._validate_workspace()
            control = os.path.join(workspace, CONTROL_DIR)
            jobs_dir = os.path.join(control, "jobs")
            for path in (control, jobs_dir):
                _ensure_dir(path)
            if self.adapter is None:
                try:
                    executables = resolve_slurm_parent_executables(self.config.executables)
                except SlurmLauncherError as e:
                    raise UnsafeComponentError(str(e)) from e
                adapter = _SlurmCliAdapter(executables, os.getuid(), self.logger)
                self._require_slurm_version(adapter)
                self.config = replace(self.config, executables=executables)
                self.adapter = adapter
            self._require_accounting()
            self.jobs_dir = jobs_dir
            self._initialized = True

    def _require_slurm_version(self, adapter=None) -> None:
        message = "Slurm 23.02 or later is required"
        result = (adapter or self.adapter).version_probe(timeout=2.0)
        match = _SLURM_VERSION_RE.match(result.stdout.strip()) if result.available else None
        if match and tuple(map(int, match.groups())) >= _MIN_SLURM_VERSION:
            return
        detail = _command_diagnostic(result) if not result.available else f"unexpected version output={result.stdout!r}"
        self.logger.error("%s: %s", message, detail)
        raise UnsafeComponentError(message)

    def _require_accounting(self) -> None:
        message = "Slurm accounting (slurmdbd) is required"
        last_result = None
        for attempt in range(3):
            last_result = self.adapter.accounting_probe(timeout=2.0)
            if last_result.available:
                return
            if attempt < 2:
                time.sleep(0.5)
        self.logger.error("%s: %s", message, _command_diagnostic(last_result))
        raise UnsafeComponentError(message)

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise SlurmLauncherError("Slurm launcher has not completed SYSTEM_BOOTSTRAP")

    def _marker(self, nvflare_job_id: str) -> str:
        return f"nvfl:{nvflare_job_id}"

    def _prepare_job_dir(self, job_id: str, needs_sandbox_root: bool) -> tuple[str, str]:
        job_key = _job_key(job_id)
        job_dir = os.path.join(self.jobs_dir, job_key)
        try:
            os.mkdir(job_dir, 0o700)
        except FileExistsError as e:
            raise SlurmLauncherError(
                f"stale Slurm artifacts prevent relaunch of job '{job_id}'; "
                "verify that no allocation is using them before removal"
            ) from e
        if needs_sandbox_root:
            try:
                os.mkdir(os.path.join(job_dir, SANDBOX_ROOT), 0o700)
            except BaseException:
                shutil.rmtree(job_dir, ignore_errors=True)
                raise
        return job_key, job_dir

    def _write_job_files(self, plan: LaunchPlan, job_dir: str) -> None:
        if plan.sandbox != "none":
            workspace = self.config.workspace_path
            launcher_mounts = (
                BindMount(os.path.join(job_dir, SANDBOX_ROOT), workspace, "rw"),
                BindMount(plan.run_dir, plan.run_dir, "rw"),
                BindMount(
                    os.path.realpath(os.path.join(workspace, "startup")),
                    os.path.join(workspace, "startup"),
                    "ro",
                ),
                BindMount(
                    os.path.realpath(os.path.join(workspace, "local")),
                    os.path.join(workspace, "local"),
                    "ro",
                ),
            )
            if plan.sandbox == "apptainer":
                launcher_mounts += (
                    BindMount(
                        CONTAINER_RESOLV_CONF,
                        CONTAINER_RESOLV_CONF,
                        "ro",
                    ),
                )
            plan = replace(plan, mounts=launcher_mounts + plan.mounts)
        script, secrets = _render_batch_script(
            plan=plan,
            job_dir=job_dir,
            config=self.config,
        )
        _write_exclusive(
            os.path.join(job_dir, SECRET_FILE),
            _render_secret_file(secrets).encode("utf-8"),
            0o600,
        )
        _write_exclusive(os.path.join(job_dir, BATCH_FILE), script.encode("utf-8"), 0o700)

    def launch(self, plan: LaunchPlan) -> "SlurmJobHandle":
        self._require_initialized()
        with self._submission_gate:
            with self._lock:
                if self._reject_launches:
                    raise SlurmLauncherError("Slurm launcher is shutting down and rejects new launches")
                if plan.job_id in self._handles:
                    raise SlurmLauncherError(f"job '{plan.job_id}' already has a live Slurm handle")
            job_key, job_dir = self._prepare_job_dir(plan.job_id, plan.sandbox != "none")
            job_name = _job_name(plan.site_name, job_key)
            try:
                self._write_job_files(plan, job_dir)
                try:
                    submission = self.adapter.submit(
                        _submission_argv(
                            plan,
                            job_dir,
                            job_name,
                            self._marker(plan.job_id),
                            self.config,
                        ),
                        _COMMAND_TIMEOUT,
                    )
                except Exception as e:
                    raise SlurmLauncherError("sbatch submission failed") from e
                if submission.command.timed_out or not submission.job_id:
                    raise SlurmLauncherError(
                        f"sbatch did not return a valid job ID: {_command_diagnostic(submission.command)}"
                    )
                if submission.cluster:
                    try:
                        outcome = self.adapter.cancel_suffix(
                            submission.job_id,
                            submission.cluster,
                            timeout=_COMMAND_TIMEOUT,
                        ).status.value
                    except Exception as e:
                        outcome = type(e).__name__
                    self.logger.critical(
                        "sbatch returned unsupported cluster identity %s;%s; cancellation=%s",
                        submission.job_id,
                        submission.cluster,
                        outcome,
                    )
                    raise SlurmLauncherError("Slurm multi-cluster routing is unsupported")
                handle = SlurmJobHandle(
                    self,
                    plan.job_id,
                    job_name,
                    job_dir,
                    submission.job_id,
                    plan.resources.pending_timeout,
                    self.config.poll_interval,
                )
                with self._lock:
                    self._handles[plan.job_id] = handle
                return handle
            except BaseException:
                try:
                    _cleanup_job_dir(job_dir)
                except OSError as e:
                    raise SlurmLauncherError("failed to remove Slurm artifacts after launch failure") from e
                raise

    def _result_for(self, handle: "SlurmJobHandle", record: SlurmRecord) -> int:
        if record.state in _INFRASTRUCTURE_TERMINAL_STATES:
            return ProcessExitCode.EXCEPTION
        # Preserve user intent when cooperative abort completes before scancel wins the race.
        if handle.user_abort and record.state in {"CANCELLED", "COMPLETED"}:
            return JobReturnCode.ABORTED
        if record.state == "CANCELLED":
            return ProcessExitCode.EXCEPTION
        if record.exit_status or record.exit_signal:
            return JobReturnCode.EXECUTION_ERROR
        return JobReturnCode.SUCCESS if record.state == "COMPLETED" else JobReturnCode.EXECUTION_ERROR

    def _finish(self, handle: "SlurmJobHandle", result: int) -> int:
        if handle.terminal_result is not None:
            return handle.terminal_result
        try:
            _cleanup_job_dir(handle.job_dir)
        except OSError:
            self.logger.exception("failed to remove terminal Slurm job artifacts: %s", handle.job_dir)
        with self._lock:
            if self._handles.get(handle.nvflare_job_id) is handle:
                self._handles.pop(handle.nvflare_job_id)
        result = int(result)
        handle._set_terminal(result)
        return result

    def _poll_accounting(self, handle: "SlurmJobHandle") -> int:
        now = self._monotonic()
        if now - handle.accounting_last_query < _ACCOUNTING_RETRY_INTERVAL:
            return JobReturnCode.UNKNOWN
        handle.accounting_last_query = now
        result = self.adapter.accounting_by_id(
            handle.job_id,
            handle.job_name,
            timeout=_COMMAND_TIMEOUT,
        )
        if result.status == LookupStatus.UNAVAILABLE:
            return JobReturnCode.UNKNOWN
        if result.status == LookupStatus.NOT_FOUND:
            handle.accounting_misses += 1
            if handle.accounting_misses < _HEALTHY_MISSES:
                return JobReturnCode.UNKNOWN
            self.logger.critical(
                "Slurm accounting has no record after five healthy retries: job_id=%s",
                handle.job_id,
            )
            return self._finish(handle, ProcessExitCode.EXCEPTION)
        handle.accounting_misses = 0
        record = result.records[0]
        if _is_terminal(record.state):
            return self._finish(handle, self._result_for(handle, record))
        return JobReturnCode.UNKNOWN

    def _poll_handle(self, handle: "SlurmJobHandle") -> int:
        self._require_initialized()
        with handle.operation_lock:
            if handle.terminal_result is not None:
                return handle.terminal_result
            active = self.adapter.active_by_id(
                handle.job_id,
                handle.job_name,
                self._marker(handle.nvflare_job_id),
                timeout=_COMMAND_TIMEOUT,
            )
            if active.status == LookupStatus.UNAVAILABLE:
                return JobReturnCode.UNKNOWN
            if active.status == LookupStatus.NOT_FOUND:
                return self._poll_accounting(handle)
            record = active.records[0]
            handle.accounting_misses = 0
            if _is_terminal(record.state):
                return self._poll_accounting(handle)
            if handle.cancel_requested:
                self.adapter.cancel(handle.job_id, timeout=_COMMAND_TIMEOUT)
                return JobReturnCode.UNKNOWN
            if record.state in _PENDING_STATES:
                now = self._monotonic()
                if handle.pending_started_at is None:
                    handle.pending_started_at = now
                if now - handle.pending_started_at >= handle.pending_timeout:
                    self.logger.warning("Slurm job %s exceeded pending_timeout", handle.job_id)
                    handle._request_cancel()
            if handle.cancel_requested:
                self.adapter.cancel(handle.job_id, timeout=_COMMAND_TIMEOUT)
            return JobReturnCode.UNKNOWN

    def _abort_handle(self, handle: "SlurmJobHandle") -> None:
        self.logger.info("user abort requested for Slurm job %s", handle.job_id)
        handle._request_cancel(user_abort=True)
        try:
            self._poll_handle(handle)
        except SlurmProtocolError:
            self.logger.exception("Slurm cancellation is waiting for a valid scheduler response")

    def shutdown(self) -> None:
        if not self._initialized:
            return
        with self._submission_gate:
            with self._lock:
                self._reject_launches = True


class SlurmJobHandle(JobHandleSpec):
    """Current-parent state and lifecycle driver for one Slurm submission."""

    def __init__(
        self,
        manager: SlurmJobManager,
        nvflare_job_id: str,
        job_name: str,
        job_dir: str,
        job_id: str,
        pending_timeout: float,
        poll_interval: float,
    ):
        self.manager = manager
        self.nvflare_job_id = nvflare_job_id
        self.job_name = job_name
        self.job_dir = job_dir
        self.pending_timeout = pending_timeout
        self.poll_interval = poll_interval
        self.condition = threading.Condition()
        self.operation_lock = threading.Lock()
        self.job_id = job_id
        self.cancel_requested = False
        self.user_abort = False
        self.terminal_result = None
        self.pending_started_at = None
        self.accounting_last_query = float("-inf")
        self.accounting_misses = 0

    def _request_cancel(self, user_abort: bool = False) -> None:
        with self.condition:
            self.cancel_requested = True
            self.user_abort = self.user_abort or user_abort

    def _set_terminal(self, result: int) -> None:
        with self.condition:
            if self.terminal_result is None:
                self.terminal_result = result
            self.condition.notify_all()

    def terminate(self):
        if self.terminal_result is None:
            self.manager._abort_handle(self)

    def poll(self):
        try:
            return self.manager._poll_handle(self)
        except SlurmProtocolError:
            self.manager.logger.exception("Slurm job remains non-terminal after a scheduler protocol error")
            return JobReturnCode.UNKNOWN

    def wait(self):
        while self.poll() == JobReturnCode.UNKNOWN:
            with self.condition:
                if self.terminal_result is None:
                    self.condition.wait(timeout=self.poll_interval)
        return None
