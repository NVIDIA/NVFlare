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

import logging
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLMetaKey
from nvflare.apis.fl_exception import UnsafeComponentError
from nvflare.apis.job_launcher_spec import JobReturnCode
from nvflare.app_opt.job_launcher.slurm import manager as manager_module
from nvflare.app_opt.job_launcher.slurm.config import (
    SANDBOX_ROOT,
    CommandResult,
    JobResources,
    LaunchPlan,
    LookupStatus,
    QueryResult,
    SlurmConfig,
    SlurmLauncherError,
    SlurmProtocolError,
    SlurmRecord,
    SubmissionResult,
)
from nvflare.app_opt.job_launcher.slurm.manager import SlurmJobManager, _ensure_dir, _job_key
from nvflare.app_opt.job_launcher.slurm.scheduler_client import _SlurmCliAdapter
from nvflare.fuel.common.exit_codes import ProcessExitCode
from nvflare.private.fed.server.fed_server import FederatedServer


def _command(returncode=0, stdout="42\n", stderr="", timed_out=False):
    return CommandResult(returncode, stdout, stderr, timed_out=timed_out)


def _query(status, *records):
    return QueryResult(status, records)


class Clock:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value


class Adapter:
    def __init__(self):
        self.submission = SubmissionResult(_command(), "42")
        self.live = _query(LookupStatus.NOT_FOUND)
        self.accounting_id = _query(LookupStatus.NOT_FOUND)
        self.cancel_result = _query(LookupStatus.FOUND)
        self.version_result = _command(stdout="slurm 23.02.0\n")
        self.probes = [_command(stdout="")]
        self.calls = []

    @staticmethod
    def _take(value):
        return value.pop(0) if isinstance(value, list) else value

    def submit(self, argv, timeout):
        self.calls.append(("submit", argv))
        self.submitted_batch = Path(argv[-1]).read_text(encoding="utf-8")
        return self._take(self.submission)

    def active_by_id(self, job_id, job_name, marker, timeout):
        self.calls.append(("live", job_name, marker, job_id))
        return self._take(self.live)

    def accounting_by_id(self, job_id, job_name, timeout):
        self.calls.append(("accounting_id", job_id, job_name))
        return self._take(self.accounting_id)

    def cancel(self, job_id, timeout):
        self.calls.append(("cancel", job_id))
        return self._take(self.cancel_result)

    def cancel_suffix(self, job_id, suffix, timeout):
        self.calls.append(("cancel_suffix", job_id, suffix))
        return self.cancel_result

    def accounting_probe(self, timeout):
        self.calls.append(("probe", timeout))
        return self._take(self.probes)

    def version_probe(self, timeout):
        self.calls.append(("version", timeout))
        return self.version_result


def _record(job_id="42", state="RUNNING", **kwargs):
    return SlurmRecord(job_id, state, **kwargs)


def _config(tmp_path):
    return SlurmConfig(
        workspace_path=str(tmp_path),
        prepared_path=str(tmp_path / "prepared"),
        sandbox="none",
        python_path="/usr/bin/python3",
        executables={name: f"/usr/bin/{name}" for name in ("sbatch", "squeue", "sacct", "scancel")},
        poll_interval=0.01,
        pending_timeout=5,
    )


def _manager(tmp_path, adapter=None, monotonic=None):
    adapter = adapter or Adapter()
    monotonic = monotonic or Clock()
    manager = SlurmJobManager(
        _config(tmp_path),
        logging.getLogger("slurm-manager-test"),
        adapter=adapter,
        monotonic_clock=monotonic,
    )
    manager.jobs_dir = str(tmp_path / ".nvflare_slurm" / "jobs")
    for path in (tmp_path / ".nvflare_slurm", Path(manager.jobs_dir)):
        _ensure_dir(str(path))
    manager._initialized = True
    return manager


def _plan(tmp_path, pending_timeout=5, setup="", sandbox="none"):
    run_dir = tmp_path / "job-1"
    run_dir.mkdir(exist_ok=True)
    return LaunchPlan(
        job_id="job-1",
        run_dir=str(run_dir),
        exe_module="worker.module",
        module_args=("-n", "job-1"),
        resources=JobResources(pending_timeout=pending_timeout),
        directives={},
        sandbox=sandbox,
        image=None,
        setup=setup,
        study_env={},
        study_secret_env={},
        mounts=(),
        python_path="/usr/bin/python3",
        python_env="",
        forward_env=(),
    )


def _runtime_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    kit = workspace / "kit"
    (kit / "startup").mkdir(parents=True)
    (kit / "local").mkdir()
    (workspace / "startup").symlink_to("kit/startup")
    (workspace / "local").symlink_to("kit/local")
    return workspace


def _make_slurm_commands(path):
    path.mkdir(parents=True)
    commands = {}
    for name in ("sbatch", "squeue", "sacct", "scancel"):
        command = path / name
        command.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        command.chmod(0o700)
        commands[name] = str(command)
    return commands


def _runtime_config(workspace, executables):
    return SlurmConfig(
        workspace_path=str(workspace),
        prepared_path=str(workspace / "prepared"),
        sandbox="none",
        python_path="/usr/bin/python3",
        executables=executables,
    )


def _bootstrap_adapter(monkeypatch):
    adapter = MagicMock()
    adapter.version_probe.return_value = _command(stdout="slurm 23.02.5\n")
    adapter.accounting_probe.return_value = _command(stdout="")
    factory = MagicMock(return_value=adapter)
    monkeypatch.setattr(manager_module, "_SlurmCliAdapter", factory)
    return factory


def test_initialize_resolves_runtime_path_once(tmp_path, monkeypatch):
    workspace = _runtime_workspace(tmp_path)
    commands = _make_slurm_commands(tmp_path / "runtime-bin")
    monkeypatch.setenv("PATH", str(tmp_path / "runtime-bin"))
    factory = _bootstrap_adapter(monkeypatch)
    configured = {name: None for name in commands}
    manager = SlurmJobManager(_runtime_config(workspace, configured), logging.getLogger("runtime-resolution"))

    manager.initialize()
    manager.initialize()

    factory.assert_called_once()
    assert manager.config.executables == commands
    assert Path(manager.jobs_dir) == workspace / ".nvflare_slurm" / "jobs"


@pytest.mark.parametrize("configured", [None, "not_executable"])
def test_initialize_rejects_missing_or_non_executable_runtime_command(tmp_path, monkeypatch, configured):
    workspace = _runtime_workspace(tmp_path)
    monkeypatch.setenv("PATH", "")
    executables = {name: None for name in ("sbatch", "squeue", "sacct", "scancel")}
    if configured is not None:
        command = tmp_path / configured
        command.write_text("not executable", encoding="utf-8")
        executables["sbatch"] = str(command)
    manager = SlurmJobManager(_runtime_config(workspace, executables), logging.getLogger("runtime-resolution"))

    message = "not found on the parent runtime PATH" if configured is None else "not an executable regular file"
    with pytest.raises(UnsafeComponentError, match=message):
        manager.initialize()


def test_parent_restart_reresolves_cluster_managed_symlink(tmp_path, monkeypatch):
    workspace = _runtime_workspace(tmp_path)
    version_one = _make_slurm_commands(tmp_path / "slurm-23.02")
    version_two = _make_slurm_commands(tmp_path / "slurm-24.11")
    current = tmp_path / "current"
    current.symlink_to(tmp_path / "slurm-23.02", target_is_directory=True)
    configured = {name: str(current / name) for name in version_one}
    factory = _bootstrap_adapter(monkeypatch)
    config = _runtime_config(workspace, configured)

    first = SlurmJobManager(config, logging.getLogger("runtime-resolution-v1"))
    first.initialize()
    current.unlink()
    current.symlink_to(tmp_path / "slurm-24.11", target_is_directory=True)
    second = SlurmJobManager(config, logging.getLogger("runtime-resolution-v2"))
    second.initialize()

    assert first.config.executables == version_one
    assert second.config.executables == version_two
    assert factory.call_count == 2


def test_launch_returns_handle_with_deterministic_identity(tmp_path):
    adapter = Adapter()
    manager = _manager(tmp_path, adapter)

    handle = manager.launch(_plan(tmp_path))

    assert handle.job_id == "42"
    assert handle.nvflare_job_id == "job-1"
    assert manager._handles[handle.nvflare_job_id] is handle
    assert manager._marker(handle.nvflare_job_id) == "nvfl:job-1"
    assert Path(manager.jobs_dir) == tmp_path / ".nvflare_slurm" / "jobs"
    assert Path(handle.job_dir).name == _job_key("job-1")
    assert Path(handle.job_dir).parent == Path(manager.jobs_dir)
    assert Path(handle.job_dir, "batch.sh").is_file()
    assert not Path(handle.job_dir, SANDBOX_ROOT).exists()
    submit_argv = adapter.calls[0][1]
    assert f"--job-name=nvfl-{handle.job_key[:8]}" in submit_argv
    assert f"--output={_plan(tmp_path).run_dir}/slurm-%j.out" in submit_argv
    assert adapter.calls[0][0] == "submit"


def test_stale_job_artifacts_block_relaunch(tmp_path):
    manager = _manager(tmp_path)
    stale = Path(manager.jobs_dir, _job_key("job-1"))
    stale.mkdir()

    with pytest.raises(SlurmLauncherError, match="stale Slurm artifacts"):
        manager.launch(_plan(tmp_path))


@pytest.mark.parametrize(
    "submission",
    [
        SubmissionResult(_command(returncode=None, stdout="42\n", timed_out=True), "42"),
        SubmissionResult(_command(stdout="site plugin output\n42\n")),
        SubmissionResult(_command(returncode=1, stdout="", stderr="invalid time")),
        SubmissionResult(_command(returncode=None, stdout="")),
    ],
)
def test_submission_uncertainty_fails_and_cleans_artifacts(tmp_path, submission):
    adapter = Adapter()
    adapter.submission = submission
    manager = _manager(tmp_path, adapter)

    with pytest.raises(SlurmLauncherError, match="valid job ID"):
        manager.launch(_plan(tmp_path))

    assert not os.listdir(manager.jobs_dir)
    assert not manager._handles


def test_unparsed_accepted_job_cannot_start_after_artifact_cleanup(tmp_path):
    adapter = Adapter()
    adapter.submission = SubmissionResult(_command(stdout="site plugin output\n42\n"))
    manager = _manager(tmp_path, adapter)
    setup_marker = tmp_path / "setup-ran"

    with pytest.raises(SlurmLauncherError, match="valid job ID"):
        manager.launch(_plan(tmp_path, setup=f"touch {setup_marker}"))

    spooled_batch = tmp_path / "spooled-unparsed-job.sh"
    spooled_batch.write_text(adapter.submitted_batch, encoding="utf-8")
    spooled_batch.chmod(0o700)
    process = subprocess.run([str(spooled_batch)], capture_output=True, text=True)
    assert process.returncode != 0
    assert "secret file is unavailable" in process.stderr
    assert not setup_marker.exists()


def test_submit_exception_fails_and_cleans_artifacts(tmp_path):
    adapter = Adapter()
    manager = _manager(tmp_path, adapter)

    def fail_submit(argv, timeout):
        raise RuntimeError("connection lost")

    adapter.submit = fail_submit
    with pytest.raises(SlurmLauncherError, match="submission failed"):
        manager.launch(_plan(tmp_path))

    assert not os.listdir(manager.jobs_dir)
    assert not manager._handles


@pytest.mark.parametrize("cancel_error", [None, RuntimeError("controller unavailable")])
def test_cluster_suffix_is_cancelled_once_and_fails_launch(tmp_path, cancel_error):
    adapter = Adapter()
    adapter.submission = SubmissionResult(_command(stdout="42;remote\n"), "42", "remote")
    adapter.cancel_suffix = MagicMock(side_effect=cancel_error, return_value=adapter.cancel_result)
    manager = _manager(tmp_path, adapter)

    with pytest.raises(SlurmLauncherError, match="multi-cluster"):
        manager.launch(_plan(tmp_path))

    adapter.cancel_suffix.assert_called_once_with("42", "remote", timeout=10.0)
    assert not os.listdir(manager.jobs_dir)
    assert not manager._handles


def test_running_allocation_remains_unknown(tmp_path):
    adapter = Adapter()
    adapter.live = _query(LookupStatus.FOUND, _record())
    handle = _manager(tmp_path, adapter).launch(_plan(tmp_path))

    assert handle.poll() == JobReturnCode.UNKNOWN
    assert not any(call[0] == "accounting_id" for call in adapter.calls)


def test_handle_monitors_only_its_returned_id_when_job_name_is_shared(tmp_path):
    results = []
    calls = []

    def run_command(argv, timeout):
        calls.append(argv)
        return results.pop(0)

    scheduler = _SlurmCliAdapter(
        _config(tmp_path).executables,
        os.getuid(),
        logging.getLogger("slurm-shared-job-name-test"),
        runner=run_command,
    )
    manager = _manager(tmp_path, scheduler)
    job_name = f"nvfl-{_job_key('job-1')[:8]}"
    marker = "nvfl:job-1"
    results.extend(
        [
            _command(stdout="42\n"),
            _command(
                stdout=(
                    f"41|RUNNING|{scheduler.uid}|old-marker|{job_name}\n"
                    f"42|RUNNING|{scheduler.uid}|{marker}|{job_name}\n"
                )
            ),
        ]
    )

    handle = manager.launch(_plan(tmp_path))

    assert handle.job_id == "42"
    assert handle.poll() == JobReturnCode.UNKNOWN
    assert any(f"--name={job_name}" in argv for argv in calls)


def test_terminal_accounting_cleans_job_artifacts_and_leaves_generic_rc_file(tmp_path):
    adapter = Adapter()
    adapter.live = _query(LookupStatus.NOT_FOUND)
    adapter.accounting_id = _query(
        LookupStatus.FOUND,
        _record(state="COMPLETED", exit_status=0, exit_signal=0),
    )
    manager = _manager(tmp_path, adapter)
    plan = _plan(tmp_path)
    rc_file = Path(plan.run_dir, FLMetaKey.PROCESS_RC_FILE)
    rc_file.write_text("0\n", encoding="utf-8")
    handle = manager.launch(plan)

    assert handle.poll() == JobReturnCode.SUCCESS
    assert not os.path.exists(handle.job_dir)
    assert not manager._handles
    assert rc_file.read_text(encoding="utf-8") == "0\n"


def test_infrastructure_terminal_state_is_an_exception(tmp_path):
    adapter = Adapter()
    adapter.live = _query(LookupStatus.NOT_FOUND)
    adapter.accounting_id = _query(LookupStatus.FOUND, _record(state="TIMEOUT"))
    handle = _manager(tmp_path, adapter).launch(_plan(tmp_path))

    assert handle.poll() == ProcessExitCode.EXCEPTION


def test_terminal_cleanup_restores_access_to_pyxis_mount_directories(tmp_path):
    adapter = Adapter()
    adapter.live = _query(LookupStatus.NOT_FOUND)
    adapter.accounting_id = _query(LookupStatus.FOUND, _record(state="COMPLETED"))
    manager = _manager(tmp_path, adapter)
    handle = manager.launch(_plan(tmp_path, sandbox="pyxis"))
    sandbox_root = Path(handle.job_dir, SANDBOX_ROOT)
    for name in ("local", "startup", "job-1"):
        mount_dir = sandbox_root / name
        mount_dir.mkdir()
        os.chmod(mount_dir, 0)

    assert handle.poll() == JobReturnCode.SUCCESS
    assert not os.path.exists(handle.job_dir)


def test_terminal_squeue_row_moves_directly_to_accounting(tmp_path):
    adapter = Adapter()
    adapter.live = _query(LookupStatus.FOUND, _record(state="CANCELLED"))
    adapter.accounting_id = _query(
        LookupStatus.FOUND,
        _record(state="CANCELLED", exit_status=0, exit_signal=0),
    )
    manager = _manager(tmp_path, adapter)
    handle = manager.launch(_plan(tmp_path))

    assert handle.poll() == ProcessExitCode.EXCEPTION
    assert not os.path.exists(handle.job_dir)
    assert not any(call[0] == "cancel" for call in adapter.calls)


def test_clean_completion_after_user_abort_remains_aborted(tmp_path):
    adapter = Adapter()
    adapter.live = _query(LookupStatus.NOT_FOUND)
    adapter.accounting_id = _query(
        LookupStatus.FOUND,
        _record(state="COMPLETED", exit_status=0, exit_signal=0),
    )
    manager = _manager(tmp_path, adapter)
    handle = manager.launch(_plan(tmp_path))
    handle._request_cancel(user_abort=True)

    assert handle.poll() == JobReturnCode.ABORTED
    assert not os.path.exists(handle.job_dir)


def test_user_abort_is_the_only_cancelled_result_mapped_to_aborted(tmp_path):
    clock = Clock()
    adapter = Adapter()
    adapter.live = _query(LookupStatus.FOUND, _record())
    manager = _manager(tmp_path, adapter, monotonic=clock)
    handle = manager.launch(_plan(tmp_path))

    handle.terminate()
    adapter.live = _query(LookupStatus.NOT_FOUND)
    adapter.accounting_id = _query(
        LookupStatus.FOUND,
        _record(state="CANCELLED", exit_status=0, exit_signal=0),
    )
    clock.value = 7

    assert handle.poll() == JobReturnCode.ABORTED


def test_pending_timeout_starts_with_first_live_pending_observation(tmp_path):
    clock = Clock()
    adapter = Adapter()
    adapter.live = _query(LookupStatus.FOUND, _record(state="PENDING"))
    manager = _manager(tmp_path, adapter, monotonic=clock)
    handle = manager.launch(_plan(tmp_path, pending_timeout=5))

    assert handle.poll() == JobReturnCode.UNKNOWN
    assert not handle.cancel_requested
    clock.value = 6
    assert handle.poll() == JobReturnCode.UNKNOWN
    assert handle.cancel_requested
    assert not handle.user_abort
    assert any(call[0] == "cancel" for call in adapter.calls)


def test_requeued_job_is_refused_by_batch_guard(tmp_path):
    adapter = Adapter()
    adapter.live = [
        _query(LookupStatus.FOUND, _record(state="REQUEUED")),
        _query(LookupStatus.NOT_FOUND),
    ]
    adapter.accounting_id = _query(
        LookupStatus.FOUND,
        _record(state="FAILED", exit_status=101),
    )
    manager = _manager(tmp_path, adapter)
    handle = manager.launch(_plan(tmp_path))

    assert "SLURM_RESTART_COUNT" in adapter.submitted_batch
    assert handle.poll() == JobReturnCode.UNKNOWN
    assert not any(call[0] == "cancel" for call in adapter.calls)
    assert handle.poll() == JobReturnCode.EXECUTION_ERROR


def test_framework_termination_path_handles_job_without_slurm_system_end_sweep(tmp_path):
    adapter = Adapter()
    adapter.live = _query(LookupStatus.FOUND, _record())
    manager = _manager(tmp_path, adapter)
    handle = manager.launch(_plan(tmp_path))
    server = object.__new__(FederatedServer)
    server.engine = MagicMock()
    order = []

    def stop_all_jobs():
        handle.terminate()
        order.append("terminate")

    def fire_event(event_type, _fl_ctx):
        order.append(event_type)
        manager.shutdown()

    server.engine.stop_all_jobs.side_effect = stop_all_jobs
    server.engine.fire_event.side_effect = fire_event
    with patch("nvflare.private.fed.server.fed_server.BaseServer.fl_shutdown"):
        server.fl_shutdown()

    assert order == ["terminate", EventType.SYSTEM_END]
    assert manager._handles[handle.nvflare_job_id] is handle
    assert len([call for call in adapter.calls if call[0] == "cancel"]) == 1
    with pytest.raises(SlurmLauncherError, match="shutting down"):
        manager.launch(_plan(tmp_path))


def test_scheduler_outage_never_synthesizes_terminal_state(tmp_path):
    adapter = Adapter()
    adapter.live = _query(LookupStatus.UNAVAILABLE)
    manager = _manager(tmp_path, adapter)
    handle = manager.launch(_plan(tmp_path))

    assert handle.poll() == JobReturnCode.UNKNOWN
    assert manager._handles[handle.nvflare_job_id] is handle


def test_accounting_outage_does_not_consume_healthy_miss_budget(tmp_path):
    adapter = Adapter()
    adapter.live = _query(LookupStatus.NOT_FOUND)
    adapter.accounting_id = _query(LookupStatus.UNAVAILABLE)
    handle = _manager(tmp_path, adapter).launch(_plan(tmp_path))

    assert handle.poll() == JobReturnCode.UNKNOWN
    assert handle.accounting_misses == 0


def test_poll_protocol_error_leaves_handle_non_terminal(tmp_path):
    adapter = Adapter()
    manager = _manager(tmp_path, adapter)
    handle = manager.launch(_plan(tmp_path))
    adapter.active_by_id = MagicMock(side_effect=SlurmProtocolError("malformed squeue row"))

    assert handle.poll() == JobReturnCode.UNKNOWN
    assert handle.terminal_result is None


def test_wait_polls_until_accounting_reports_terminal(tmp_path):
    adapter = Adapter()
    adapter.live = [
        _query(LookupStatus.FOUND, _record()),
        _query(LookupStatus.NOT_FOUND),
    ]
    adapter.accounting_id = _query(LookupStatus.FOUND, _record(state="COMPLETED"))
    handle = _manager(tmp_path, adapter).launch(_plan(tmp_path))

    assert handle.wait() is None
    assert handle.terminal_result == JobReturnCode.SUCCESS


def test_five_spaced_healthy_accounting_misses_are_infrastructure_failure(tmp_path):
    clock = Clock()
    adapter = Adapter()
    adapter.live = _query(LookupStatus.NOT_FOUND)
    adapter.accounting_id = _query(LookupStatus.NOT_FOUND)
    manager = _manager(tmp_path, adapter, monotonic=clock)
    handle = manager.launch(_plan(tmp_path))

    for index in range(4):
        clock.value = index * 6
        assert handle.poll() == JobReturnCode.UNKNOWN
    clock.value = 24

    assert handle.poll() == ProcessExitCode.EXCEPTION
    assert not os.path.exists(handle.job_dir)
    assert not manager._handles


def test_duplicate_live_handle_is_rejected_in_memory(tmp_path):
    adapter = Adapter()
    manager = _manager(tmp_path, adapter)
    manager.launch(_plan(tmp_path))

    with pytest.raises(SlurmLauncherError, match="already has a live Slurm handle"):
        manager.launch(_plan(tmp_path))

    assert len([call for call in adapter.calls if call[0] == "submit"]) == 1


@pytest.mark.parametrize(
    "probes, succeeds",
    [
        ([_command(stdout="")], True),
        ([_command(returncode=1), _command(stdout="")], True),
        ([_command(returncode=1), _command(returncode=1), _command(returncode=1)], False),
    ],
)
def test_accounting_probe_is_required_and_briefly_retried(tmp_path, monkeypatch, probes, succeeds):
    adapter = Adapter()
    adapter.probes = probes
    manager = _manager(tmp_path, adapter)
    monkeypatch.setattr("nvflare.app_opt.job_launcher.slurm.manager.time.sleep", lambda _: None)

    if succeeds:
        manager._require_accounting()
    else:
        with pytest.raises(UnsafeComponentError, match="slurmdbd"):
            manager._require_accounting()


@pytest.mark.parametrize("version", ["slurm 23.02.0\n", "slurm 26.05.1\n"])
def test_supported_slurm_version_passes_bootstrap_check(tmp_path, version):
    adapter = Adapter()
    adapter.version_result = _command(stdout=version)

    _manager(tmp_path, adapter)._require_slurm_version()


@pytest.mark.parametrize("result", [_command(stdout="slurm 22.05.9\n"), _command(stdout="unexpected\n"), _command(1)])
def test_unsupported_or_unreadable_slurm_version_fails_bootstrap(tmp_path, result):
    adapter = Adapter()
    adapter.version_result = result

    with pytest.raises(UnsafeComponentError, match="23.02 or later"):
        _manager(tmp_path, adapter)._require_slurm_version()
