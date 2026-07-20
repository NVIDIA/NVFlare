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

import pytest

from nvflare.app_opt.job_launcher.slurm import scheduler_client
from nvflare.app_opt.job_launcher.slurm.config import CommandResult, LookupStatus, SlurmProtocolError
from nvflare.app_opt.job_launcher.slurm.scheduler_client import _scheduler_env, _SlurmCliAdapter


class Runner:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, argv, timeout):
        self.calls.append((argv, timeout))
        return self.results.pop(0)


def _result(stdout="", stderr="", returncode=0, timed_out=False):
    return CommandResult(returncode, stdout, stderr, timed_out=timed_out)


def _adapter(runner=None):
    executables = {name: f"/usr/bin/{name}" for name in ("sbatch", "squeue", "sacct", "scancel")}
    return _SlurmCliAdapter(executables, os.getuid(), logging.getLogger("test"), runner=runner)


@pytest.mark.parametrize(
    "stdout, expected",
    [
        ("42\n", ("42", None)),
        ("42;cluster-a\n", ("42", "cluster-a")),
        ("", (None, None)),
        ("42\n43\n", (None, None)),
        ("not-a-job\n", (None, None)),
    ],
)
def test_submit_parses_only_one_complete_identity(stdout, expected):
    adapter = _adapter(Runner(_result(stdout)))

    submission = adapter.submit(["sbatch"], 1)

    assert (submission.job_id, submission.cluster) == expected


def test_submit_treats_parsed_job_id_as_authoritative_despite_return_code():
    submission = _adapter(Runner(_result("42\n", returncode=1))).submit(["sbatch"], 1)

    assert submission.job_id == "42"


def test_scheduler_environment_removes_cli_overrides(monkeypatch):
    monkeypatch.setenv("SBATCH_CLUSTERS", "remote")
    monkeypatch.setenv("SLURM_CLUSTERS", "remote")
    monkeypatch.setenv("SLURM_HINT", "nomultithread")
    monkeypatch.setenv("SQUEUE_FORMAT", "bad")
    monkeypatch.setenv("KEEP_ME", "yes")

    env = _scheduler_env()

    assert env["KEEP_ME"] == "yes"
    assert env["LC_ALL"] == "C"
    assert "SBATCH_CLUSTERS" not in env
    assert "SLURM_CLUSTERS" not in env
    assert "SLURM_HINT" not in env
    assert "SQUEUE_FORMAT" not in env


def test_live_lookup_is_name_scoped_selects_id_and_verifies_ownership():
    runner = Runner()
    adapter = _adapter(runner)
    runner.results.append(_result(f"42|RUNNING|{adapter.uid}|marker|job-name\n"))

    result = adapter.active_by_id("42", "job-name", "marker", 1)

    assert result.status == LookupStatus.FOUND
    assert result.records[0].job_id == "42"
    argv = runner.calls[0][0]
    assert "--name=job-name" in argv
    assert "--states=all" not in argv
    assert not any(arg.startswith("--jobs") for arg in argv)


def test_live_lookup_empty_is_not_found():
    result = _adapter(Runner(_result())).active_by_id("42", "job-name", "marker", 1)

    assert result.status == LookupStatus.NOT_FOUND


@pytest.mark.parametrize(
    "row",
    [
        "bad-row\n",
        "42|RUNNING|not-a-uid|marker|job-name\n",
        "42|RUNNING|0|wrong|job-name\n",
        "42||0|marker|job-name\n",
    ],
)
def test_malformed_or_unowned_live_row_is_protocol_error(row):
    adapter = _adapter(Runner(_result(row)))
    row = row.replace("|0|", f"|{adapter.uid}|")
    adapter._runner.results[0] = _result(row)

    with pytest.raises(SlurmProtocolError):
        adapter.active_by_id("42", "job-name", "marker", 1)


def test_same_name_row_with_another_id_is_ignored(caplog):
    runner = Runner()
    adapter = _adapter(runner)
    rows = f"41|RUNNING|{adapter.uid}|old-marker|job-name\n" f"42|RUNNING|{adapter.uid}|new-marker|job-name\n"
    runner.results.extend((_result(rows), _result(rows)))

    with caplog.at_level(logging.WARNING):
        first = adapter.active_by_id("42", "job-name", "new-marker", 1)
        second = adapter.active_by_id("42", "job-name", "new-marker", 1)

    assert first.records[0].job_id == second.records[0].job_id == "42"
    assert sum("sharing name" in record.message for record in caplog.records) == 1


def test_id_accounting_rejects_a_different_allocation():
    runner = Runner()
    adapter = _adapter(runner)
    runner.results.append(_result(f"43|job-name|{adapter.user}|COMPLETED|0:0\n"))

    with pytest.raises(SlurmProtocolError, match="identity mismatch"):
        adapter.accounting_by_id("42", "job-name", 1)

    argv = runner.calls[0][0]
    assert "--jobs=42" in argv
    assert not any(arg.startswith(("--starttime=", "--endtime=", "--name=")) for arg in argv)


def test_id_accounting_parses_one_owned_allocation():
    adapter = _adapter()
    adapter._runner = Runner(_result(f"\n42|job-name|{adapter.user}|COMPLETED|7:9\n"))

    result = adapter.accounting_by_id("42", "job-name", 1)

    assert result.status == LookupStatus.FOUND
    assert result.records[0].job_id == "42"
    assert result.records[0].state == "COMPLETED"
    assert result.records[0].exit_status == 7
    assert result.records[0].exit_signal == 9


@pytest.mark.parametrize(
    "command, expected",
    [
        (_result(returncode=1, stderr="accounting unavailable"), LookupStatus.UNAVAILABLE),
        (_result(), LookupStatus.NOT_FOUND),
    ],
)
def test_id_accounting_distinguishes_unavailable_from_not_found(command, expected):
    result = _adapter(Runner(command)).accounting_by_id("42", "job-name", 1)

    assert result.status == expected


def test_id_accounting_rejects_malformed_exit_code():
    adapter = _adapter()
    adapter._runner = Runner(_result(f"42|job-name|{adapter.user}|FAILED|invalid\n"))

    with pytest.raises(SlurmProtocolError, match="numeric field"):
        adapter.accounting_by_id("42", "job-name", 1)


@pytest.mark.parametrize(
    "row",
    [
        "42|wrong-name|{user}|COMPLETED|0:0\n",
        "42|job-name|other-user|COMPLETED|0:0\n",
    ],
)
def test_id_accounting_rejects_reused_id_with_wrong_user_or_name(row):
    adapter = _adapter()
    adapter._runner = Runner(_result(row.format(user=adapter.user)))

    with pytest.raises(SlurmProtocolError, match="ownership"):
        adapter.accounting_by_id("42", "job-name", 1)


def test_scheduler_failure_is_unavailable_not_absence():
    adapter = _adapter(Runner(_result(returncode=1, stderr="controller down")))

    assert adapter.active_by_id("42", "job-name", "marker", 1).status == LookupStatus.UNAVAILABLE


def test_cancel_is_one_direct_request_after_manager_ownership_check():
    runner = Runner(_result())
    adapter = _adapter(runner)

    result = adapter.cancel("42", 1)

    assert result.status == LookupStatus.FOUND
    assert runner.calls[0][0] == ["/usr/bin/scancel", "-Q", "--me", "42"]


def test_cancel_failure_is_unavailable():
    runner = Runner(_result(returncode=1))
    adapter = _adapter(runner)

    result = adapter.cancel("42", 1)

    assert result.status == LookupStatus.UNAVAILABLE
    assert len(runner.calls) == 1


def test_cluster_suffix_cancel_is_one_direct_scoped_request():
    runner = Runner(_result())
    adapter = _adapter(runner)

    result = adapter.cancel_suffix("42", "cluster-a", 1)

    assert result.status == LookupStatus.FOUND
    assert runner.calls[0][0] == ["/usr/bin/scancel", "-M", "cluster-a", "-Q", "--me", "42"]


def test_accounting_probe_is_trivial():
    runner = Runner(_result())
    adapter = _adapter(runner)

    assert adapter.accounting_probe(1).available
    assert runner.calls[0][0] == [
        "/usr/bin/sacct",
        "-X",
        "-n",
        "--starttime=now",
        "--format=JobIDRaw",
    ]


def test_subprocess_runner_clamps_timeout_and_uses_scrubbed_environment(monkeypatch):
    captured = {}

    def run(argv, **kwargs):
        captured.update(kwargs)
        return scheduler_client.subprocess.CompletedProcess(argv, 0, b"42\n", b"warning\n")

    monkeypatch.setenv("SBATCH_CLUSTERS", "remote")
    monkeypatch.setattr(scheduler_client.subprocess, "run", run)
    adapter = _adapter()

    result = adapter._run(["sbatch"], 0)

    assert result.available
    assert result.stdout == "42\n"
    assert result.stderr == "warning\n"
    assert captured["capture_output"] is True
    assert captured["timeout"] == 0.001
    assert "SBATCH_CLUSTERS" not in captured["env"]


def test_subprocess_runner_discards_partial_output_on_timeout(monkeypatch):
    def run(argv, **kwargs):
        raise scheduler_client.subprocess.TimeoutExpired(argv, kwargs["timeout"], output=b"42\n")

    monkeypatch.setattr(scheduler_client.subprocess, "run", run)
    result = _adapter()._run(["sbatch"], 1)

    assert result.timed_out
    assert result.returncode is None
    assert result.stdout == ""


def test_subprocess_runner_reports_spawn_failure(monkeypatch):
    def run(argv, **kwargs):
        raise FileNotFoundError(argv[0])

    monkeypatch.setattr(scheduler_client.subprocess, "run", run)
    result = _adapter()._run(["missing-sbatch"], 1)

    assert result.returncode is None
    assert result.stderr == "FileNotFoundError"


def test_subprocess_runner_rejects_captured_stdout_over_limit(monkeypatch):
    monkeypatch.setattr(scheduler_client, "_MAX_STDOUT_BYTES", 4)
    monkeypatch.setattr(
        scheduler_client.subprocess,
        "run",
        lambda argv, **kwargs: scheduler_client.subprocess.CompletedProcess(argv, 0, b"12345", b""),
    )

    with pytest.raises(SlurmProtocolError, match="hard limit"):
        _adapter()._run(["sbatch"], 1)


def test_subprocess_runner_truncates_stderr_for_diagnostics(monkeypatch):
    monkeypatch.setattr(scheduler_client, "_MAX_STDERR_BYTES", 4)
    monkeypatch.setattr(
        scheduler_client.subprocess,
        "run",
        lambda argv, **kwargs: scheduler_client.subprocess.CompletedProcess(argv, 1, b"", b"12345"),
    )

    result = _adapter()._run(["sbatch"], 1)

    assert result.stderr == "1234\n...[stderr truncated]"
