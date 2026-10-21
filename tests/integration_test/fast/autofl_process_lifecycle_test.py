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


import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.autofl_test_utils import _load_runner
from tests.timing_utils import ThreadGroup, wait_for


@contextmanager
def _timeout_after_output(runner, log_path, marker):
    """Establish the real I/O precondition, then expire the runner's policy clock.

    The real clock keeps advancing after expiry so process termination grace
    periods still work. The bounded observer expires it even on readiness failure,
    allowing run_python's finally block to clean up its owned children.
    """
    real_time = runner.time
    expired_at = []

    def monotonic():
        return 0.0 if not expired_at else 100.0 + real_time.monotonic() - expired_at[0]

    def expire_when_ready():
        try:
            wait_for(lambda: log_path.exists() and marker in log_path.read_text(encoding="utf-8"))
        finally:
            expired_at.append(real_time.monotonic())

    runner.time = SimpleNamespace(monotonic=monotonic, sleep=real_time.sleep)
    try:
        with ThreadGroup() as workers:
            workers.start(expire_when_ready)
            yield
    finally:
        runner.time = real_time


def test_run_python_does_not_interpret_shell_metacharacters(tmp_path):
    runner = _load_runner()
    marker = tmp_path / "must-not-exist"
    literal = f"$(touch {marker})"

    rc, output, _, command = runner.run_python(
        sys.executable,
        ["-c", "import json, sys; print(json.dumps(sys.argv[1:]))", literal, "; touch ignored"],
        cwd=tmp_path,
        timeout=10,
        log_path=tmp_path / "run.log",
        env={},
    )

    assert rc == 0
    assert command[0] == os.path.abspath(sys.executable)
    assert command[1:] == ["-c", "import json, sys; print(json.dumps(sys.argv[1:]))", literal, "; touch ignored"]
    assert json.loads(output.strip()) == [literal, "; touch ignored"]
    assert not marker.exists()


def test_run_python_uses_only_the_explicit_child_environment(tmp_path, monkeypatch):
    runner = _load_runner()
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "host-secret")

    rc, output, _, _command = runner.run_python(
        sys.executable,
        [
            "-c",
            (
                "import json, os; "
                "print(json.dumps({'safe': os.environ.get('SAFE_VALUE'), "
                "'secret': os.environ.get('AWS_SECRET_ACCESS_KEY')}))"
            ),
        ],
        cwd=tmp_path,
        timeout=10,
        log_path=tmp_path / "run.log",
        env={"SAFE_VALUE": "forwarded"},
    )

    assert rc == 0
    assert json.loads(output.strip()) == {"safe": "forwarded", "secret": None}


def test_run_streams_output_before_timeout(tmp_path):
    runner = _load_runner()
    log_path = tmp_path / "run.log"

    with _timeout_after_output(runner, tmp_path / "run.log", "started"):
        rc, output, _, _command = runner.run_python(
            sys.executable,
            [
                "-c",
                "import time; print('started', flush=True); __import__('threading').Event().wait()",
            ],
            cwd=tmp_path,
            timeout=1,
            log_path=log_path,
            env={},
        )

    log_text = log_path.read_text(encoding="utf-8")
    assert rc == 124
    assert "started" in output
    assert "started" in log_text
    assert "TIMEOUT after 1s" in log_text


def test_run_streams_partial_line_before_timeout(tmp_path):
    runner = _load_runner()
    log_path = tmp_path / "run.log"

    with _timeout_after_output(runner, tmp_path / "run.log", "partial output"):
        rc, output, _, _command = runner.run_python(
            sys.executable,
            [
                "-c",
                "import sys, time; sys.stdout.write('partial output'); sys.stdout.flush(); __import__('threading').Event().wait()",
            ],
            cwd=tmp_path,
            timeout=1,
            log_path=log_path,
            env={},
        )

    assert rc == 124
    assert "partial output" in output
    assert "partial output" in log_path.read_text(encoding="utf-8")


@pytest.mark.skipif(os.name == "nt", reason="process-group cleanup uses POSIX process groups")
def test_run_terminates_inherited_stdout_descendant_after_leader_exits(tmp_path):
    runner = _load_runner()
    child_pid_path = tmp_path / "descendant.pid"
    child_code = (
        "import os, pathlib, time; "
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(os.getpid())); "
        "print('descendant started', flush=True); __import__('threading').Event().wait()"
    )
    parent_code = (
        "import subprocess, sys; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        "print('leader exits', flush=True)"
    )

    with _timeout_after_output(runner, tmp_path / "run.log", "descendant started"):
        rc, output, _, _command = runner.run_python(
            sys.executable,
            ["-c", parent_code],
            cwd=tmp_path,
            timeout=1,
            log_path=tmp_path / "run.log",
            env={},
        )

    assert rc == 124
    assert "leader exits" in output
    assert "descendant started" in tmp_path.joinpath("run.log").read_text(encoding="utf-8")
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    deadline = runner.time.monotonic() + 5
    while runner.time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        proc_stat = Path(f"/proc/{child_pid}/stat")
        if proc_stat.is_file() and proc_stat.read_text(encoding="utf-8").split()[2] == "Z":
            break
        runner.time.sleep(0.05)
    else:
        pytest.fail(f"descendant process {child_pid} survived process-group cleanup")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="escaped-process cleanup uses Linux /proc ownership")
@pytest.mark.parametrize("ignore_sigterm", [False, True], ids=["sigterm_honored", "sigterm_ignored"])
def test_run_terminates_detached_descendant_that_escapes_process_group(tmp_path, ignore_sigterm):
    runner = _load_runner()
    child_pid_path = tmp_path / "detached.pid"
    signal_setup = "signal.signal(signal.SIGTERM, signal.SIG_IGN); " if ignore_sigterm else ""
    child_code = (
        "import os, pathlib, signal, time; "
        f"{signal_setup}"
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(os.getpid())); "
        "print('detached descendant started', flush=True); __import__('threading').Event().wait()"
    )
    parent_code = (
        "import subprocess, sys; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}], start_new_session=True); "
        "print('leader exits', flush=True)"
    )

    with _timeout_after_output(runner, tmp_path / "run.log", "detached descendant started"):
        rc, output, _, _command = runner.run_python(
            sys.executable,
            ["-c", parent_code],
            cwd=tmp_path,
            timeout=2 if ignore_sigterm else 1,
            log_path=tmp_path / "run.log",
            env={},
        )

    assert rc == 124
    assert "leader exits" in output
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    deadline = runner.time.monotonic() + 5
    while runner.time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        proc_stat = Path(f"/proc/{child_pid}/stat")
        if proc_stat.is_file() and proc_stat.read_text(encoding="utf-8").split()[2] == "Z":
            break
        runner.time.sleep(0.05)
    else:
        pytest.fail(f"detached descendant process {child_pid} survived trial-token cleanup")


@pytest.mark.skipif(os.name == "nt", reason="process-group cleanup uses POSIX process groups")
@pytest.mark.parametrize("monitor_error", [KeyboardInterrupt(), RuntimeError("monitor failed")])
def test_run_terminates_child_process_group_when_monitor_raises(tmp_path, monkeypatch, monitor_error):
    runner = _load_runner()
    pid_path = tmp_path / "child.pid"

    def fail_monitor(_roots):
        deadline = runner.time.monotonic() + 5
        while runner.time.monotonic() < deadline:
            try:
                int(pid_path.read_text(encoding="utf-8"))
                break
            except (FileNotFoundError, ValueError):
                pass
            runner.time.sleep(0.01)
        else:
            pytest.fail("child PID was not published before the monitor deadline")
        raise monitor_error

    monkeypatch.setattr(runner, "simulator_stall_message", fail_monitor)
    with pytest.raises(type(monitor_error), match=str(monitor_error) if str(monitor_error) else None):
        runner.run_python(
            sys.executable,
            [
                "-c",
                f"import os, pathlib, time; pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid())); time.sleep(30)",
            ],
            cwd=tmp_path,
            timeout=0,
            log_path=tmp_path / "run.log",
            env={},
            simulator_stall_roots=[tmp_path / "simulation"],
            stall_check_interval=0,
        )

    child_pid = int(pid_path.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_run_keeps_complete_log_but_only_bounded_output_tail(tmp_path):
    runner = _load_runner()
    result_dir = tmp_path / "early-result"
    log_path = tmp_path / "run.log"
    payload_size = runner.MAX_CAPTURED_PROCESS_OUTPUT + 65536
    script = (
        f"print('Result can be found in : {result_dir}'); "
        "print('PermissionError'); print('[Errno 1] Operation not permitted'); print('socket.bind'); "
        f"print('x' * {payload_size})"
    )

    rc, output, _, _command = runner.run_python(
        sys.executable,
        ["-c", script],
        cwd=tmp_path,
        timeout=10,
        log_path=log_path,
        env={},
    )

    printed_result, socket_failure = runner.scan_run_log(log_path, tmp_path)
    assert rc == 0
    assert len(output.encode("utf-8")) <= runner.MAX_CAPTURED_PROCESS_OUTPUT
    assert log_path.stat().st_size > runner.MAX_CAPTURED_PROCESS_OUTPUT
    assert "Result can be found" not in output
    assert printed_result == result_dir.resolve()
    assert socket_failure is True


def test_campaign_workspace_lock_rejects_concurrent_same_job_lifecycle(tmp_path):
    runner = _load_runner()
    job = tmp_path / "job.py"
    job.write_text("print('job')\n", encoding="utf-8")

    with runner.locked_campaign_workspace(tmp_path, "evaluate"):
        # Lock rejection uses only the standard library, before optional YAML
        # imports are needed. Isolate the child from site initialization hooks.
        result = subprocess.run(
            [sys.executable, "-I", "-S", runner.__file__, "status", str(job)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

    assert result.returncode == 2
    assert "campaign workspace is already in use" in result.stderr


def test_run_job_collects_configured_sim_result_and_standard_nvflare_text_metric(tmp_path):
    runner = _load_runner()
    job = tmp_path / "job.py"
    job.write_text(
        f"""
import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="run")
args = parser.parse_args()
result = Path(os.environ["{runner.SIMULATOR_WORKSPACE_ROOT_ENV_VAR}"]) / args.name
server = result / "server"
server.mkdir(parents=True, exist_ok=True)
server.joinpath("log.txt").write_text("accuracy: 0.76\\n")
print(f"Result can be found in : {{result}}")
""".lstrip(),
        encoding="utf-8",
    )

    record = runner.run_job(
        runner.JobRun("candidate", [], "candidate"),
        python=sys.executable,
        job=job,
        cwd=tmp_path,
        help_text="--name NAME",
        fixed_args=[],
        base_args=[],
        output_root=tmp_path / "runs",
        timeout=10,
        simulator_no_progress_timeout=0,
        metrics=["accuracy"],
        config={
            "job": {},
            "environment": {
                "discovered": {
                    "name": "SimEnv",
                    "args": {
                        "workspace_root": {
                            "value": str(tmp_path / "simulation"),
                            "confidence": "high",
                        }
                    },
                }
            },
        },
    )

    assert record.status == "candidate"
    assert record.score == pytest.approx(0.76)
    assert record.metric_source == "text:log.txt:line=1"
    assert tmp_path.joinpath("runs/candidate/simulation/server/log.txt").is_file()
    assert tmp_path.joinpath("runs/candidate/run.log").is_file()
    assert not tmp_path.joinpath("simulation").exists()


def test_run_stops_on_nvflare_simulator_stall_log(tmp_path):
    runner = _load_runner()
    log_path = tmp_path / "run.log"
    sim_root = tmp_path / "simulation" / "autofl_candidate"
    server_log = sim_root / "server" / "log_fl.txt"
    server_log.parent.mkdir(parents=True)
    server_log.write_text(
        "SimulatorClientRunner - ERROR - run_client_thread error: RuntimeError: "
        "Failed to create connection to the child process in SimulatorClientRunner, timeout: 60.0\n",
        encoding="utf-8",
    )

    rc, output, runtime, _command = runner.run_python(
        sys.executable,
        [
            "-c",
            "import time; print('started', flush=True); time.sleep(30)",
        ],
        cwd=tmp_path,
        timeout=30,
        log_path=log_path,
        env={},
        simulator_stall_roots=[sim_root],
        stall_check_interval=0.01,
    )

    log_text = log_path.read_text(encoding="utf-8")
    assert rc == runner.SIMULATOR_STALL_EXIT_CODE
    assert runtime < 5
    assert "SIMULATOR_STALL:" in output
    assert "SIMULATOR_STALL:" in log_text


def test_run_stops_on_nvflare_simulator_no_progress_log(tmp_path):
    runner = _load_runner()
    log_path = tmp_path / "run.log"
    sim_root = tmp_path / "simulation" / "autofl_candidate"
    server_log = sim_root / "server" / "log_fl.txt"
    server_log.parent.mkdir(parents=True)
    server_log.write_text("Round 0 started\n", encoding="utf-8")

    rc, output, runtime, _command = runner.run_python(
        sys.executable,
        [
            "-c",
            "import time; print('started', flush=True); time.sleep(30)",
        ],
        cwd=tmp_path,
        timeout=30,
        log_path=log_path,
        env={},
        simulator_stall_roots=[sim_root],
        stall_check_interval=0.01,
        simulator_no_progress_timeout=1,
    )

    log_text = log_path.read_text(encoding="utf-8")
    assert rc == runner.SIMULATOR_STALL_EXIT_CODE
    assert runtime < 5
    assert "SIMULATOR_STALL: no simulator progress markers changed" in output
    assert "SIMULATOR_STALL: no simulator progress markers changed" in log_text


def test_run_stops_on_stale_partial_simulator_aggregation(tmp_path):
    runner = _load_runner()
    log_path = tmp_path / "run.log"
    sim_root = tmp_path / "simulation" / "autofl_candidate"
    server_log = sim_root / "server" / "log_fl.txt"
    site_log = sim_root / "site-1" / "log_fl.txt"
    server_log.parent.mkdir(parents=True)
    site_log.parent.mkdir(parents=True)
    server_log.write_text(
        "Round 0 started\n" "2026-06-25 06:32:33 - FedAvg - INFO - Aggregated 1/8 results\n",
        encoding="utf-8",
    )
    site_log.write_text("[site=site-1] round=0\n", encoding="utf-8")

    rc, output, runtime, _command = runner.run_python(
        sys.executable,
        [
            "-c",
            (
                "import pathlib, time; "
                f"path = pathlib.Path({str(site_log)!r}); "
                "time.sleep(0.2); "
                "path.write_text('[site=site-1] round=0\\n[site=site-2] round=0\\n'); "
                "time.sleep(30)"
            ),
        ],
        cwd=tmp_path,
        timeout=30,
        log_path=log_path,
        env={},
        simulator_stall_roots=[sim_root],
        stall_check_interval=0.01,
        simulator_no_progress_timeout=1,
    )

    log_text = log_path.read_text(encoding="utf-8")
    assert rc == runner.SIMULATOR_STALL_EXIT_CODE
    assert runtime < 5
    assert "SIMULATOR_STALL: partial simulator aggregation made no server-side progress" in output
    assert "Aggregated 1/8 results" in log_text


@pytest.mark.parametrize(
    ("metric", "mode", "baseline_score", "candidate_score"),
    [
        pytest.param("accuracy", "max", 0.5, 0.8, id="accuracy-max"),
        pytest.param("loss", "min", 0.5, 0.2, id="loss-min"),
    ],
)
def test_cli_lifecycle_runs_agent_code_candidate_end_to_end(tmp_path, metric, mode, baseline_score, candidate_score):
    repo_root = Path(__file__).parents[3]
    runner_path = repo_root / "skills" / "nvflare-autofl" / "scripts" / "run_job_campaign.py"
    job = tmp_path / "job.py"
    simulation_root = tmp_path / "simulation"
    job.write_text(
        f"""
import argparse
import json
import os
from pathlib import Path
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe as ImportedFedAvgRecipe
from nvflare.recipe import SimEnv as ImportedSimEnv

SCORE = {baseline_score}

class FakeFedAvgRecipe:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

class FakeSimEnv:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

ImportedFedAvgRecipe = FakeFedAvgRecipe
ImportedSimEnv = FakeSimEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="run")
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument("--n_clients", type=int, default=2)
    args = parser.parse_args()
    ImportedFedAvgRecipe(
        model=object(),
        num_rounds=args.num_rounds,
        min_clients=args.n_clients,
        key_metric={metric!r},
        key_metric_mode={mode!r},
    )
    ImportedSimEnv(num_clients=args.n_clients, workspace_root={str(simulation_root)!r})
    result = Path(os.environ["NVFLARE_SIMULATOR_WORKSPACE_ROOT"]) / args.name
    result.mkdir(parents=True, exist_ok=True)
    result.joinpath("metrics_summary.json").write_text(json.dumps({{{metric!r}: SCORE}}))
    print(f"Result can be found in : {{result.resolve()}}")

if __name__ == "__main__":
    main()
""".lstrip(),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(repo_root), env.get("PYTHONPATH")]))

    subprocess.run(
        [
            sys.executable,
            str(runner_path),
            "initialize",
            str(job),
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(runner_path),
            "prepare",
            str(job),
            "--name",
            "code_candidate",
            "--hypothesis",
            "raise the reported score through a source change",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    draft_job = tmp_path / ".nvflare" / "autofl" / "candidates" / "code_candidate" / "source" / "job.py"
    draft_job.write_text(
        draft_job.read_text(encoding="utf-8").replace(f"SCORE = {baseline_score}", f"SCORE = {candidate_score}"),
        encoding="utf-8",
    )
    subprocess.run(
        [sys.executable, str(runner_path), "evaluate", str(job)],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    runner = _load_runner()
    records = runner.load_results(tmp_path / "results.tsv")
    config = runner.read_yaml(tmp_path / "autofl.yaml")
    state = json.loads(tmp_path.joinpath(".nvflare/autofl/campaign_state.json").read_text(encoding="utf-8"))
    assert [(record.status, record.score) for record in records] == [
        ("baseline", baseline_score),
        ("keep", candidate_score),
    ]
    assert f"SCORE = {candidate_score}" in job.read_text(encoding="utf-8")
    assert config["objective"]["mode"] == mode
    assert state["mode"] == mode
    assert state["improvement"] == pytest.approx(abs(candidate_score - baseline_score))
    manifest = json.loads(
        tmp_path.joinpath(".nvflare/autofl/candidates/code_candidate/candidate_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "keep"
    assert manifest["changed_files"] == ["job.py"]


pytestmark = pytest.mark.timeout(120)
