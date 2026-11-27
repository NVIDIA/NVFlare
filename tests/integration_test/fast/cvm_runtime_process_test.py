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

"""Exercise CVM runtime failure paths in isolated Linux interpreters."""

import json
import os
import signal
import subprocess
import sys
import textwrap

import pytest

from tests.cvm_builder_test_utils import BUILDER, builder_environment
from tests.process_test_utils import run_owned_process
from tests.timing_utils import PROCESS_STARTUP_TIMEOUT

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="CVM Builder requires Linux runtime interfaces")

_BOOTSTRAP_PROGRAM = """
import os, signal, sys, time
from cvm.common.errors import StartupPhaseError
from cvm.runtime import bootstrap
sys.argv = ["bootstrap", "bootstrap"]
def fail():
    global failure_started
    error = StartupPhaseError(PHASE)
    failure_started = time.monotonic()
    raise error from RuntimeError("private-token-content")
bootstrap.bootstrap = fail
bootstrap.emit = lambda event: None
NOTIFY
bootstrap.main()
"""


@pytest.fixture
def builder_env():
    return builder_environment(PYTHONPATH=str(BUILDER))


@pytest.mark.parametrize("phase", ["attestation", "vault_identity"])
def test_phase_reaches_stderr_before_immediate_watchdog_kill(builder_env, phase):
    code = _BOOTSTRAP_PROGRAM.replace("PHASE", repr(phase)).replace(
        "NOTIFY", "bootstrap.notify = lambda message: os.kill(os.getpid(), signal.SIGKILL)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=BUILDER,
        env=builder_env,
        capture_output=True,
        text=True,
        timeout=PROCESS_STARTUP_TIMEOUT,
    )
    assert result.returncode == -signal.SIGKILL
    assert result.stderr == f"CVM bootstrap failed [phase={phase}]; PID 1 will power off\n"
    assert "private-token-content" not in result.stderr


@pytest.mark.parametrize("notify_fails", [False, True])
def test_full_stderr_cannot_delay_shutdown_even_if_notify_fails(builder_env, notify_fails):
    read_fd, write_fd = os.pipe()
    try:
        os.set_blocking(write_fd, False)
        with pytest.raises(BlockingIOError):
            while True:
                os.write(write_fd, b"x" * 4096)
        os.set_blocking(write_fd, True)
        notify = """def notify(message):
    elapsed = time.monotonic() - failure_started
    assert message == "WATCHDOG=trigger"
    assert os.get_blocking(2), "stderr flags were not restored"
    print("shutdown requested", elapsed, flush=True)
    if FAILS:
        raise OSError("notify unavailable")
bootstrap.notify = notify""".replace(
            "FAILS", repr(notify_fails)
        )
        code = _BOOTSTRAP_PROGRAM.replace("PHASE", repr("attestation")).replace("NOTIFY", notify)
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=BUILDER,
            env=builder_env,
            stderr=write_fd,
            stdout=subprocess.PIPE,
            text=True,
            timeout=PROCESS_STARTUP_TIMEOUT,
        )
        assert result.returncode == 1
        label, _, elapsed = result.stdout.strip().rpartition(" ")
        assert label == "shutdown requested", result.stdout
        assert float(elapsed) < 5.0, f"bootstrap failure took {elapsed} seconds to reach notify()"
        assert os.get_blocking(write_fd)
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_flush_waits_until_prior_records_are_written(builder_env):
    # The production worker is intentionally process-long. Exercise it in
    # a child so the test cannot leave a daemon consuming the suite's
    # shared audit queue after its mocks are restored.
    program = textwrap.dedent(
        """
        from unittest.mock import patch
        from cvm.runtime import audit
        with patch.object(audit.Path, 'exists', return_value=False), \
             patch.object(audit, 'read_json', return_value={
                 'build_id': 'build', 'attestation_policy_id': 'policy'
             }), \
             patch.object(audit, 'append'):
            audit.emit('quarantine')
            audit.emit_diagnostic('gpu_quarantine_terminal')
            assert audit.flush(1)
        """
    )
    result = run_owned_process(
        [sys.executable, "-c", program], cwd=BUILDER, env=builder_env, timeout=PROCESS_STARTUP_TIMEOUT
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert [line.split("=", 1)[0] for line in lines] == ["CVM_ATTESTATION", "CVM_DIAGNOSTIC"]
    assert json.loads(lines[0].split("=", 1)[1])["decision"] == "quarantine"
    assert json.loads(lines[1].split("=", 1)[1])["code"] == "gpu_quarantine_terminal"


def test_stalled_audit_cannot_delay_quarantine_or_denial(builder_env):
    # Isolate the intentionally stalled daemon writer from the test runner.
    program = textwrap.dedent(
        """
        import subprocess, sys, tempfile, threading, time
        from pathlib import Path
        from unittest.mock import patch
        from cvm.common.errors import BuildError
        from cvm.runtime import audit, bootstrap, supervisor
        entered, release = threading.Event(), threading.Event()
        def stuck(path, line):
            entered.set()
            release.wait()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(audit, 'read_json', return_value={'build_id': 'test', 'attestation_policy_id': 'default'}), \
             patch.object(audit.Path, 'exists', return_value=False), \
             patch.object(audit, 'append', side_effect=stuck):
            audit.emit('allow')
            workload = None
            def stop(argv, **kwargs):
                if argv[:2] == ['systemctl', 'stop']:
                    workload.terminate()
                    workload.wait(timeout=1)
            try:
                assert entered.wait(2)
                writer = audit._writer
                workload = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
                started = time.monotonic()
                with patch.object(supervisor, 'run', side_effect=stop), \
                     patch.object(supervisor, 'revoke_vault'), \
                     patch.object(supervisor, 'watchdog') as watchdog, \
                     patch.object(supervisor, 'QUARANTINE_WINDOW_SECONDS', 0):
                    try:
                        supervisor.quarantine({}, ['cvm_app.service'], Path(directory))
                    except BuildError:
                        pass
                    else:
                        raise AssertionError('Denial must fail closed')
                    watchdog.assert_called_once_with(supervisor.REVOCATION_TIMEOUT_SECONDS)
                assert workload.poll() is not None
                with patch.object(sys, 'argv', ['bootstrap', 'bootstrap']), \
                     patch.object(bootstrap, 'bootstrap', side_effect=BuildError('denied')), \
                     patch.object(bootstrap, 'notify') as notify:
                    try:
                        bootstrap.main()
                    except SystemExit:
                        pass
                    else:
                        raise AssertionError('Must exit after denial')
                    notify.assert_called_once_with('WATCHDOG=trigger')
                for _ in range(100):
                    audit.emit('deny')
                    audit.emit_diagnostic('gpu_readiness_revoke_failed')
                assert audit._writer is writer and audit._records.qsize() <= 8
                assert time.monotonic() - started < 2
            finally:
                try:
                    if workload is not None:
                        workload.kill() if workload.poll() is None else None
                        workload.wait(timeout=1)
                finally:
                    release.set()
                    # Drain under the mocks, before interpreter shutdown can
                    # race the daemon's buffered stdout writes. The queue can
                    # be full, so wait for room for this test-only barrier.
                    drained = threading.Event()
                    audit._records.put(audit._Barrier(drained), timeout=2)
                    assert drained.wait(2), 'Audit writer did not finish test cleanup'
        """
    )
    result = run_owned_process(
        [sys.executable, "-c", program], cwd=BUILDER, env=builder_env, timeout=PROCESS_STARTUP_TIMEOUT
    )
    assert result.returncode == 0, result.stderr


def test_timed_out_reopen_leaves_no_running_descendant(builder_env, tmp_path):
    pidfile = tmp_path / "descendant.pid"
    program = textwrap.dedent(
        r"""
        import os, subprocess, sys, time
        from pathlib import Path
        from unittest.mock import patch
        from cvm.common.errors import BuildError
        from cvm.runtime import supervisor

        spawn = subprocess.Popen
        pidfile = Path(sys.argv[1])
        startup_timeout = float(sys.argv[2])
        reopen = (
            "import subprocess, sys, time\n"
            "from pathlib import Path\n"
            "pidfile = Path(sys.argv[1])\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "pending = pidfile.with_suffix('.tmp')\n"
            "pending.write_text(str(child.pid))\n"
            "pending.replace(pidfile)\n"
            "time.sleep(60)\n"
        )

        def start_reopen(argv, **kwargs):
            process = spawn([sys.executable, "-c", reopen, str(pidfile)], **kwargs)
            # Start the reopen deadline after the process group is established.
            deadline = time.monotonic() + startup_timeout
            while not pidfile.exists():
                assert process.poll() is None, "reopen child exited before publishing the descendant PID"
                assert time.monotonic() < deadline, "reopen child did not publish the descendant PID"
                time.sleep(0.01)
            return process

        with patch.object(supervisor.subprocess, "Popen", side_effect=start_reopen):
            try:
                supervisor.run_reopen(2, dict(os.environ))
            except BuildError as error:
                assert "TimeoutExpired" in str(error), error
            else:
                raise AssertionError("reopen did not time out")

        pid = int(pidfile.read_text())
        deadline = time.monotonic() + 5
        while True:
            try:
                status = Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[0]
            except (FileNotFoundError, ProcessLookupError):
                break
            if status in ("Z", "X"):
                break
            assert time.monotonic() < deadline, f"reopen descendant {pid} survived cleanup"
            time.sleep(0.01)
        """
    )
    result = run_owned_process(
        [sys.executable, "-c", program, str(pidfile), str(PROCESS_STARTUP_TIMEOUT)],
        cwd=BUILDER,
        env=builder_env,
        timeout=2 * PROCESS_STARTUP_TIMEOUT + 10,
    )
    assert result.returncode == 0, result.stderr
