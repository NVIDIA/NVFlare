#!/usr/bin/env python3
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

"""Run the complete official CVM acceptance suites and emit an exact report.

The executable follows the ``RUNNER BUNDLE REPORT`` acceptance-runner contract.
It needs an already sealed candidate vault and an isolated lab Trustee. Paths
and the site evidence signing key are supplied through the environment described
in BUILD_GUIDE.md; the offline bundle approval key is never passed to this runner.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[5]
BUILDER = REPOSITORY / "nvflare/lighter/cc/image_builder"
UNIT_TESTS = REPOSITORY / "tests/unit_test/lighter/cc/image_builder"
sys.path[:0] = [str(BUILDER), str(UNIT_TESTS)]

from cvm.artifacts.acceptance import aggregate  # noqa: E402
from cvm.artifacts.bundle import load_signing_key, verify_bundle  # noqa: E402
from cvm.common.errors import BuildError, require  # noqa: E402
from cvm.common.io import read_json, write_json  # noqa: E402
from cvm.host.launcher import (  # noqa: E402
    GRACEFUL_SHUTDOWN_SECONDS,
    SHUTDOWN_WAIT_SECONDS,
    _process_start,
    qmp_powerdown,
    qmp_socket_path,
    require_detached,
    runtime_state_path,
    shutdown,
)

SUITE_SHUTDOWN_SECONDS = SHUTDOWN_WAIT_SECONDS + 10
ORPHAN_TERMINATE_SECONDS = 30


class _SuiteSignal(Exception):
    def __init__(self, signum):
        self.signum = signum


def _same_process(pid, started):
    try:
        return _process_start(pid) == started
    except FileNotFoundError:
        return False


def _wait_process(pid, started, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _same_process(pid, started):
            return True
        time.sleep(0.2)
    return not _same_process(pid, started)


def _terminate_recorded_qemu(delivery, state):
    """Stop an orphan only when its protected runtime identity still matches."""
    require(state.get("vault_directory") == str(delivery.resolve()), "Acceptance QEMU state names another vault")
    pid = state.get("qemu_pid")
    started = state.get("qemu_start")
    require(type(pid) is int and type(started) is int, "Invalid acceptance QEMU runtime state")
    if not _same_process(pid, started):
        return
    try:
        qmp_powerdown(qmp_socket_path(delivery))
    except (OSError, ValueError, BuildError):
        pass
    if _wait_process(pid, started, GRACEFUL_SHUTDOWN_SECONDS):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if _wait_process(pid, started, ORPHAN_TERMINATE_SECONDS):
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    require(_wait_process(pid, started, 10), "Acceptance QEMU did not stop")


def _shutdown_vaults(output):
    """Stop every copied acceptance delivery, including a launcher-orphaned QEMU."""
    for delivery in sorted(output.glob("*/delivery")):
        if not (delivery / "vault.qcow2").is_file():
            continue
        state_path = runtime_state_path(delivery)
        state = read_json(state_path) if state_path.is_file() and not state_path.is_symlink() else None
        try:
            shutdown(delivery)
        except BuildError:
            if state:
                _terminate_recorded_qemu(delivery, state)
        require_detached(delivery)
        state_path.unlink(missing_ok=True)


def _stop_suite(process, output):
    """Stop the suite before scanning for resources it could otherwise recreate."""
    try:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            try:
                process.wait(timeout=SUITE_SHUTDOWN_SECONDS)
            except subprocess.TimeoutExpired:
                pass
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    _shutdown_vaults(output)


def _cleanup_suite_firewall(pid):
    """Remove only this hardware suite's table, tolerating an already absent table."""
    table = f"cvm_acceptance_{pid}"
    result = subprocess.run(["nft", "-j", "list", "tables"], capture_output=True, check=True, timeout=10)
    for entry in json.loads(result.stdout)["nftables"]:
        existing = entry.get("table", {})
        if existing.get("family") == "inet" and existing.get("name") == table:
            subprocess.run(["nft", "delete", "table", "inet", table], capture_output=True, check=True, timeout=10)
            break


def required_environment():
    names = (
        "CVM_ACCEPTANCE_VAULT",
        "CVM_ACCEPTANCE_RESULT_KEY",
        "CVM_LAB_DIRECTORY",
        "CVM_OTHER_RESOURCE",
    )
    values = {name: os.environ.get(name) for name in names}
    missing = [name for name, value in values.items() if not value]
    require(not missing, "Acceptance runner requires: " + ", ".join(missing))
    return values


def run_suite(pattern, environment, output):
    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "--catch",
        "-s",
        str(Path(__file__).parent),
        "-p",
        pattern,
        "-v",
    ]
    log_path = output / (pattern.removesuffix(".py") + ".log")
    previous_handlers = {}

    def interrupted(signum, _frame):
        raise _SuiteSignal(signum)

    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        failure = None
        cleanup_errors = []
        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.signal(signum, interrupted)
            returncode = process.wait()
            require(returncode == 0, f"{pattern} failed with exit status {returncode}; inspect {log_path}")
        except BaseException as error:
            failure = error
            try:
                _stop_suite(process, output)
            except Exception as cleanup_error:
                cleanup_errors.append(f"suite resources: {cleanup_error}")
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
            if pattern == "test_hardware.py":
                try:
                    _cleanup_suite_firewall(process.pid)
                except Exception as cleanup_error:
                    cleanup_errors.append(f"host firewall: {cleanup_error}")
    if isinstance(failure, _SuiteSignal):
        name = signal.Signals(failure.signum).name
        failure = BuildError(f"{pattern} interrupted by {name}; cleanup was given a bounded grace period")
    if cleanup_errors:
        detail = str(failure) if failure is not None else f"{pattern} completed"
        raise BuildError(f"{detail}; cleanup failed: {'; '.join(cleanup_errors)}; inspect {log_path}") from failure
    if failure is not None:
        raise failure


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    require(len(argv) == 2, "Usage: acceptance_runner.py BUNDLE REPORT")
    bundle, report = map(Path, argv)
    bundle = bundle.resolve()
    report = report.resolve()
    manifest = verify_bundle(bundle)
    require(manifest.get("production_ready") is True, "Acceptance requires a production-ready profile")
    require(not report.exists(), "Acceptance report already exists; choose a new output path")
    configured = required_environment()
    vault = Path(configured["CVM_ACCEPTANCE_VAULT"]).resolve()
    lab = Path(configured["CVM_LAB_DIRECTORY"]).resolve()
    signing_key = Path(configured["CVM_ACCEPTANCE_RESULT_KEY"]).resolve()
    require(vault.is_dir(), "CVM_ACCEPTANCE_VAULT must be a candidate vault directory")
    require((lab / "lab-kbs/admin.json").is_file(), "CVM_LAB_DIRECTORY does not contain a prepared lab Trustee")
    signer = load_signing_key(signing_key).public_key()

    configured_output = os.environ.get("CVM_ACCEPTANCE_OUTPUT")
    output = Path(configured_output).resolve() if configured_output else report.parent / (report.stem + "-evidence")
    require(not output.exists(), "CVM acceptance output already exists; use a new empty path")
    output.mkdir(parents=True, mode=0o700)

    pythonpath = [str(BUILDER), str(UNIT_TESTS)]
    if os.environ.get("PYTHONPATH"):
        pythonpath.append(os.environ["PYTHONPATH"])
    environment = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(pythonpath),
        CVM_BUNDLE=str(bundle),
        CVM_VAULT=str(vault),
        CVM_LAB_DIRECTORY=str(lab),
        CVM_ACCEPTANCE_RESULT_KEY=str(signing_key),
        CVM_HARDWARE_OUTPUT=str(output),
        CVM_ACCEPTANCE_OUTPUT=str(output),
        CVM_HTTP_TESTS="1",
        CVM_HARDWARE_TESTS="1",
        CVM_EXTENDED_AGENT="1",
        CVM_NETWORK_FAULTS="1",
        CVM_BACKEND_LOCAL="1",
    )
    if manifest["contract"]["gpu"] == "nvidia_cc":
        environment["CVM_GPU_HARDWARE_TESTS"] = "1"
    else:
        environment.pop("CVM_GPU_HARDWARE_TESTS", None)

    try:
        run_suite("test_http.py", environment, output)
        run_suite("test_hardware.py", environment, output)
        write_json(report, aggregate(bundle, [output], [signer]), mode=0o600)
    except BaseException:
        print(f"Acceptance evidence retained for diagnosis: {output}", file=sys.stderr)
        raise
    print(f"Acceptance report: {report}")
    print(f"Signed evidence: {output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"CVM acceptance failed: {error}", file=sys.stderr)
        sys.exit(1)
