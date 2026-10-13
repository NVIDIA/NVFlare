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

import os
import signal
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[5]
BUILDER = REPOSITORY / "nvflare/lighter/cc/image_builder"
UNIT_TESTS = REPOSITORY / "tests/unit_test/lighter/cc/image_builder"
sys.path[:0] = [str(BUILDER), str(UNIT_TESTS)]

from cvm.artifacts.acceptance import aggregate  # noqa: E402
from cvm.artifacts.bundle import load_signing_key, verify_bundle  # noqa: E402
from cvm.common.errors import BuildError, require  # noqa: E402
from cvm.common.io import write_json  # noqa: E402

SUITE_SHUTDOWN_SECONDS = 30


class _SuiteSignal(Exception):
    def __init__(self, signum):
        self.signum = signum


def _stop_suite(process):
    """Give the suite a bounded interrupt window so unittest cleanups can run."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=SUITE_SHUTDOWN_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


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
        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.signal(signum, interrupted)
            returncode = process.wait()
        except BaseException as error:
            _stop_suite(process)
            if isinstance(error, _SuiteSignal):
                name = signal.Signals(error.signum).name
                raise BuildError(f"{pattern} interrupted by {name}; cleanup was given a bounded grace period") from None
            raise
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
    require(returncode == 0, f"{pattern} failed; inspect its retained runner log")


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
