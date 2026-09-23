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

"""Exercise the real cluster-side verification entrypoint without a cluster."""

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.unit_test.lighter.cc_provision.impl.deployment_guards_test import require_coco_bash

ROOT = Path(__file__).resolve().parents[5] / "examples/devops/coco"
RUNTIME = "kata-qemu-nvidia-gpu-snp"
NVFLARE_COMMAND = ["/opt/nvflare/startup/sub_start.sh", "--once", "--verify"]
READ_DENIAL = "ReadStreamRequest is blocked by policy"
EXEC_DENIAL = "ExecProcessRequest is blocked by policy"
SECRET = "fixture-sensitive-startup-output"

FAKE_KUBECTL = r"""
import json
import os
import sys
from pathlib import Path

config = json.loads(Path(os.environ["FAKE_KUBECTL_CONFIG"]).read_text())
args = sys.argv[1:]
with open(os.environ["FAKE_KUBECTL_CALLS"], "a") as calls:
    calls.write(json.dumps(args) + "\n")
operation = args[0]
if operation == "create":
    assert "--dry-run=client" in args and "--validate=false" in args
    print(json.dumps(config["expected"]))
elif operation == "wait":
    assert "--for=condition=Ready" in args and "--timeout=2m" in args
    sys.exit(config["wait_status"])
elif operation == "get":
    if args[-1] == "json":
        print(json.dumps(config["live"]))
    else:
        assert args[-1] == "wide"
        print("fixture-pod Running")
elif operation == "logs":
    assert "--limit-bytes=1" in args and "--request-timeout=30s" in args
    sys.stdout.write(config["log_stdout"])
    sys.stderr.write(config["log_stderr"])
    sys.exit(config["log_status"])
elif operation == "exec":
    assert args[-2:] == ["--", "/coco-app"]
    sys.stdout.write(config["exec_stdout"])
    sys.stderr.write(config["exec_stderr"])
    sys.exit(config["exec_status"])
else:
    raise AssertionError("unexpected kubectl operation")
"""


@pytest.fixture
def run_verifier(tmp_path):
    bash = require_coco_bash()
    for command in ("realpath", "sha256sum", "awk", "wc", "grep", "mktemp", "rm"):
        if not shutil.which(command):
            pytest.skip(f"CoCo entrypoint test requires {command} on PATH")

    script = tmp_path / "coco/70-verify-running-workload.sh"
    script.parent.mkdir()
    shutil.copyfile(ROOT / "coco/70-verify-running-workload.sh", script)
    library = script.parent / "lib/common.sh"
    library.parent.mkdir()
    library.write_text(
        f"RUNTIME_CLASS={RUNTIME}\n"
        "die() { printf '%s\\n' \"$*\" >&2; exit 1; }\n"
        'need_file() { [[ -f "$1" ]] || die "missing fixture file"; }\n'
        'need_cmd() { command -v "$1" >/dev/null || die "missing command"; }\n'
        'kctl() { kubectl "$@"; }\n'
    )
    binaries = tmp_path / "bin"
    binaries.mkdir()
    kubectl = binaries / "kubectl"
    kubectl.write_text(f"#!{sys.executable}\n" + FAKE_KUBECTL)
    kubectl.chmod(0o700)
    # Embedded Python runs through the same project interpreter as pytest.
    (binaries / "python3").symlink_to(sys.executable)
    temporary_files = tmp_path / "tmp"
    temporary_files.mkdir()
    config_file = tmp_path / "kubectl.json"
    calls_file = tmp_path / "calls.jsonl"
    handoff = tmp_path / "fixture-pod.yaml"
    environment = {
        **os.environ,
        "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
        "TMPDIR": str(temporary_files),
        "FAKE_KUBECTL_CONFIG": str(config_file),
        "FAKE_KUBECTL_CALLS": str(calls_file),
    }

    def run(command=None, mutate_live=None, expected_hash=None, **overrides):
        expected = {
            "metadata": {
                "name": "fixture-pod",
                "namespace": "fixture-namespace",
                "annotations": {"io.katacontainers.config.hypervisor.cc_init_data": "authenticated-init-data"},
            },
            "spec": {
                "runtimeClassName": RUNTIME,
                "containers": [
                    {"name": "workload", "image": "fixture@sha256:" + "a" * 64, "command": command or NVFLARE_COMMAND}
                ],
            },
        }
        live = copy.deepcopy(expected)
        live["status"] = {"phase": "Running", "containerStatuses": [{"ready": True, "restartCount": 0}]}
        if mutate_live:
            mutate_live(live)
        config = {
            "expected": expected,
            "live": live,
            "wait_status": 0,
            "log_stdout": "",
            "log_stderr": "",
            "log_status": 0,
            "exec_stdout": "",
            "exec_stderr": EXEC_DENIAL,
            "exec_status": 1,
            **overrides,
        }
        config_file.write_text(json.dumps(config))
        handoff.write_text(json.dumps(expected))
        expected_hash = expected_hash or hashlib.sha256(handoff.read_bytes()).hexdigest()
        result = subprocess.run(
            [bash, str(script), str(handoff), expected_hash],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
        )
        calls = [json.loads(line) for line in calls_file.read_text().splitlines()] if calls_file.exists() else []
        for call in calls:
            if call[0] == "logs":
                assert "--limit-bytes=1" in call
                assert "--request-timeout=30s" in call
        assert not list(temporary_files.iterdir()), "verification must remove its captured output files"
        return result, calls

    return run


@pytest.mark.parametrize("command", [NVFLARE_COMMAND, ["/coco-app"]], ids=["nvflare", "demo"])
def test_silent_workloads_pass_cluster_checks_and_require_trusted_federation_verification(run_verifier, command):
    result, calls = run_verifier(command=command)
    assert result.returncode == 0, result.stderr
    assert "Silent workload: no application output observed" in result.stdout
    assert "Cluster-side checks passed for authenticated Pod" in result.stdout
    assert "do not establish NVFlare registration, peer attestation, or application readiness" in result.stdout
    assert "trusted federation operator must complete provision/VERIFY-RUNNING-FEDERATION.md" in result.stdout
    assert [call[0] for call in calls] == ["create", "wait", "get", "logs", "exec", "get"]


@pytest.mark.parametrize("output", ["x", SECRET, "Starting NVFlare\n" + SECRET])
def test_visible_startup_output_fails_without_disclosing_captured_content(run_verifier, output):
    result, calls = run_verifier(log_stdout=output)
    assert result.returncode != 0
    assert "silent workload emitted output visible to CoCo" in result.stderr
    assert SECRET not in result.stdout + result.stderr
    assert "Starting NVFlare" not in result.stdout + result.stderr
    assert "exec" not in [call[0] for call in calls]


@pytest.mark.parametrize("status", [1, 7])
def test_explicit_read_stream_policy_denial_is_accepted_without_stdout(run_verifier, status):
    result, _ = run_verifier(log_status=status, log_stderr=READ_DENIAL)
    assert result.returncode == 0, result.stderr
    assert "Application log access denied by guest ReadStreamRequest policy" in result.stdout


@pytest.mark.parametrize("error", ["connection refused", "Forbidden: cannot get pods/log", "PermissionDenied", ""])
def test_unrelated_log_transport_and_access_failures_are_not_policy_denials(run_verifier, error):
    result, calls = run_verifier(log_status=1, log_stderr=error + SECRET)
    assert result.returncode != 0
    assert "Kubernetes access/transport errors are not a policy denial" in result.stderr
    assert SECRET not in result.stdout + result.stderr
    assert "exec" not in [call[0] for call in calls]


def test_read_stream_denial_does_not_excuse_visible_stdout(run_verifier):
    result, _ = run_verifier(log_status=1, log_stderr=READ_DENIAL, log_stdout=SECRET)
    assert result.returncode != 0
    assert "silent workload emitted output" in result.stderr
    assert SECRET not in result.stdout + result.stderr


@pytest.mark.parametrize(
    "status,error,expected",
    [
        (0, EXEC_DENIAL, "kubectl exec unexpectedly succeeded"),
        (1, "connection refused", "not at the expected guest-policy boundary"),
        (1, "PermissionDenied", "not at the expected guest-policy boundary"),
    ],
)
def test_exec_requires_a_nonzero_explicit_exec_process_policy_denial(run_verifier, status, error, expected):
    result, _ = run_verifier(exec_status=status, exec_stderr=error + SECRET)
    assert result.returncode != 0
    assert expected in result.stderr
    assert SECRET not in result.stdout + result.stderr


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda pod: pod["spec"].update(runtimeClassName="runc"), "wrong runtime class"),
        (lambda pod: pod["spec"]["containers"][0].update(image="changed"), "image or command differs"),
        (lambda pod: pod["spec"]["containers"][0].update(command=["/bin/sh"]), "image or command differs"),
        (lambda pod: pod["metadata"].update(annotations={}), "init-data differs"),
        (lambda pod: pod["status"]["containerStatuses"][0].update(ready=False), "not Ready"),
        (lambda pod: pod["status"]["containerStatuses"][0].update(restartCount=1), "restarted"),
        (lambda pod: pod["status"].update(phase="Pending"), "not Running"),
        (lambda pod: pod["status"].update(containerStatuses=[]), "not Ready"),
    ],
)
def test_live_workload_must_match_authenticated_handoff_and_health_requirements(run_verifier, mutation, error):
    result, calls = run_verifier(mutate_live=mutation)
    assert result.returncode != 0
    assert error in result.stderr
    assert [call[0] for call in calls] == ["create", "wait", "get"]


def test_readiness_wait_failure_does_not_continue_to_live_checks(run_verifier):
    result, calls = run_verifier(wait_status=1)
    assert result.returncode != 0
    assert [call[0] for call in calls] == ["create", "wait"]


def test_handoff_hash_mismatch_stops_before_any_cluster_command(run_verifier):
    result, calls = run_verifier(expected_hash="0" * 64)
    assert result.returncode != 0
    assert "Pod SHA-256 mismatch" in result.stderr
    assert not calls
