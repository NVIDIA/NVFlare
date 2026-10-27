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

"""Exercise the manifest-only launcher without contacting Kubernetes."""

import hashlib
import json
import os
import runpy
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from nvflare.lighter.cc_provision import workload_security
from tests.unit_test.lighter.cc_provision.impl.deployment_guards_test import ROOT, require_coco_bash

RUNTIMES = ("kata-qemu-snp", "kata-qemu-nvidia-gpu-snp", "kata-qemu-tdx", "kata-qemu-nvidia-gpu-tdx")

KUBECTL = r"""
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

args = sys.argv[1:]
entry = {"args": args, "kubeconfig": os.environ.get("KUBECONFIG")}
if args[0] == "apply":
    source = Path(args[args.index("-f") + 1])
    entry.update(
        snapshot=str(source),
        digest=hashlib.sha256(source.read_bytes()).hexdigest(),
        file_mode=stat.S_IMODE(source.stat().st_mode),
        directory_mode=stat.S_IMODE(source.parent.stat().st_mode),
    )
with Path(os.environ["LAUNCH_CALLS"]).open("a") as stream:
    stream.write(json.dumps(entry) + "\n")
if args[0] == "apply" and "--dry-run=server" in args:
    if os.environ.get("CHANGE_SOURCE") == "1":
        Path(os.environ["SOURCE_POD"]).write_text("changed after validation\n")
    sys.exit(int(os.environ.get("DRY_RUN_STATUS", "0")))
case = os.environ.get("LAUNCH_CASE", "ready")
namespace = os.environ.get("ACTUAL_NAMESPACE", "default")
if args[0] == "apply":
    print(json.dumps({"metadata": {"name": "review", "namespace": namespace, "uid": "current-uid"}}))
elif args[:2] == ["get", "events"]:
    if case == "event-forbidden":
        print("Error from server (Forbidden): cannot list events", file=sys.stderr)
        sys.exit(1)
    if case == "invalid-events":
        print("invalid JSON")
        sys.exit(0)
    event_uid = "old-uid" if case == "stale-event" else "current-uid"
    print(json.dumps({"items": [{
        "involvedObject": {"uid": event_uid},
        "reason": "FailedCreatePodSandBox",
        "message": "no runtime for kata-qemu-snp is configured",
        "lastTimestamp": "2020-01-01T00:00:00Z",
    }]}))
elif args[:2] == ["get", "pod"] and "json" in args:
    entries = [json.loads(line) for line in Path(os.environ["LAUNCH_CALLS"]).read_text().splitlines()]
    count = sum(entry["args"][:2] == ["get", "pod"] for entry in entries)
    ready = case in ("ready", "ready-retained") or (case == "stale-event" and count > 1)
    uid = "replacement-uid" if case == "replaced" else "current-uid"
    print(json.dumps({
        "metadata": {"name": "review", "namespace": namespace, "uid": uid},
        "status": {"conditions": [{"type": "Ready", "status": "True" if ready else "False"}]},
    }))
elif args[0] == "get":
    print("review Running")
"""


@pytest.fixture
def launch(tmp_path):
    kit = tmp_path / "coco"
    (kit / "lib").mkdir(parents=True)
    shutil.copyfile(ROOT / "coco/50-launch-handoff.sh", kit / "50-launch-handoff.sh")
    shutil.copyfile(
        Path(workload_security.__file__),
        kit / "lib/workload-security-context.py",
    )
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "kubectl").write_text(f"#!{sys.executable}\n" + KUBECTL)
    (binaries / "kubectl").chmod(0o700)
    (binaries / "python3").write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' "$@"\n')
    (binaries / "python3").chmod(0o700)
    source = tmp_path / "received.json"
    calls = tmp_path / "calls.jsonl"
    temporary = tmp_path / "private"
    temporary.mkdir()
    kubeconfig = tmp_path / "existing-kubeconfig"
    kubeconfig.write_text("current-context: existing-cluster\n")

    def run(
        runtime="kata-qemu-snp",
        hash_mode="omitted",
        mutate=None,
        change_source=False,
        dry_run_status=0,
        case="ready",
        namespace="default",
    ):
        fixture = runpy.run_path(str(Path(__file__).with_name("workload_security_context_test.py")))
        pod = fixture["final_workload_pod"](runtime)
        if mutate:
            mutate(pod)
        source.write_text(json.dumps(pod))
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        arguments = [require_coco_bash(), str(kit / "50-launch-handoff.sh"), str(source)]
        if hash_mode != "omitted":
            arguments.append({"empty": "", "correct": digest, "mismatch": "0" * 64, "malformed": "bad"}[hash_mode])
        result = subprocess.run(
            arguments,
            capture_output=True,
            text=True,
            timeout=15,
            env={
                **os.environ,
                "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
                "TMPDIR": str(temporary),
                "KUBECONFIG": str(kubeconfig),
                "LAUNCH_CALLS": str(calls),
                "SOURCE_POD": str(source),
                "CHANGE_SOURCE": str(int(change_source)),
                "DRY_RUN_STATUS": str(dry_run_status),
                "LAUNCH_CASE": case,
                "ACTUAL_NAMESPACE": namespace,
            },
        )
        operations = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
        assert not list(temporary.iterdir()), "private snapshots must be removed after success or failure"
        assert kubeconfig.read_text() == "current-context: existing-cluster\n"
        return result, operations, digest, source

    return run


@pytest.mark.parametrize("runtime", RUNTIMES)
@pytest.mark.parametrize("hash_mode", ["omitted", "empty", "correct"])
@pytest.mark.skipif(sys.version_info < (3, 11), reason="CoCo launch helpers require Python 3.11+")
def test_manifest_only_launch_preserves_supported_target_and_existing_context(launch, runtime, hash_mode):
    result, operations, digest, source = launch(runtime, hash_mode)
    assert result.returncode == 0, result.stderr
    assert "image, runtime, security-context and InitData policy checks" in result.stdout
    assert [operation["args"][0] for operation in operations] == ["apply", "apply", "get", "get"]
    first, second = operations[:2]
    assert first["args"] == ["apply", "--dry-run=server", "-f", first["snapshot"], "--request-timeout=30s"]
    assert second["args"] == ["apply", "-f", second["snapshot"], "--request-timeout=30s", "-o", "json"]
    assert first["snapshot"] == second["snapshot"] != str(source)
    assert first["digest"] == second["digest"] == digest
    assert all(item["kubeconfig"].endswith("existing-kubeconfig") for item in operations)
    assert first["file_mode"] == 0o600 and first["directory_mode"] == 0o700
    assert operations[2]["args"] == [
        "get",
        "pod",
        "review",
        "-n",
        "default",
        "-o",
        "json",
        "--request-timeout=10s",
    ]
    assert not Path(first["snapshot"]).exists()


@pytest.mark.parametrize("hash_mode", ["mismatch", "malformed"])
def test_explicit_invalid_hash_fails_before_kubernetes_access(launch, hash_mode):
    result, operations, _, _ = launch(hash_mode=hash_mode)
    assert result.returncode != 0
    assert not operations
    assert ("Pod SHA-256 mismatch" if hash_mode == "mismatch" else "not 64 lowercase hex") in result.stderr


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["spec"].update(runtimeClassName="runc"),
        lambda p: p["spec"]["containers"][0].update(image="registry.example/unpinned:latest"),
        lambda p: p["spec"]["containers"][0].update(command=["/bin/sh"]),
        lambda p: p["spec"]["containers"][0]["securityContext"].update(privileged=True),
        lambda p: p["spec"]["containers"][0].update(resources={"limits": {"nvidia.com/pgpu": 2}}),
        lambda p: p["metadata"].update(annotations={}),
        lambda p: p["spec"].update(automountServiceAccountToken=True),
    ],
)
@pytest.mark.skipif(sys.version_info < (3, 11), reason="CoCo launch helpers require Python 3.11+")
def test_received_manifest_security_constraints_remain_enforced_without_hash(launch, mutation):
    result, operations, _, _ = launch(mutate=mutation)
    assert result.returncode != 0
    assert not operations


@pytest.mark.skipif(sys.version_info < (3, 11), reason="CoCo launch helpers require Python 3.11+")
def test_source_changes_after_validation_cannot_change_applied_snapshot(launch):
    result, operations, digest, source = launch(change_source=True)
    assert result.returncode == 0, result.stderr
    assert source.read_text() == "changed after validation\n"
    assert all(operation["digest"] == digest for operation in operations if "digest" in operation)


@pytest.mark.skipif(sys.version_info < (3, 11), reason="CoCo launch helpers require Python 3.11+")
def test_server_dry_run_failure_prevents_apply_and_cleans_snapshot(launch):
    result, operations, _, _ = launch(dry_run_status=7)
    assert result.returncode == 7
    assert len(operations) == 1 and "--dry-run=server" in operations[0]["args"]


@pytest.mark.skipif(sys.version_info < (3, 11), reason="CoCo launch helpers require Python 3.11+")
def test_sandbox_failure_reports_event_and_handler_guidance_without_waiting(launch):
    result, operations, _, _ = launch(case="sandbox-failure")
    assert result.returncode != 0
    assert "FailedCreatePodSandBox: no runtime for kata-qemu-snp is configured" in result.stderr
    assert "RuntimeClass" in result.stderr and "containerd/Kata handler" in result.stderr
    assert "Pod is retained for diagnosis" in result.stderr
    assert [entry["args"][:2] for entry in operations[2:]] == [["get", "pod"], ["get", "events"]]
    assert "involvedObject.uid=current-uid" in operations[3]["args"]
    assert not any(entry["args"][0] in ("wait", "delete") for entry in operations)


@pytest.mark.skipif(sys.version_info < (3, 11), reason="CoCo launch helpers require Python 3.11+")
def test_retained_warning_from_another_pod_uid_does_not_fail_launch(launch):
    result, operations, _, _ = launch(case="stale-event")
    assert result.returncode == 0, result.stderr
    assert any(entry["args"][:2] == ["get", "events"] for entry in operations)
    assert "FailedCreatePodSandBox" not in result.stderr


@pytest.mark.skipif(sys.version_info < (3, 11), reason="CoCo launch helpers require Python 3.11+")
def test_ready_pod_takes_precedence_over_retained_sandbox_warnings(launch):
    result, operations, _, _ = launch(case="ready-retained")
    assert result.returncode == 0, result.stderr
    assert not any(entry["args"][:2] == ["get", "events"] for entry in operations)


@pytest.mark.skipif(sys.version_info < (3, 11), reason="CoCo launch helpers require Python 3.11+")
def test_replaced_pod_is_not_silently_followed(launch):
    result, operations, _, _ = launch(case="replaced")
    assert result.returncode != 0
    assert "Pod was replaced while waiting" in result.stderr
    assert not any(entry["args"][:2] == ["get", "events"] for entry in operations)


@pytest.mark.parametrize(
    ("case", "message"),
    [("event-forbidden", "cannot list events"), ("invalid-events", "did not return valid JSON")],
)
@pytest.mark.skipif(sys.version_info < (3, 11), reason="CoCo launch helpers require Python 3.11+")
def test_event_query_failures_are_reported(launch, case, message):
    result, _, _, _ = launch(case=case)
    assert result.returncode != 0
    assert message in result.stderr
    if case == "event-forbidden":
        assert "permission to get Pods and list their events" in result.stderr


@pytest.mark.skipif(sys.version_info < (3, 11), reason="CoCo launch helpers require Python 3.11+")
def test_context_namespace_is_taken_from_actual_applied_pod(launch):
    result, operations, _, _ = launch(namespace="existing-context-namespace")
    assert result.returncode == 0, result.stderr
    assert "existing-context-namespace/review" in result.stdout
    assert all(entry["args"][entry["args"].index("-n") + 1] == "existing-context-namespace" for entry in operations[2:])


@pytest.mark.parametrize("failure", ["deadline", "request-timeout"])
def test_readiness_deadline_and_api_request_timeout_are_bounded(tmp_path, failure):
    applied = tmp_path / "applied.json"
    applied.write_text(json.dumps({"metadata": {"name": "review", "namespace": "default", "uid": "current-uid"}}))
    script = (ROOT / "coco/50-launch-handoff.sh").read_text()
    program = script.split("<<'PY'\n", 2)[2].split("\nPY\n", 1)[0]
    clock = SimpleNamespace(monotonic=Mock(side_effect=[0, 601 if failure == "deadline" else 1]), sleep=Mock())
    request = Mock(side_effect=subprocess.TimeoutExpired("kubectl", 15))
    api = SimpleNamespace(run=request, TimeoutExpired=subprocess.TimeoutExpired)
    with (
        patch.dict(sys.modules, {"time": clock, "subprocess": api}),
        patch.object(sys, "argv", ["readiness", str(applied)]),
    ):
        with pytest.raises(SystemExit, match="timed out"):
            exec(compile(program, str(ROOT / "coco/50-launch-handoff.sh"), "exec"), {})
    if failure == "deadline":
        request.assert_not_called()
    else:
        assert request.call_args.kwargs["timeout"] <= 15
        assert "--request-timeout=10s" in request.call_args.args[0]
    clock.sleep.assert_not_called()
