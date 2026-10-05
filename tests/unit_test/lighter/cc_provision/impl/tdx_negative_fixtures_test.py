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

"""Offline preparation checks; these assertions never certify guest rejection."""

import base64
import copy
import gzip
import hashlib
import json
import os
import runpy
import stat
from pathlib import Path

import pytest
import yaml

# NVFlare supports Python 3.10; these deployment helpers require Python 3.11+.
# Skip before loading the helper, matching the other CoCo deployment tests.
tomllib = pytest.importorskip("tomllib", reason="CoCo negative-fixture helpers require Python 3.11+")

ROOT = Path(__file__).resolve().parents[5]
API = runpy.run_path(str(ROOT / "examples/devops/coco/acceptance/negative_fixtures.py"))


@pytest.fixture
def baseline(tmp_path):
    initdata = b'algorithm = "sha256"\nversion = "0.1.0"\n[data]\n"policy.rego" = "package agent_policy"\n'
    pod = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": "site-1",
            "namespace": "approved-baseline",
            "uid": "live-source-uid",
            "resourceVersion": "123",
            "ownerReferences": [{"uid": "controller-source-uid"}],
            "annotations": {API["INITDATA"]: base64.b64encode(gzip.compress(initdata)).decode()},
        },
        "spec": {
            "runtimeClassName": "kata-qemu-tdx",
            "containers": [
                {
                    "name": "app",
                    "image": "registry.example/nvflare@sha256:" + "a" * 64,
                    "command": ["/opt/nvflare/startup/sub_start.sh", "--once", "--verify"],
                    "securityContext": {
                        "runAsUser": 65532,
                        "runAsGroup": 65532,
                        "runAsNonRoot": True,
                        "privileged": False,
                        "allowPrivilegeEscalation": False,
                        "readOnlyRootFilesystem": False,
                    },
                }
            ],
        },
        "status": {"phase": "Running"},
    }
    path = tmp_path / "baseline.yaml"
    path.write_text(yaml.safe_dump(pod))
    return path, pod


def prepare(path, output, **overrides):
    args = dict(
        pod_path=path,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        output=output,
        run_id="fixture-1",
        topology="A",
        participant="site-1",
        namespace="isolated-fixtures",
        authority_note="Digest verified from authenticated signed participant handoff.",
    )
    args.update(overrides)
    return API["prepare"](**args)


def test_private_candidates_preserve_baseline_and_never_assert_execution(baseline, tmp_path):
    path, pod = baseline
    original = path.read_bytes()
    output = tmp_path / "output"
    manifest = prepare(path, output)
    assert path.read_bytes() == original
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in output.iterdir())
    assert manifest["hardware_acceptance"] == "NOT_RUN"
    assert not manifest["applied"] and not manifest["approval_changed"] and not manifest["policy_or_resource_written"]
    assert manifest["baseline"]["identity"]["pod_uid"] == "live-source-uid"
    assert set(manifest["fixtures"]) == {
        "control",
        "changed-command",
        "weakened-context",
        "host-mount",
        "altered-initdata",
    }
    for fixture in manifest["fixtures"].values():
        content = (output / fixture["file"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == fixture["sha256"]
        candidate = yaml.safe_load(content)
        assert "status" not in candidate
        assert "resourceVersion" not in candidate["metadata"]
        assert "ownerReferences" not in candidate["metadata"]
        assert fixture["identity"]["pod_uid"] is None
        assert fixture["identity"]["namespace"] == "isolated-fixtures"
        assert fixture["observed_status"] == "NOT_RUN"
        assert candidate["spec"]["containers"][0]["image"] == pod["spec"]["containers"][0]["image"]


def test_variants_change_only_target_behavior_and_fresh_identity(baseline, tmp_path):
    path, _ = baseline
    output = tmp_path / "output"
    manifest = prepare(path, output)
    pods = {key: yaml.safe_load((output / value["file"]).read_text()) for key, value in manifest["fixtures"].items()}
    control = pods["control"]
    assert pods["changed-command"]["spec"]["containers"][0]["command"] == ["/bin/true"]
    assert pods["changed-command"]["spec"]["containers"][0]["args"] == []
    context = pods["weakened-context"]["spec"]["containers"][0]["securityContext"]
    assert (context["runAsUser"], context["runAsGroup"], context["runAsNonRoot"]) == (0, 0, False)
    host = pods["host-mount"]["spec"]
    assert host["volumes"][0]["hostPath"] == {"path": "/proc", "type": "Directory"}
    assert host["containers"][0]["volumeMounts"][0]["readOnly"] is True
    original_initdata = API["initdata_bytes"](control)
    altered_initdata = API["initdata_bytes"](pods["altered-initdata"])
    assert original_initdata != altered_initdata
    assert tomllib.loads(original_initdata.decode()) == tomllib.loads(altered_initdata.decode())
    for key in ("changed-command", "weakened-context", "host-mount"):
        assert API["initdata_bytes"](pods[key]) == original_initdata
    changed_paths = set(manifest["fixtures"]["weakened-context"]["differences_from_isolated_control"])
    assert changed_paths == {
        "/metadata/name",
        "/spec/containers/0/securityContext/runAsUser",
        "/spec/containers/0/securityContext/runAsGroup",
        "/spec/containers/0/securityContext/runAsNonRoot",
    }
    assert manifest["fixtures"]["weakened-context"]["identity"]["run_as_user"] == 0


def test_untrusted_digest_and_same_namespace_rejected_before_writing(baseline, tmp_path):
    path, _ = baseline
    for options in ({"expected_sha256": "b" * 64}, {"namespace": "approved-baseline"}):
        output = tmp_path / "output"
        with pytest.raises(ValueError):
            prepare(path, output, **options)
        assert not output.exists()


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p["spec"].update(runtimeClassName="kata-qemu"),
        lambda p: p["spec"].update(hostNetwork=True),
        lambda p: p["spec"].update(volumes=[{"name": "unknown"}]),
        lambda p: p["spec"]["containers"].append(copy.deepcopy(p["spec"]["containers"][0])),
        lambda p: p["spec"]["containers"][0].update(image="registry.example/unpinned:latest"),
        lambda p: p["spec"]["containers"][0]["securityContext"].update(privileged=True),
        lambda p: p["spec"]["containers"][0]["securityContext"].update(runAsUser=0),
        lambda p: p["metadata"]["annotations"].update({API["INITDATA"]: "not-base64"}),
    ],
)
def test_baseline_must_be_protected_approved_shape(baseline, tmp_path, change):
    path, pod = baseline
    change(pod)
    path.write_text(yaml.safe_dump(pod))
    with pytest.raises((ValueError, OSError)):
        prepare(path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_duplicate_yaml_fields_rejected(baseline, tmp_path):
    path, _ = baseline
    path.write_text(path.read_text() + "kind: Pod\n")
    with pytest.raises(ValueError, match="Duplicate"):
        prepare(path, tmp_path / "output")


def test_existing_outputs_never_overwritten_and_umask_restored(baseline, tmp_path):
    path, _ = baseline
    output = tmp_path / "output"
    old_umask = os.umask(0o022)
    try:
        prepare(path, output)
        original = (output / "manifest.json").read_bytes()
        with pytest.raises(FileExistsError):
            prepare(path, output)
        assert (output / "manifest.json").read_bytes() == original
        assert os.umask(0o022) == 0o022
    finally:
        os.umask(old_umask)


def test_expanded_initdata_limit_rejects_compression_bomb(baseline, tmp_path):
    path, pod = baseline
    pod["metadata"]["annotations"][API["INITDATA"]] = base64.b64encode(
        gzip.compress(b"x" * (API["MAX_INITDATA_BYTES"] + 1))
    ).decode()
    path.write_text(yaml.safe_dump(pod))
    with pytest.raises(ValueError, match="size limit"):
        prepare(path, tmp_path / "output")


def test_no_shell_or_api_call_is_needed_and_cli_reports_only_preparation(baseline, tmp_path, capsys):
    path, _ = baseline
    status = API["main"](
        [
            "--pod",
            str(path),
            "--baseline-sha256",
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "--output",
            str(tmp_path / "output"),
            "--run-id",
            "cli-fixture",
            "--topology",
            "B",
            "--participant",
            "server",
            "--namespace",
            "isolated-fixtures",
            "--authority-note",
            "Verified signed handoff digest; secret sentinel must remain private.",
        ]
    )
    assert status == 0
    stdout = capsys.readouterr().out
    assert "secret sentinel" not in stdout
    assert "no Pods applied" in stdout
    manifest = json.loads((tmp_path / "output/manifest.json").read_text())
    assert manifest["topology"] == "B" and manifest["participant"] == "server"


@pytest.mark.parametrize("overrides", [{"topology": "A", "participant": "server"}, {"run_id": "unsafe/id"}])
def test_scope_and_identity_inputs_fail_closed(baseline, tmp_path, overrides):
    path, _ = baseline
    with pytest.raises(ValueError):
        prepare(path, tmp_path / "output", **overrides)
