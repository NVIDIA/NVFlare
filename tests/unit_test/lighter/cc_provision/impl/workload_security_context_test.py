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

"""Offline workload-context contract and pinned-policy checks; no new dependencies."""

import ast
import base64
import copy
import gzip
import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[5] / "examples/devops/coco"
API = runpy.run_path(str(ROOT / "shared/workload-security-context.py"))
ADMIN = runpy.run_path(str(ROOT / "admin/lib/workload-launch-profile.py"))
IMAGE = "registry.example.invalid/app@sha256:" + "a" * 64


def context(readonly=False):
    return {
        "privileged": False,
        "allowPrivilegeEscalation": False,
        "runAsNonRoot": True,
        "runAsUser": 65532,
        "runAsGroup": 65532,
        "readOnlyRootFilesystem": readonly,
        "capabilities": {"drop": ["ALL"]},
        "seccompProfile": {"type": "RuntimeDefault"},
    }


def pod(readonly=False):
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "spec": {
            "runtimeClassName": "kata-qemu-nvidia-gpu-snp",
            "containers": [
                {
                    "name": "app",
                    "image": IMAGE,
                    "securityContext": context(readonly),
                    "resources": {"limits": {"nvidia.com/pgpu": "1"}},
                }
            ],
        },
    }


def contract(readonly=False):
    return {
        "schema": API["SCHEMA"],
        "profile_id": "reviewed",
        "runtime_class": "kata-qemu-nvidia-gpu-snp",
        "kata_version": "3.29.0",
        "guest_token_api": "guest-local-aa-token/v1",
        "kata_deploy_image": "registry.example.invalid/runtime@sha256:" + "b" * 64,
        "kata_config_sha256": "c" * 64,
        "launch_inputs_sha256": "d" * 64,
        "vm_defaults": {"vcpus": 1, "memory_mib": 8192},
        "workload_security_context": context(readonly),
        "pod_constraints": {
            "container_count": 1,
            "gpu_resource": "nvidia.com/pgpu",
            "gpu_count": 1,
            "cpu_memory_resources": "omitted",
            "host_namespaces": False,
            "allowed_annotations": ["io.katacontainers.config.hypervisor.cc_init_data"],
        },
    }


@pytest.mark.parametrize("readonly", [False, True])
def test_approved_writable_and_readonly_contracts(tmp_path, readonly):
    profile = contract(readonly)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    loaded = ADMIN["load_profile"](
        path, hashlib.sha256(path.read_bytes()).hexdigest(), profile["runtime_class"], "3.29.0"
    )
    ADMIN["validate_pod"](loaded, pod(readonly))
    changed = pod(not readonly)
    with pytest.raises(ValueError, match="differs"):
        ADMIN["validate_pod"](loaded, changed)


@pytest.mark.parametrize(
    "key,value",
    [
        ("privileged", True),
        ("privileged", 0),
        ("allowPrivilegeEscalation", True),
        ("allowPrivilegeEscalation", "false"),
        ("runAsNonRoot", False),
        ("runAsNonRoot", 1),
        ("runAsUser", 0),
        ("runAsUser", True),
        ("runAsUser", "65532"),
        ("runAsUser", -1),
        ("runAsGroup", 0),
        ("runAsGroup", 2**32 - 1),
        ("readOnlyRootFilesystem", "false"),
        ("readOnlyRootFilesystem", 0),
        ("capabilities", {"drop": ["ALL"], "add": ["SYS_ADMIN"]}),
        ("capabilities", {"drop": []}),
        ("capabilities", {"drop": ["ALL"], "add": None}),
        ("seccompProfile", {"type": "Unconfined"}),
        ("seccompProfile", {"type": "Localhost", "localhostProfile": "x"}),
        ("seccompProfile", {"type": "RuntimeDefault", "extra": True}),
        ("procMount", "Unmasked"),
    ],
)
def test_unsafe_or_unsupported_context_rejected(key, value):
    bad = context()
    bad[key] = value
    with pytest.raises(ValueError):
        API["validate_context"](bad)


@pytest.mark.parametrize("key", sorted(API["CONTEXT_KEYS"]))
def test_missing_fields_fail_closed(key):
    bad = context()
    del bad[key]
    with pytest.raises(ValueError):
        API["validate_context"](bad)


@pytest.mark.parametrize("key,value", [("runAsUser", 1000), ("runAsGroup", 1000), ("readOnlyRootFilesystem", True)])
def test_safe_but_unapproved_drift_rejected(key, value):
    changed = pod()
    changed["spec"]["containers"][0]["securityContext"][key] = value
    with pytest.raises(ValueError, match="differs"):
        ADMIN["validate_pod"](contract(), changed)


def test_empty_added_capabilities_are_canonical():
    value = context()
    value["capabilities"]["add"] = []
    assert API["validate_context"](value) == context()


@pytest.mark.parametrize("value", [{}, {"runAsUser": 0}, None])
def test_pod_level_override_rejected(value):
    changed = pod()
    changed["spec"]["securityContext"] = value
    with pytest.raises(ValueError):
        API["pod_context"](changed)
    with pytest.raises(ValueError):
        ADMIN["validate_pod"](contract(), changed)


def policy_data(readonly=False):
    return {
        "request_defaults": {
            "ReadStreamRequest": False,
            "WriteStreamRequest": False,
            "ExecProcessRequest": {"allowed_commands": [], "regex": []},
        },
        "containers": [
            {
                "exec_commands": [],
                "OCI": {
                    "Annotations": {"io.kubernetes.cri.image-name": IMAGE},
                    "Process": {
                        "User": {"UID": 65532, "GID": 65532},
                        "NoNewPrivileges": True,
                        "Capabilities": {key: [] for key in API["CAPABILITY_SETS"]},
                    },
                    "Root": {"Readonly": readonly},
                },
            }
        ],
    }


def policy(data):
    # Structural fixture, NOT an executable Rego/guest attestation test.
    guards = list(API["REQUIRED_GUARDS"]) + [
        f"match_caps(p_caps.{key}, i_caps.{key})" for key in API["CAPABILITY_SETS"]
    ]
    guards += [f"default {name} := false" for name in API["DENIED_REQUESTS"]]
    guards += ["default AllowRequestsFailingPolicy := false"]
    return "\n".join(guards) + "\npolicy_data := " + json.dumps(data)


@pytest.mark.parametrize("readonly", [False, True])
def test_generated_policy_matches_context(readonly):
    API["validate_policy"](policy(policy_data(readonly)), IMAGE, context(readonly))


@pytest.mark.parametrize("request_name", ["ReadStreamRequest", "WriteStreamRequest"])
@pytest.mark.parametrize("value", [True, None, 0, "false", "missing"])
def test_effective_stream_permissions_fail_closed(request_name, value):
    data = policy_data()
    if value == "missing":
        del data["request_defaults"][request_name]
    else:
        data["request_defaults"][request_name] = value
    with pytest.raises(ValueError, match=request_name):
        API["validate_policy"](policy(data), IMAGE, context())


@pytest.mark.parametrize("target", ["allowed_commands", "regex", "container", "second_container"])
def test_every_exec_allowlist_must_be_empty(target):
    data = policy_data()
    if target in ("allowed_commands", "regex"):
        data["request_defaults"]["ExecProcessRequest"][target] = [".*"]
    else:
        if target == "second_container":
            data["containers"].append(copy.deepcopy(data["containers"][0]))
        data["containers"][-1]["exec_commands"] = [["sh"]]
    with pytest.raises(ValueError, match="exec"):
        API["validate_request_policy"](policy(data))


@pytest.mark.parametrize(
    "change", ["set_policy", "fail_open", "commented_default", "missing_defaults", "duplicate_defaults"]
)
def test_request_policy_structure_fails_closed(change):
    data = policy_data()
    value = policy(data)
    if change == "set_policy":
        value = "SetPolicyRequest if { true }\n" + value
    elif change == "fail_open":
        value = value.replace("AllowRequestsFailingPolicy := false", "AllowRequestsFailingPolicy := true")
    elif change == "commented_default":
        value = value.replace("default ReadStreamRequest", "# default ReadStreamRequest")
    elif change == "missing_defaults":
        del data["request_defaults"]
        value = policy(data)
    else:
        value = value.replace('"request_defaults":', '"request_defaults": {}, "request_defaults":', 1)
    with pytest.raises(ValueError):
        API["validate_request_policy"](value)


@pytest.mark.parametrize("role", ["admin", "coco"])
@pytest.mark.parametrize("stream_enabled", [False, True])
def test_actual_role_policy_gate_checks_stream_settings(tmp_path, role, stream_enabled):
    pytest.importorskip("tomllib", reason="deployment entrypoints require Python 3.11+")
    import re

    import yaml

    data = policy_data()
    data["request_defaults"]["ReadStreamRequest"] = stream_enabled
    data["containers"][0]["OCI"]["Process"]["Args"] = ["python3"]
    # Structural fixtures only; no claim of executing Rego in a guest.
    mount_guards = '\np_mount.source != ""\np_mount.source == ""\ni_storage.driver in {"blk", "scsi"}\nexpect_root_path == i_storage.mount_point\n'
    raw = '[data]\n"policy.rego" = ' + "'''\n" + mount_guards + policy(data) + "\n'''\n"
    value = pod()
    value["metadata"] = {
        "name": "review",
        "annotations": {
            "io.katacontainers.config.hypervisor.cc_init_data": base64.b64encode(gzip.compress(raw.encode())).decode()
        },
    }
    value["spec"].update(
        {
            "hostNetwork": False,
            "hostPID": False,
            "hostIPC": False,
            "restartPolicy": "Never",
            "automountServiceAccountToken": False,
            "enableServiceLinks": False,
        }
    )
    value["spec"]["containers"][0].update(
        {"stdin": False, "tty": False, "command": ["python3"], "imagePullPolicy": "Always"}
    )
    helper = ROOT / role / "lib/workload-security-context.py"
    if role == "admin":
        source = (ROOT / "admin/30-generate-pod-and-policies.sh").read_text()
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", source, re.S)
        code = next(b for b in blocks if "Pod and generated agent-policy invariants verified" in b)
        (tmp_path / "pod.yaml").write_text(yaml.safe_dump(value))
        args = [tmp_path, IMAGE, '["python3"]', "kata-qemu-nvidia-gpu-snp", "65532", "65532", "false", helper]
    else:
        source = (ROOT / "coco/50-launch-handoff.sh").read_text()
        code = re.findall(r"<<'PY'\n(.*?)\nPY", source, re.S)[0]
        (tmp_path / "pod.json").write_text(json.dumps(value))
        args = [tmp_path / "pod.json", "kata-qemu-nvidia-gpu-snp", "registry.example.invalid", helper]
    result = subprocess.run([sys.executable, "-c", code, *map(str, args)], capture_output=True, text=True)
    assert (result.returncode == 0) is (not stream_enabled), result.stderr
    if stream_enabled:
        assert "ReadStreamRequest must be false" in result.stderr


@pytest.mark.parametrize(
    "target,key,value",
    [
        ("User", "UID", 0),
        ("User", "GID", 0),
        ("User", "UID", True),
        ("Process", "NoNewPrivileges", False),
        ("Process", "NoNewPrivileges", 1),
        ("Root", "Readonly", True),
        ("Root", "Readonly", 0),
        *[("Capabilities", key, ["CAP_SYS_ADMIN"]) for key in API["CAPABILITY_SETS"]],
    ],
)
def test_weakened_generated_policy_rejected(target, key, value):
    data = policy_data()
    oci = data["containers"][0]["OCI"]
    objects = {
        "User": oci["Process"]["User"],
        "Process": oci["Process"],
        "Root": oci["Root"],
        "Capabilities": oci["Process"]["Capabilities"],
    }
    objects[target][key] = value
    with pytest.raises(ValueError):
        API["validate_policy"](policy(data), IMAGE, context())


@pytest.mark.parametrize("guard", API["REQUIRED_GUARDS"])
def test_missing_pinned_security_guards_rejected(guard):
    with pytest.raises(ValueError, match="guard"):
        API["validate_policy"](policy(policy_data()).replace(guard, ""), IMAGE, context())


def test_duplicate_or_missing_application_policy_rejected():
    data = policy_data()
    data["containers"].append(copy.deepcopy(data["containers"][0]))
    with pytest.raises(ValueError):
        API["validate_policy"](policy(data), IMAGE, context())
    with pytest.raises(ValueError):
        API["validate_policy"](policy({"containers": []}), IMAGE, context())
    with pytest.raises(ValueError):
        API["validate_policy"](policy(policy_data()) + "\npolicy_data := {}", IMAGE, context())


def test_final_pod_policy_is_checked_not_just_yaml():
    pytest.importorskip("tomllib", reason="embedded init-data validation requires Python 3.11+")
    value = pod()

    def annotate(data):
        raw = '[data]\n"policy.rego" = ' + json.dumps(policy(data)) + "\n"
        value["metadata"] = {
            "annotations": {
                "io.katacontainers.config.hypervisor.cc_init_data": base64.b64encode(
                    gzip.compress(raw.encode())
                ).decode()
            }
        }

    annotate(policy_data())
    ADMIN["validate_pod"](contract(), value, require_policy=True)
    bad = policy_data()
    bad["containers"][0]["OCI"]["Process"]["NoNewPrivileges"] = False
    annotate(bad)
    with pytest.raises(ValueError):
        ADMIN["validate_pod"](contract(), value)


def test_yaml_duplicate_fields_rejected(tmp_path):
    pytest.importorskip("yaml")
    path = tmp_path / "pod.yaml"
    path.write_text("spec:\n  securityContext: {}\n  securityContext: {}\n")
    with pytest.raises(ValueError, match="duplicate"):
        API["read_pod"](path)


def test_final_handoff_cannot_omit_policy():
    with pytest.raises(ValueError, match="requires embedded"):
        ADMIN["validate_pod"](contract(), pod(), require_policy=True)
    assert '--pod "${OUTPUT_DIR}/pod.yaml" --require-policy' in (ROOT / "admin/40-create-handoffs.sh").read_text()


@pytest.mark.parametrize("role", ["admin", "trusted_system", "coco"])
def test_shared_helper_is_materialized_in_standalone_kit(tmp_path, role):
    kits = runpy.run_path(str(ROOT / "role_kits.py"))
    target = f"{role}/lib/workload-security-context.py"
    assert kits["GENERATED"][target] == "shared/workload-security-context.py"
    wrapper = runpy.run_path(str(ROOT / target))
    assert wrapper["pod_context"](pod()) == context()
    isolated = tmp_path / "kit/lib/workload-security-context.py"
    isolated.parent.mkdir(parents=True)
    isolated.write_bytes((ROOT / kits["GENERATED"][target]).read_bytes())
    assert runpy.run_path(str(isolated))["pod_context"](pod()) == context()
    if role == "admin":
        consumer = isolated.parent / "workload-launch-profile.py"
        consumer.write_bytes((ROOT / "admin/lib/workload-launch-profile.py").read_bytes())
        runpy.run_path(str(consumer))["validate_pod"](contract(), pod())


def test_stage05_records_application_not_collector_context():
    source = (ROOT / "trusted_system/05-define-approved-launch-profile.py").read_text()
    assert 'security["pod_context"](workload)' in source
    tree = ast.parse(source)
    assignment = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "data" for t in n.targets)
    )
    keys = [k.value for k in assignment.value.keys]
    assert "workload_security_context" in keys


@pytest.mark.parametrize("readonly", [True, False])
def test_stage05_actual_entrypoint(tmp_path, readonly):
    pytest.importorskip("tomllib", reason="trusted deployment scripts require Python 3.11+")
    yaml = pytest.importorskip("yaml")
    source, config, output = (tmp_path / name for name in ("source.yaml", "kata.toml", "approved.json"))
    source.write_text(yaml.safe_dump(pod(readonly)))
    config.write_text(
        '[hypervisor.qemu]\nkernel_params = "agent.guest_components_rest_api=all"\n'
        "default_vcpus = 1\ndefault_memory = 8192\n"
    )
    entry = ROOT / "trusted_system/05-define-approved-launch-profile.py"
    with (
        patch.object(sys, "argv", [str(entry), str(source), str(config), str(output)]),
        patch("subprocess.check_output", return_value=b'{"lscpu": []}'),
        patch("os.umask"),
    ):
        runpy.run_path(str(entry), run_name="__main__")
    approved = json.loads(output.read_text())
    assert approved["workload_security_context"] == context(readonly)
    assert approved["workload_yaml_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    # Unsafe source must be rejected before even querying host CPU/artifacts.
    bad = pod(readonly)
    bad["spec"]["containers"][0]["securityContext"]["privileged"] = True
    source.write_text(yaml.safe_dump(bad))
    rejected = tmp_path / "rejected.json"
    with (
        patch.object(sys, "argv", [str(entry), str(source), str(config), str(rejected)]),
        patch("subprocess.check_output") as cpu,
        patch("os.umask"),
    ):
        with pytest.raises(ValueError, match="privileged"):
            runpy.run_path(str(entry), run_name="__main__")
        cpu.assert_not_called()
    assert not rejected.exists()


@pytest.mark.parametrize("changed_uid", [False, True])
def test_stage09_revalidates_source_security_context(tmp_path, changed_uid):
    pytest.importorskip("yaml")
    shell = (ROOT / "trusted_system/09-finalize-platform-reference.sh").read_text()
    start = shell.index("import hashlib, json, runpy, sys")
    code = shell[start : shell.index("\nPY", start)]
    # Deliberately update the source hash to isolate semantic approval checking.
    source = tmp_path / "source.json"
    import yaml

    changed = pod()
    if changed_uid:
        changed["spec"]["containers"][0]["securityContext"]["runAsUser"] = 1000
    source.write_text(yaml.safe_dump(changed))
    profile = tmp_path / "profile.json"
    artifacts = {name: {"sha256": name} for name in ("path", "firmware", "kernel", "initrd")}
    resources = changed["spec"]["containers"][0]["resources"]
    profile.write_text(
        json.dumps(
            {
                "workload_yaml_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "workload_security_context": context(),
                "pod_resources": resources,
                "kata_config_sha256": "config",
                "artifacts": artifacts,
            }
        )
    )
    actual = tmp_path / "actual.json"
    actual.write_text(
        json.dumps(
            {
                "pod_resources": [resources],
                "artifacts": {
                    **{("qemu_executable" if key == "path" else key): value for key, value in artifacts.items()},
                    "kata_config": {"sha256": "config"},
                },
            }
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(profile),
            str(actual),
            str(source),
            str(ROOT / "trusted_system/lib/workload-security-context.py"),
        ],
        capture_output=True,
        text=True,
    )
    assert (result.returncode != 0) == changed_uid, result.stderr
    if changed_uid:
        assert "securityContext differs" in result.stderr


def test_stage30_checks_complete_context_before_genpolicy_and_final_output():
    source = (ROOT / "admin/30-generate-pod-and-policies.sh").read_text()
    strip = source.index('del pod["spec"]["containers"][0]["securityContext"]["runAsNonRoot"]')
    generate = source.index('"\u0024{GENPOLICY}" \\')
    restore = source.index('pod["spec"]["containers"][0]["securityContext"]["runAsNonRoot"] = True')
    assert source.index("\ncheck_launch_profile\n") < strip < generate < restore
    assert "\ncheck_launch_profile --require-policy\n" in source[restore:]
