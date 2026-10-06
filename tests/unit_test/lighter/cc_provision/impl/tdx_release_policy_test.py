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

"""Target-aware release tests; real OPA is optional, no additional Python packages."""

import copy
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from nvflare.lighter.cc_provision import workload_release as release

ROOT = Path(__file__).resolve().parents[5]
COCO = ROOT / "examples/devops/coco"
TEMPLATE = (COCO / "service/policies/workload-resource-policy.rego.template").read_text()
LEGACY_TEMPLATE = (COCO / "service/policies/workload-resource-policy-v1.rego.template").read_text()
OPA = os.environ.get("COCO_TEST_OPA") or shutil.which("opa")
RUNTIMES = list(release.RUNTIME_TARGETS)
DIGEST = "ab" * 32
IMAGE = "secure-services.example.com:5000/workload@sha256:" + "c" * 64
PATHS = ["default/" + kind + "/demo" for kind in ("image-key", "sig-public-key", "security-policy")]


def authorization(runtime):
    return release.build_authorization(runtime, "demo", IMAGE, ["/bin/workload", "--silent"], DIGEST, PATHS)


def legacy_authorization():
    auth = authorization(RUNTIMES[0])
    auth["schema"] = release.V1
    auth["snp_init_data_sha256"] = auth.pop("init_data_sha256")
    auth["snp_init_data_encoding_in_trustee_v0_21"] = "lowercase-hex"
    for key in ("cpu_tee", "gpu", "init_data_claim"):
        del auth[key]
    return auth


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_target_authentication_contract(runtime):
    auth = authorization(runtime)
    cpu, gpu = release.RUNTIME_TARGETS[runtime]
    assert auth["cpu_tee"] == cpu
    assert auth["gpu"] == gpu
    assert auth["init_data_claim"] == DIGEST + ("0" * 32 if cpu == "tdx" else "")
    assert auth["required_ear_submods"] == (["cpu0", "gpu0"] if gpu == "nvidia" else ["cpu0"])
    assert auth["required_ear_trust_vectors"]["cpu0"]["configuration"] == 2
    assert release.validate_authorization(auth) == auth


@pytest.mark.parametrize("runtime", RUNTIMES)
@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "other"),
        ("cpu_tee", "sgx"),
        ("gpu", "amd"),
        ("init_data_sha256", "ab" * 48),
        ("init_data_claim", "0" * 64),
        ("required_ear_submods", ["cpu0", "gpu1"]),
        ("required_ear_trust_vectors", {}),
        ("kbs_resource_paths", PATHS[:2]),
        ("process_args", ["sh"]),
        ("encrypted_image", "repo:latest"),
        ("release_name", "../escape"),
    ],
)
def test_malformed_authorization_rejected(runtime, field, value):
    auth = authorization(runtime)
    auth[field] = value
    with pytest.raises(ValueError):
        release.validate_authorization(auth)


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_authorization_rejects_numeric_boolean_and_extra_fields(runtime):
    auth = authorization(runtime)
    auth["required_ear_trust_vectors"]["cpu0"]["file-system"] = False
    with pytest.raises(ValueError, match="trust vectors"):
        release.validate_authorization(auth)
    auth = authorization(runtime)
    auth["arbitrary_trust_override"] = True
    with pytest.raises(ValueError, match="fields"):
        release.validate_authorization(auth)


def test_legacy_authorization_is_snp_gpu_only():
    assert release.validate_authorization(legacy_authorization()) == authorization(RUNTIMES[0])
    legacy = legacy_authorization()
    legacy["required_ear_submods"] = ["cpu0"]
    with pytest.raises(ValueError):
        release.validate_authorization(legacy)
    with pytest.raises(ValueError, match="SNP\\+GPU"):
        release.render_fragment(authorization("kata-qemu-tdx"), LEGACY_TEMPLATE, legacy=True)


def test_duplicate_keys_rejected(tmp_path):
    path = tmp_path / "auth.json"
    path.write_text('{"schema":"a","schema":"b"}')
    with pytest.raises(ValueError, match="duplicate"):
        release.load_authorization(path)


@pytest.mark.skipif(not OPA, reason="set COCO_TEST_OPA to the service-pinned OPA 1.8.0 executable")
@pytest.mark.parametrize("runtime", RUNTIMES)
def test_actual_generated_rego_accepts_target_and_denies_all_mutations(tmp_path, runtime):
    auth = authorization(runtime)
    # Include hostile placeholder-like text as data, never executable source.
    auth["process_args"] += ["${prefix}", '"}\nallow := true\n#']
    policy = tmp_path / "policy.rego"
    policy.write_text(
        "package policy\nimport rego.v1\ndefault allow := false\n\n" + release.render_fragment(auth, TEMPLATE)
    )
    subprocess.run([OPA, "check", "--strict", str(policy)], check=True, capture_output=True, text=True)
    for name, (data, token, expected) in release.policy_test_cases(auth).items():
        datafile = tmp_path / "data.json"
        datafile.write_text(json.dumps(data))
        result = subprocess.run(
            [
                OPA,
                "eval",
                "--format",
                "raw",
                "--stdin-input",
                "-d",
                str(policy),
                "-d",
                str(datafile),
                "data.policy.allow",
            ],
            input=json.dumps(token),
            check=True,
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == str(expected).lower(), (runtime, name, result.stderr)


def embedded(script, marker):
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", script.read_text(), re.DOTALL)
    return next(block for block in blocks if marker in block)


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_pod_generator_allocates_gpu_only_for_approved_gpu_runtime(tmp_path, runtime):
    block = embedded(COCO / "admin/30-generate-pod-and-policies.sh", "pod = {")
    pod = tmp_path / "pod.yaml"
    gpu_count = "1" if release.RUNTIME_TARGETS[runtime][1] == "nvidia" else "0"
    subprocess.run(
        [
            sys.executable,
            "-c",
            block,
            str(pod),
            "demo",
            runtime,
            IMAGE,
            '["/bin/workload"]',
            "1000",
            "1000",
            "10.96.0.1",
            "443",
            "true",
            gpu_count,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = yaml.safe_load(pod.read_text())
    assert data["spec"]["runtimeClassName"] == runtime
    container = data["spec"]["containers"][0]
    assert container["resources"] == ({"limits": {"nvidia.com/pgpu": "1"}} if gpu_count == "1" else {})
    assert container["securityContext"]["privileged"] is False
    assert container["securityContext"]["runAsNonRoot"] is True
    assert container["command"] == ["/bin/workload"]


def service_reconstruct(tmp_path, auth, fragment):
    incoming = tmp_path / "handoff"
    incoming.mkdir(exist_ok=True)
    (incoming / "release-authorization.json").write_text(json.dumps(auth))
    (incoming / "resource-policy-fragment.rego").write_text(fragment)
    (incoming / "cosign.pub").write_text("-----BEGIN PUBLIC KEY-----\nTEST\n-----END PUBLIC KEY-----\n")
    repository = IMAGE.split("@")[0]
    policy = {
        "default": [{"type": "reject"}],
        "transports": {
            "docker": {
                repository: [
                    {
                        "type": "sigstoreSigned",
                        "keyPath": "kbs:///default/sig-public-key/demo",
                        "signedIdentity": {"type": "matchRepository"},
                    }
                ]
            }
        },
    }
    (incoming / "image-security-policy.json").write_text(json.dumps(policy))
    result = tmp_path / "reviewed.rego"
    block = embedded(COCO / "service/12-install-trusted-service-handoff.sh", "received = json.loads")
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            block,
            str(incoming),
            "secure-services.example.com:5000",
            str(COCO / "service/policies/workload-resource-policy.rego.template"),
            str(result),
            str(COCO / "service/lib/workload-release.py"),
            str(COCO / "service/policies/workload-resource-policy-v1.rego.template"),
        ],
        capture_output=True,
        text=True,
    )
    return process, result


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_secure_services_reconstructs_only_approved_target_fragment(tmp_path, runtime):
    auth = authorization(runtime)
    fragment = release.render_fragment(auth, TEMPLATE)
    process, result = service_reconstruct(tmp_path, auth, fragment)
    assert process.returncode == 0, process.stderr
    assert result.read_text() == fragment
    process, _ = service_reconstruct(tmp_path, auth, fragment + "\nallow := true\n")
    assert process.returncode != 0
    assert "differs from the secure-services template" in process.stderr


def test_secure_services_upgrades_legacy_fragment_without_installing_it(tmp_path):
    legacy = legacy_authorization()
    fragment = release.render_fragment(legacy, LEGACY_TEMPLATE, legacy=True)
    process, result = service_reconstruct(tmp_path, legacy, fragment)
    assert process.returncode == 0, process.stderr
    assert result.read_text() == release.render_fragment(legacy, TEMPLATE)
    assert result.read_text() != fragment


def test_expected_target_mismatch_is_checked_before_any_image_build():
    runner = (COCO / "admin/build_coco_image.sh").read_text()
    assert runner.index('source "${ADMIN_DIR}/lib/release.sh"') < runner.index('"${ADMIN_DIR}/10-build-plaintext.sh"')
    assert runner.index('python3 "${WORKLOAD_PROFILE_VALIDATOR}"') < runner.index(
        '"${ADMIN_DIR}/10-build-plaintext.sh"'
    )
    checks = (COCO / "admin/lib/release.sh").read_text()
    assert "${COCO_RUNTIME_CLASS:-$RUNTIME_CLASS}" in checks
    assert "${COCO_GPU_COUNT:-$EXPECTED_GPU_COUNT}" in checks


def test_unapproved_trust_vector_is_rejected():
    auth = copy.deepcopy(authorization("kata-qemu-nvidia-gpu-tdx"))
    auth["required_ear_trust_vectors"]["cpu0"]["configuration"] = 3
    with pytest.raises(ValueError, match="trust vectors"):
        release.validate_authorization(auth)
