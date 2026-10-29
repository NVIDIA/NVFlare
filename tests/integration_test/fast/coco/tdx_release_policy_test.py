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

"""CoCo external-runtime contracts; run serially outside unit workers."""

import json
import os
import re
import shutil
import sys
from pathlib import Path

import pytest
import yaml

from nvflare.lighter.cc_provision import workload_release as release
from tests.coco_test_utils.process import run_coco_command
from tests.coco_test_utils.tdx_release_policy import authorization, legacy_authorization

ROOT = Path(__file__).resolve().parents[4]

COCO = ROOT / "examples/devops/coco"

TEMPLATE = (COCO / "service/policies/workload-resource-policy.rego.template").read_text()

LEGACY_TEMPLATE = (COCO / "service/policies/workload-resource-policy-v1.rego.template").read_text()

OPA = os.environ.get("COCO_TEST_OPA") or shutil.which("opa")

RUNTIMES = list(release.RUNTIME_TARGETS)

DIGEST = "ab" * 32

IMAGE = "secure-services.example.com:5000/workload@sha256:" + "c" * 64

PATHS = ["default/" + kind + "/demo" for kind in ("image-key", "sig-public-key", "security-policy")]


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
    run_coco_command([OPA, "check", "--strict", str(policy)], check=True, capture_output=True, text=True)
    for name, (data, token, expected) in release.policy_test_cases(auth).items():
        datafile = tmp_path / "data.json"
        datafile.write_text(json.dumps(data))
        result = run_coco_command(
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
    run_coco_command(
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
    block = (
        "import runpy, sys\nfrom pathlib import Path\n"
        "helper = runpy.run_path(sys.argv[1])\n"
        "_, _, fragment = helper['release_inputs'](Path(sys.argv[2]))\n"
        "Path(sys.argv[3]).write_text(fragment)\n"
    )
    process = run_coco_command(
        [
            sys.executable,
            "-c",
            block,
            str(COCO / "service/lib/install-workload-release.py"),
            str(incoming),
            str(result),
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
    assert "differs from the service-owned template" in process.stderr


def test_secure_services_upgrades_legacy_fragment_without_installing_it(tmp_path):
    legacy = legacy_authorization()
    fragment = release.render_fragment(legacy, LEGACY_TEMPLATE, legacy=True)
    process, result = service_reconstruct(tmp_path, legacy, fragment)
    assert process.returncode == 0, process.stderr
    assert result.read_text() == release.render_fragment(legacy, TEMPLATE)
    assert result.read_text() != fragment
