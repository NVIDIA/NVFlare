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

"""Evaluate the reviewed Trustee GPU policy with real OPA and stubbed RVPS lookup."""

import copy
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from nvflare.lighter.cc_provision import workload_release as release

ROOT = Path(__file__).resolve().parents[5]
POLICY = ROOT / "examples/devops/coco/service/policies/default_gpu.rego"
PINNED = Path(__file__).parent / "fixtures/ear_default_policy_gpu.rego"
PINNED_BLOB_SHA = "0cd04bb6e236653a3d633bcefa1da8c3e42eabfd"
OPA = os.environ.get("COCO_TEST_OPA") or shutil.which("opa")
ZERO_PADDING = ("file-system", "instance-identity", "runtime-opaque", "storage-opaque", "sourced-data")


def test_gpu_policy_is_pinned_trustee_policy_with_only_padding_claims_removed():
    pinned = PINNED.read_bytes().replace(b"\n}\n\nelse := 3 if {", b"\n} \n\nelse := 3 if {")
    blob = b"blob " + str(len(pinned)).encode() + b"\0" + pinned
    assert hashlib.sha1(blob).hexdigest() == PINNED_BLOB_SHA

    lines = pinned.decode().splitlines()
    removed = [line for line in lines if any(f'"{name}":' in line for name in ZERO_PADDING)]
    assert len(removed) == len(ZERO_PADDING)
    expected = [line for line in lines if line not in removed]

    def substantive(source):
        return [line.rstrip() for line in source if line.strip() and not line.lstrip().startswith("#")]

    assert substantive(POLICY.read_text().splitlines()) == substantive(expected)


def approved_evidence():
    return {
        "nvidia": {
            "x-nvidia-gpu-attestation-report-cert-chain": {
                "x-nvidia-cert-ocsp-status": "good",
                "x-nvidia-cert-status": "valid",
            },
            "x-nvidia-gpu-attestation-report-cert-chain-fwid-match": True,
            "x-nvidia-gpu-attestation-report-parsed": True,
            "x-nvidia-gpu-attestation-report-signature-verified": True,
            "x-nvidia-gpu-arch-check": True,
            "secboot": True,
            "dbgstat": "disabled",
            "x-nvidia-gpu-vbios-version": "approved-vbios",
            "x-nvidia-gpu-driver-version": "approved-driver",
            "x-nvidia-gpu-vbios-rim-cert-chain": {
                "x-nvidia-cert-ocsp-status": "good",
                "x-nvidia-cert-status": "valid",
            },
            "x-nvidia-gpu-driver-rim-fetched": True,
            "x-nvidia-gpu-driver-rim-measurements-available": True,
            "x-nvidia-gpu-driver-rim-schema-validated": True,
            "x-nvidia-gpu-driver-rim-signature-verified": True,
            "x-nvidia-gpu-driver-rim-version-match": True,
            "x-nvidia-gpu-vbios-rim-fetched": True,
            "x-nvidia-gpu-vbios-rim-measurements-available": True,
            "x-nvidia-gpu-vbios-rim-schema-validated": True,
            "x-nvidia-gpu-vbios-rim-signature-verified": True,
            "x-nvidia-gpu-vbios-rim-version-match": True,
            "measres": "success",
        }
    }


@pytest.fixture
def evaluate(tmp_path):
    if not OPA:
        pytest.skip("set COCO_TEST_OPA to the service-pinned OPA 1.8.0 executable")
    policy = tmp_path / "gpu.rego"
    policy.write_text(POLICY.read_text() + "\nquery_reference_value(name) := data.references[name]\n")
    data = tmp_path / "references.json"
    default_refs = {"allowed_vbios_versions": ["approved-vbios"], "allowed_driver_versions": ["approved-driver"]}

    def run(claims, references=None, sample_references=None):
        data.write_text(
            json.dumps(
                {"references": default_refs if references is None else references, "reference": sample_references or {}}
            )
        )
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
                str(data),
                "data.policy.trust_claims",
            ],
            input=json.dumps(claims),
            text=True,
            capture_output=True,
            check=True,
        )
        return json.loads(result.stdout)

    return run


def test_approved_gpu_has_exact_three_complete_claims(evaluate):
    assert evaluate(approved_evidence()) == {"executables": 3, "hardware": 2, "configuration": 2}


@pytest.mark.parametrize("field", ["x-nvidia-gpu-vbios-version", "x-nvidia-gpu-driver-version"])
@pytest.mark.parametrize("invalid", [None, "unapproved"])
def test_gpu_unapproved_or_missing_version_never_gets_configuration_two(evaluate, field, invalid):
    claims = approved_evidence()
    if invalid is None:
        del claims["nvidia"][field]
    else:
        claims["nvidia"][field] = invalid
    result = evaluate(claims)
    assert result == {"executables": 3, "hardware": 2, "configuration": 3}


@pytest.mark.parametrize("reference", [None, [], "approved-vbios", {"bad": "unapproved-vbios"}])
def test_missing_or_malformed_vbios_reference_does_not_approve(evaluate, reference):
    refs = {"allowed_vbios_versions": reference, "allowed_driver_versions": ["approved-driver"]}
    assert evaluate(approved_evidence(), refs)["configuration"] != 2


@pytest.mark.parametrize("reference_id", ["allowed_vbios_versions", "allowed_driver_versions"])
def test_json_encoded_string_reference_is_not_an_approved_list(evaluate, reference_id):
    # Trustee's Rego extension receives the decoded RVPS JSON value. A string
    # containing JSON is a different value and must not accidentally approve.
    refs = {"allowed_vbios_versions": ["approved-vbios"], "allowed_driver_versions": ["approved-driver"]}
    refs[reference_id] = json.dumps(refs[reference_id])
    assert evaluate(approved_evidence(), refs)["configuration"] != 2


@pytest.mark.parametrize(
    ("path", "invalid"),
    [
        (("x-nvidia-gpu-attestation-report-cert-chain", "x-nvidia-cert-ocsp-status"), "revoked"),
        (("x-nvidia-gpu-attestation-report-cert-chain", "x-nvidia-cert-status"), "invalid"),
        (("x-nvidia-gpu-attestation-report-cert-chain-fwid-match",), False),
        (("x-nvidia-gpu-attestation-report-parsed",), False),
        (("x-nvidia-gpu-attestation-report-signature-verified",), False),
        (("x-nvidia-gpu-arch-check",), False),
    ],
)
def test_nras_report_checks_still_gate_hardware(evaluate, path, invalid):
    claims = approved_evidence()
    target = claims["nvidia"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = invalid
    assert evaluate(claims)["hardware"] == 97


@pytest.mark.parametrize(
    ("path", "invalid"),
    [
        (("x-nvidia-gpu-vbios-rim-cert-chain", "x-nvidia-cert-ocsp-status"), "revoked"),
        (("x-nvidia-gpu-vbios-rim-cert-chain", "x-nvidia-cert-status"), "invalid"),
        (("x-nvidia-gpu-driver-rim-fetched",), False),
        (("x-nvidia-gpu-driver-rim-measurements-available",), False),
        (("x-nvidia-gpu-driver-rim-schema-validated",), False),
        (("x-nvidia-gpu-driver-rim-signature-verified",), False),
        (("x-nvidia-gpu-driver-rim-version-match",), False),
        (("x-nvidia-gpu-vbios-rim-fetched",), False),
        (("x-nvidia-gpu-vbios-rim-measurements-available",), False),
        (("x-nvidia-gpu-vbios-rim-schema-validated",), False),
        (("x-nvidia-gpu-vbios-rim-signature-verified",), False),
        (("x-nvidia-gpu-vbios-rim-version-match",), False),
        (("measres",), "failed"),
    ],
)
def test_rim_and_measurement_checks_still_gate_executables(evaluate, path, invalid):
    claims = approved_evidence()
    target = claims["nvidia"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = invalid
    assert evaluate(claims)["executables"] == 33


@pytest.mark.parametrize(("field", "invalid", "expected"), [("secboot", False, 36), ("dbgstat", "enabled", 36)])
def test_secure_boot_and_debug_checks_still_gate_configuration(evaluate, field, invalid, expected):
    claims = approved_evidence()
    claims["nvidia"][field] = invalid
    assert evaluate(claims)["configuration"] == expected


def test_cpu_only_evidence_cannot_approve_gpu_policy(evaluate):
    cpu = {"snp": {"measurement": "approved", "policy_debug_allowed": False}}
    assert evaluate(cpu) == {"executables": 33, "hardware": 97, "configuration": 36}
    # The pinned upstream sample-device branch can set hardware alone; it must
    # never turn non-NVIDIA evidence into a complete GPU appraisal.
    cpu["sampledevice"] = {"svn": 1}
    refs = {
        "allowed_vbios_versions": ["approved-vbios"],
        "allowed_driver_versions": ["approved-driver"],
        "device_svn": [1],
    }
    assert evaluate(cpu, refs, {"device_svn": [1]}) == {"executables": 33, "hardware": 2, "configuration": 36}


def test_partial_nvidia_evidence_never_has_all_approved_claims(evaluate):
    claims = approved_evidence()
    claims["nvidia"].pop("x-nvidia-gpu-attestation-report-signature-verified")
    claims["nvidia"].pop("x-nvidia-gpu-vbios-rim-signature-verified")
    result = evaluate(claims)
    assert result != {"executables": 3, "hardware": 2, "configuration": 2}
    assert result["hardware"] == 97
    assert result["executables"] == 33


@pytest.mark.parametrize("runtime", ["kata-qemu-nvidia-gpu-snp", "kata-qemu-nvidia-gpu-tdx"])
def test_actual_gpu_appraisal_drives_rendered_release_decision(evaluate, tmp_path, runtime):
    auth = release.build_authorization(
        runtime,
        "gpu-contract",
        "registry.example.test/workload@sha256:" + "c" * 64,
        ["/bin/workload"],
        "ab" * 32,
        ["default/" + kind + "/gpu-contract" for kind in ("image-key", "sig-public-key", "security-policy")],
    )
    template = (ROOT / "examples/devops/coco/service/policies/workload-resource-policy.rego.template").read_text()
    policy = tmp_path / "release.rego"
    policy.write_text(
        "package policy\nimport rego.v1\ndefault allow := false\n\n" + release.render_fragment(auth, template)
    )
    data, token, _ = release.policy_test_cases(auth)["positive"]
    datafile = tmp_path / "release-data.json"
    datafile.write_text(json.dumps(data))
    gpu_vector = token["submods"]["gpu0"]["ear.trustworthiness-vector"]
    approved_vector = evaluate(approved_evidence())
    assert approved_vector == auth["required_ear_trust_vectors"]["gpu0"]

    def permitted(vector):
        candidate = copy.deepcopy(token)
        candidate["submods"]["gpu0"]["ear.trustworthiness-vector"] = vector
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
            input=json.dumps(candidate),
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.strip() == "true"

    # The release rule receives the actual three-key AS result, not a
    # hand-built approximation of the GPU trust vector.
    assert permitted(approved_vector)
    assert gpu_vector == approved_vector
    for name in ZERO_PADDING:
        assert not permitted({**approved_vector, name: 0}), name
    unapproved = approved_evidence()
    unapproved["nvidia"]["x-nvidia-gpu-vbios-version"] = "unapproved-vbios"
    weak_vector = evaluate(unapproved)
    assert weak_vector["configuration"] == 3
    assert not permitted(weak_vector)
