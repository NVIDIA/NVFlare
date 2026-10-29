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


"""In-process CoCo policy and validation contracts."""

import copy
from pathlib import Path

import pytest

from nvflare.lighter.cc_provision import workload_release as release
from tests.coco_test_utils.tdx_release_policy import authorization, legacy_authorization

ROOT = Path(__file__).resolve().parents[5]

COCO = ROOT / "examples/devops/coco"

LEGACY_TEMPLATE = (COCO / "service/policies/workload-resource-policy-v1.rego.template").read_text()

RUNTIMES = list(release.RUNTIME_TARGETS)

DIGEST = "ab" * 32

IMAGE = "secure-services.example.com:5000/workload@sha256:" + "c" * 64

PATHS = ["default/" + kind + "/demo" for kind in ("image-key", "sig-public-key", "security-policy")]


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
