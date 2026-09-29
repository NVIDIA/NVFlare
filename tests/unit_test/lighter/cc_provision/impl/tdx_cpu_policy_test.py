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

"""Evaluate the actual service CPU policy with pinned OPA; only RVPS is stubbed."""

import copy
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[5]
OPA = os.environ.get("COCO_TEST_OPA") or shutil.which("opa")
pytestmark = pytest.mark.skipif(not OPA, reason="set COCO_TEST_OPA to the service-pinned OPA 1.8.0 executable")


def profile(identifier="approved", digit="a"):
    return {
        "id": identifier,
        **{key: digit * 96 for key in ("mr_td", "rtmr_1", "rtmr_2", "tdvfkernel", "tdvfkernelparams")},
        "xfam": digit * 16,
    }


def evidence(reference=None):
    reference = reference or profile()
    return {
        "tdx": {
            "quote": {
                "header": {"tee_type": "81000000", "vendor_id": "939a7233f79c4ca9940a0db3957f0607"},
                "body": {field: reference[field] for field in ("mr_td", "rtmr_1", "rtmr_2", "xfam")},
            },
            "tcb_status": "UpToDate",
            "collateral_expiration_status": "0",
            "td_attributes": {"debug": False},
            "uefi_event_logs": [
                {
                    "type_name": "EV_EFI_BOOT_SERVICES_APPLICATION",
                    "details": {"device_paths": ["File(kernel)"]},
                    "digests": [{"alg": "SHA-384", "digest": reference["tdvfkernel"]}],
                },
                {
                    "type_name": "EV_EVENT_TAG",
                    "details": {"string": "LOADED_IMAGE::LoadOptions"},
                    "digests": [{"alg": "SHA-384", "digest": reference["tdvfkernelparams"]}],
                },
            ],
        }
    }


@pytest.fixture
def evaluate(tmp_path):
    policy = tmp_path / "cpu.rego"
    policy.write_text(
        (ROOT / "examples/devops/coco/service/policies/default_cpu.rego").read_text()
        + "\nquery_reference_value(name) := data.references[name]\n"
    )
    data = tmp_path / "references.json"

    def run(claims, profiles=None, references=None):
        refs = references if references is not None else {"coco_tdx_profiles_v2": profiles or [profile()]}
        data.write_text(json.dumps({"references": refs}))
        result = subprocess.run(
            [OPA, "eval", "--format", "raw", "--stdin-input", "-d", str(policy), "-d", str(data), "data.policy"],
            input=json.dumps(claims),
            text=True,
            capture_output=True,
            check=True,
        )
        return json.loads(result.stdout)

    return run


def approved(result):
    claims = result["trust_claims"]
    return [claims[key] for key in ("executables", "hardware", "configuration")] == [3, 2, 2]


def test_tdx_complete_profile_passes(evaluate):
    assert approved(evaluate(evidence()))


def test_either_complete_profile_passes_without_cross_profile_mixing(evaluate):
    profiles = [profile(), profile("second", "b")]
    assert approved(evaluate(evidence(profiles[0]), profiles))
    assert approved(evaluate(evidence(profiles[1]), profiles))
    for field in ("mr_td", "rtmr_1", "rtmr_2", "xfam"):
        claims = evidence()
        claims["tdx"]["quote"]["body"][field] = profiles[1][field]
        assert not approved(evaluate(claims, profiles))


@pytest.mark.parametrize("index", [0, 1])
def test_event_digest_cannot_be_borrowed_from_other_profile(evaluate, index):
    claims = evidence()
    claims["tdx"]["uefi_event_logs"][index]["digests"][0]["digest"] = "b" * 96
    assert not approved(evaluate(claims, [profile(), profile("second", "b")]))


@pytest.mark.parametrize("field", ["mr_td", "rtmr_1", "rtmr_2", "xfam"])
@pytest.mark.parametrize("invalid", [None, True, 0, "", "f" * 96])
def test_invalid_or_unapproved_quote_measurements_rejected(evaluate, field, invalid):
    claims = evidence()
    claims["tdx"]["quote"]["body"][field] = invalid
    assert not approved(evaluate(claims))


@pytest.mark.parametrize("status", [None, "OutOfDate", "ConfigurationNeeded", "SWHardeningNeeded", "UpToDate ", 0])
def test_bad_tcb_is_not_approved(evaluate, status):
    claims = evidence()
    claims["tdx"]["tcb_status"] = status
    assert evaluate(claims)["hardware"] == 97


@pytest.mark.parametrize("status", [None, "OutOfDate", "ConfigurationNeeded", True])
def test_reported_current_tcb_must_be_up_to_date(evaluate, status):
    claims = evidence()
    claims["tdx"]["tcb_status_current"] = status
    assert evaluate(claims)["hardware"] == 97


def test_current_tcb_up_to_date_passes(evaluate):
    claims = evidence()
    claims["tdx"]["tcb_status_current"] = "UpToDate"
    assert approved(evaluate(claims))


@pytest.mark.parametrize("status", [None, "1", 0, False, ""])
def test_expired_or_malformed_collateral_status_rejected(evaluate, status):
    claims = evidence()
    claims["tdx"]["collateral_expiration_status"] = status
    assert evaluate(claims)["hardware"] == 97


@pytest.mark.parametrize("debug", [None, True, "false", 0])
def test_debug_missing_or_enabled_rejected(evaluate, debug):
    claims = evidence()
    claims["tdx"]["td_attributes"]["debug"] = debug
    assert evaluate(claims)["configuration"] == 36


@pytest.mark.parametrize("field", ["tee_type", "vendor_id"])
def test_wrong_quote_header_rejected(evaluate, field):
    claims = evidence()
    claims["tdx"]["quote"]["header"][field] = "unknown"
    assert evaluate(claims)["hardware"] == 97


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "bad_digest", "wrong_alg", "duplicate_digest", "bad_type"]
)
def test_missing_or_ambiguous_event_log_never_falls_back(evaluate, index, mutation):
    claims = evidence()
    events = claims["tdx"]["uefi_event_logs"]
    if mutation == "missing":
        events.pop(index)
    elif mutation == "duplicate":
        events.append(copy.deepcopy(events[index]))
    elif mutation == "bad_digest":
        events[index]["digests"][0]["digest"] = "f" * 96
    elif mutation == "wrong_alg":
        events[index]["digests"][0]["alg"] = "SHA-256"
    elif mutation == "duplicate_digest":
        events[index]["digests"].append(copy.deepcopy(events[index]["digests"][0]))
    else:
        events[index]["type_name"] = "GRUB"
    assert evaluate(claims)["executables"] == 33


@pytest.mark.parametrize("references", [{}, {"coco_tdx_profiles_v2": None}, {"coco_tdx_profiles_v2": []}])
def test_missing_or_empty_references_fail_closed(evaluate, references):
    assert not approved(evaluate(evidence(), references=references))


def test_independent_legacy_tdx_references_do_not_bypass_profile_approval(evaluate):
    refs = {field: [value] for field, value in profile().items() if field != "id"}
    assert not approved(evaluate(evidence(), references=refs))


def test_snp_coexists_and_mixed_evidence_is_rejected(evaluate):
    refs = {"coco_tdx_profiles_v2": [profile()], "snp_launch_measurement": ["approved-snp"]}
    refs.update({"snp_min_reported_tcb_" + name: 1 for name in ("bootloader", "tee", "snp", "microcode")})
    snp = {"measurement": "approved-snp", "policy_debug_allowed": False, "policy_migrate_ma": False}
    snp.update({"reported_tcb_" + name: 1 for name in ("bootloader", "tee", "snp", "microcode")})
    claims = evaluate({"snp": snp}, references=refs)["trust_claims"]
    assert [claims[key] for key in ("executables", "hardware", "configuration")] == [3, 2, 3]
    mixed = {**evidence(), "snp": snp}
    claims = evaluate(mixed, references=refs)["trust_claims"]
    assert [claims[key] for key in ("executables", "hardware", "configuration")] == [33, 97, 36]


def test_identifier_extensions_are_preserved(evaluate):
    claims = evidence()
    claims["init_data_claims"] = {
        "agent_policy_claims": {
            "containers": [
                {
                    "OCI": {
                        "Annotations": {"io.kubernetes.cri.image-name": "encrypted@sha256:abc"},
                        "Process": {"User": {"UID": 1000}},
                    }
                }
            ]
        }
    }
    ids = evaluate(claims)["extensions"][0]["value"]["validated"]
    assert ids == {"container_images": ["encrypted@sha256:abc"], "container_uids": [1000]}
