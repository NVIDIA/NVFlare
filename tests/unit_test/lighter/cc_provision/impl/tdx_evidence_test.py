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

"""Host integration fail-closed tests; not a substitute for real quote verification."""

import base64
import copy
import importlib.util
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[5]
WORKFLOW = ROOT / "examples/devops/coco/trusted_system/lib/platform-evidence.py"
spec = importlib.util.spec_from_file_location("tdx_reference", WORKFLOW)
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)
collector_spec = importlib.util.spec_from_file_location(
    "tdx_collector", WORKFLOW.parent.parent / "rehearsal-collector/collect-tdx-evidence.py"
)
collector = importlib.util.module_from_spec(collector_spec)
collector_spec.loader.exec_module(collector)


def verified_claims():
    return {
        "quote": {
            "body": {
                "mr_td": "a" * 96,
                "rtmr_0": "1" * 96,
                "rtmr_1": "b" * 96,
                "rtmr_2": "c" * 96,
                "rtmr_3": "2" * 96,
                "xfam": "d" * 16,
            }
        },
        "uefi_event_logs": [
            {
                "type_name": "EV_EFI_BOOT_SERVICES_APPLICATION",
                "details": {"device_paths": ["File(kernel)"]},
                "digests": [{"alg": "SHA-384", "digest": "e" * 96}],
            },
            {
                "type_name": "EV_EVENT_TAG",
                "details": {"string": "LOADED_IMAGE::LoadOptions"},
                "digests": [{"alg": "SHA-384", "digest": "f" * 96}],
            },
        ],
    }


def test_complete_verified_profile():
    result = workflow.extract_profile(verified_claims(), "tdx-profile-1")
    assert result == {
        "id": "tdx-profile-1",
        "mr_td": "a" * 96,
        "rtmr_0": "1" * 96,
        "rtmr_1": "b" * 96,
        "rtmr_2": "c" * 96,
        "rtmr_3": "2" * 96,
        "xfam": "d" * 16,
        "tdvfkernel": "e" * 96,
        "tdvfkernelparams": "f" * 96,
    }


@pytest.mark.parametrize("field", ["mr_td", "rtmr_0", "rtmr_1", "rtmr_2", "rtmr_3", "xfam"])
@pytest.mark.parametrize("value", [None, 0, "", "A" * 96, "g" * 96, "0" * 64])
def test_invalid_verified_field_rejected(field, value):
    claims = verified_claims()
    claims["quote"]["body"][field] = value
    with pytest.raises(ValueError, match="Invalid verified"):
        workflow.extract_profile(claims, "tdx")


@pytest.mark.parametrize("field", ["mr_td", "rtmr_0", "rtmr_1", "rtmr_2", "rtmr_3", "xfam"])
def test_missing_verified_field_rejected(field):
    claims = verified_claims()
    del claims["quote"]["body"][field]
    with pytest.raises(ValueError, match="Invalid verified"):
        workflow.extract_profile(claims, "tdx")


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "sha256", "bad_digest", "tdshim"])
@pytest.mark.parametrize("event_index", [0, 1])
def test_incomplete_or_ambiguous_event_log_rejected(mutation, event_index):
    claims = verified_claims()
    event = claims["uefi_event_logs"][event_index]
    if mutation == "missing":
        claims["uefi_event_logs"].pop(event_index)
    elif mutation == "duplicate":
        claims["uefi_event_logs"].append(copy.deepcopy(event))
    elif mutation == "sha256":
        event["digests"][0]["alg"] = "SHA-256"
    elif mutation == "bad_digest":
        event["digests"][0]["digest"] = "x" * 96
    else:
        event["details"] = {"string": "td_payload"}
    with pytest.raises(ValueError, match="exactly one verified TDVF"):
        workflow.extract_profile(claims, "tdx")


def test_frame_decode_requires_raw_quote_and_ccel():
    evidence = {"quote": "cXVvdGU=", "cc_eventlog": "Y2NlbA=="}
    frame = workflow.FRAME + base64.b64encode(json.dumps(evidence).encode()).decode()
    assert workflow.decode_evidence(frame) == evidence
    with pytest.raises(ValueError):
        workflow.decode_evidence(frame + "\n" + frame)
    with pytest.raises(ValueError):
        workflow.decode_evidence(workflow.FRAME + "notbase64")
    for missing in ("quote", "cc_eventlog"):
        bad = dict(evidence)
        bad.pop(missing)
        with pytest.raises(ValueError):
            workflow.decode_evidence(workflow.FRAME + base64.b64encode(json.dumps(bad).encode()).decode())


def prepare_tdx_inputs(directory, nonce=b"a" * 64):
    (directory / "request-data.bin").write_bytes(nonce)
    (directory / "evidence.json").write_text('{"quote":"raw-quote","cc_eventlog":"raw-ccel"}')
    (directory / "initdata.toml").write_text("reviewed initdata")


def test_verification_failure_cannot_export_decoded_measurement(tmp_path):
    prepare_tdx_inputs(tmp_path)
    with patch.object(workflow.subprocess, "run", side_effect=subprocess.CalledProcessError(1, ["pinned-verifier"])):
        with pytest.raises(subprocess.CalledProcessError):
            workflow.verify_tdx(tmp_path, "pinned-verifier", "tdx")


def test_verify_uses_raw_report_nonce_and_exact_initdata(tmp_path):
    prepare_tdx_inputs(tmp_path, b"b" * 64)
    with patch.object(
        workflow.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, json.dumps(verified_claims()), "")
    ) as verify:
        profile, claims = workflow.verify_tdx(tmp_path, "pinned-verifier", "tdx")
    assert profile["mr_td"] == "a" * 96
    assert claims == verified_claims()
    command = verify.call_args.args[0]
    assert command[0] == "pinned-verifier"
    paths = [Path(value) for value in command[1:]]
    assert [path.name for path in paths] == ["evidence.json", "request-data.bin", "initdata.toml"]
    assert all(path.parent == paths[0].parent and path.parent.parent == tmp_path for path in paths)
    assert paths[0].parent.name.startswith(".tdx-verify-")
    assert not paths[0].parent.exists()
    assert verify.call_args.kwargs["check"] is True
    assert verify.call_args.kwargs["timeout"] == 300


@pytest.mark.parametrize("nonce", [b"", b"a" * 63, b"a" * 65])
def test_verify_rejects_invalid_nonce_before_verifier(tmp_path, nonce):
    (tmp_path / "request-data.bin").write_bytes(nonce)
    with patch.object(workflow.subprocess, "run") as verifier:
        with pytest.raises(ValueError, match="64 bytes"):
            workflow.verify_tdx(tmp_path, "pinned-verifier", "tdx")
    verifier.assert_not_called()


@pytest.mark.parametrize("nonce", [b"", b"x" * 64, b"a" * 63, b"A" * 64, b"a" * 65, bytes(range(64))])
def test_collector_rejects_invalid_nonce_without_network(tmp_path, nonce):
    path = tmp_path / "nonce"
    path.write_bytes(nonce)
    with patch.object(collector.urllib.request, "build_opener") as opener:
        with pytest.raises(ValueError, match="challenge"):
            collector.collect(path)
        opener.assert_not_called()


def test_collector_redirects_forbidden():
    with pytest.raises(ValueError, match="redirect"):
        collector.NoRedirect().redirect_request(None, None, 302, "", {}, "https://elsewhere")


def test_verifier_duplicate_claim_fields_rejected(tmp_path):
    prepare_tdx_inputs(tmp_path, b"b" * 64)
    with patch.object(
        workflow.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, '{"quote":{},"quote":{}}', "")
    ):
        with pytest.raises(ValueError, match="Duplicate evidence"):
            workflow.verify_tdx(tmp_path, "pinned-verifier", "tdx")


def test_tdx_verifies_private_snapshot_when_original_inputs_are_replaced(tmp_path):
    prepare_tdx_inputs(tmp_path)
    originals = {
        name: (tmp_path / name).read_bytes() for name in ("request-data.bin", "evidence.json", "initdata.toml")
    }
    snapshots = []

    def replace_inputs(command, **kwargs):
        snapshots.extend(Path(value) for value in command[1:])
        for name in originals:
            (tmp_path / name).write_bytes(b"replaced original")
        for path in snapshots:
            assert path.parent.parent == tmp_path
            assert path.read_bytes() == originals[path.name]
            assert path.stat().st_mode & 0o777 == 0o600
            assert path.parent.stat().st_mode & 0o777 == 0o700
        return subprocess.CompletedProcess(command, 0, json.dumps(verified_claims()), "")

    with patch.object(workflow.subprocess, "run", side_effect=replace_inputs):
        profile, _ = workflow.verify_tdx(tmp_path, "pinned-verifier", "tdx")
    assert profile["mr_td"] == "a" * 96
    assert all(not path.exists() for path in snapshots)
