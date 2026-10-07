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

"""SNP/TDX reference schema and fail-closed replacement tests (stdlib only)."""

import base64
import copy
import importlib.util
import json
import stat
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "examples/devops/coco/service/lib/platform-reference-values.py"
SPEC = importlib.util.spec_from_file_location("tdx_reference_installer", SOURCE)
INSTALLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSTALLER)


def profile(identifier="tdx-test", digit="a"):
    return {"id": identifier, **{key: digit * (16 if key == "xfam" else 96) for key in INSTALLER._schema.TDX_FIELDS}}


def tdx_values(*profiles):
    return {"schema": "coco-platform-reference-values/v2", "tee": "tdx", "profiles": list(profiles or [profile()])}


def snp_values():
    return {key: "a" * 96 if key == "snp_launch_measurement" else 1 for key in INSTALLER.FIELDS}


def test_complete_profile_replacement_uses_one_fixed_reference():
    values = tdx_values(profile("b", "b"), profile("a"))
    message = INSTALLER.reference_message(values)
    payload = json.loads(base64.b64decode(message["payload"]))
    assert payload == {"coco_tdx_profiles_v2": [profile("a"), profile("b", "b")]}
    assert message["type"] == "sample"


def test_reference_schema_requires_all_four_rtmrs():
    assert set(INSTALLER._schema.TDX_FIELDS) == {
        "mr_td",
        "rtmr_0",
        "rtmr_1",
        "rtmr_2",
        "rtmr_3",
        "xfam",
        "tdvfkernel",
        "tdvfkernelparams",
    }


def test_previous_six_field_profile_cannot_be_installed(monkeypatch):
    values = tdx_values()
    del values["profiles"][0]["rtmr_0"]
    del values["profiles"][0]["rtmr_3"]
    requests = []
    monkeypatch.setattr(INSTALLER, "post_reference", lambda message, *_: requests.append(message))
    with pytest.raises(ValueError, match="all eight measurement fields"):
        INSTALLER.install_references(values, "https://example.com", "ca", "token")
    assert not requests


@pytest.mark.parametrize("field", ["rtmr_0", "rtmr_3"])
def test_distinct_profiles_may_differ_only_in_one_rtmr(field):
    first = profile("first")
    second = profile("second")
    second[field] = "b" * 96
    values = tdx_values(first, second)
    assert INSTALLER.reference_payload(values) == {"coco_tdx_profiles_v2": [first, second]}


def test_existing_versioned_platform_profile_identifier_is_valid():
    values = tdx_values(profile("kata-3.29.0-tdx-approved-v1"))
    assert INSTALLER.validate_values(values) == values


@pytest.mark.parametrize("field", INSTALLER._schema.TDX_FIELDS)
@pytest.mark.parametrize("invalid", [True, 0, None, [], {}, "", "A" * 96, "f" * 95, "f" * 97, "0x" + "f" * 96])
def test_rejects_malformed_measurements(field, invalid):
    values = tdx_values()
    values["profiles"][0][field] = invalid
    with pytest.raises(ValueError):
        INSTALLER.validate_values(values)


@pytest.mark.parametrize("identifier", ["", "A", "../other", "a/b", "x y", "x\n", "a" * 65, 42])
def test_rejects_untrusted_profile_identifiers(identifier):
    with pytest.raises(ValueError):
        INSTALLER.validate_values(tdx_values(profile(identifier)))


@pytest.mark.parametrize("key", ["schema", "tee", "profiles"])
def test_missing_or_extra_schema_keys_fail(key):
    values = tdx_values()
    del values[key]
    with pytest.raises(ValueError):
        INSTALLER.validate_values(values)
    values = tdx_values()
    values["unexpected"] = "ignored?"
    with pytest.raises(ValueError):
        INSTALLER.validate_values(values)


@pytest.mark.parametrize("profiles", [[], [profile(), profile()], [profile("a"), profile("b")], [profile()] * 65])
def test_profile_duplicates_and_bounds(profiles):
    with pytest.raises(ValueError):
        INSTALLER.validate_values({"schema": "coco-platform-reference-values/v2", "tee": "tdx", "profiles": profiles})


@pytest.mark.parametrize("field", ["id", *INSTALLER._schema.TDX_FIELDS])
def test_incomplete_profile_is_not_approved(field):
    values = tdx_values()
    del values["profiles"][0][field]
    with pytest.raises(ValueError):
        INSTALLER.validate_values(values)


def test_duplicate_json_keys_fail(tmp_path):
    path = tmp_path / "references.json"
    path.write_text('{"tee":"snp","tee":"tdx"}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        INSTALLER.load_values(path)


def test_legacy_snp_schema_and_complete_allowlist_are_preserved():
    values = snp_values()
    values["snp_launch_measurement"] = ["b" * 96, "a" * 96]
    assert INSTALLER.validate_values(values) == values
    assert INSTALLER.reference_payload(values)["snp_launch_measurement"] == ["a" * 96, "b" * 96]


@pytest.mark.parametrize("floor", [True, "1", -1, 256, None])
def test_snp_floors_stay_strict(floor):
    values = snp_values()
    values["snp_min_reported_tcb_microcode"] = floor
    with pytest.raises(ValueError):
        INSTALLER.validate_values(values)


def test_tdx_readback_requires_exact_set_but_allows_reordering():
    values = tdx_values(profile("a"), profile("b", "b"))
    INSTALLER.compare_reference(values, "coco_tdx_profiles_v2", json.dumps(list(reversed(values["profiles"]))))
    for actual in [values["profiles"][:1], values["profiles"] + [profile("c", "c")], None]:
        with pytest.raises(ValueError):
            INSTALLER.compare_reference(values, "coco_tdx_profiles_v2", json.dumps(actual))


@pytest.mark.parametrize("field", ["rtmr_0", "rtmr_3"])
def test_tdx_readback_rejects_changed_rtmr(field):
    values = tdx_values()
    actual = copy.deepcopy(values["profiles"])
    actual[0][field] = "b" * 96
    with pytest.raises(ValueError):
        INSTALLER.compare_reference(values, "coco_tdx_profiles_v2", json.dumps(actual))


def test_update_env_saves_private_validated_snapshot_without_touching_snp(tmp_path):
    env = tmp_path / "platform.env"
    env.write_text('SNP_LAUNCH_MEASUREMENT="existing"\nPLATFORM_REFERENCE_VALUES_FILE=""\n')
    values = tdx_values()
    INSTALLER.update_env(values, env)
    snapshot = tmp_path / "approved-platform-reference-values.json"
    assert INSTALLER.load_values(snapshot) == values
    assert stat.S_IMODE(snapshot.stat().st_mode) == 0o600
    assert 'SNP_LAUNCH_MEASUREMENT="existing"' in env.read_text()
    assert str(snapshot) in env.read_text()


def test_update_env_rejects_reference_snapshot_symlink(tmp_path):
    env = tmp_path / "platform.env"
    env.write_text("")
    (tmp_path / "approved-platform-reference-values.json").symlink_to(tmp_path / "other")
    with pytest.raises(ValueError, match="symlink"):
        INSTALLER.update_env(tdx_values(), env)


def test_tdx_install_is_one_reference_replacement_and_leaves_snp_untouched(monkeypatch):
    requests = []
    monkeypatch.setattr(INSTALLER, "post_reference", lambda message, *_: requests.append(message))
    INSTALLER.install_references(tdx_values(), "https://example.com", "ca", "token")
    assert len(requests) == 1
    assert set(json.loads(base64.b64decode(requests[0]["payload"]))) == {"coco_tdx_profiles_v2"}


def test_snp_update_blocks_launches_before_floors_and_activates_last(monkeypatch):
    requests = []
    monkeypatch.setattr(
        INSTALLER,
        "post_reference",
        lambda message, *_: requests.append(json.loads(base64.b64decode(message["payload"]))),
    )
    INSTALLER.install_references(snp_values(), "https://example.com", "ca", "token")
    assert requests[0] == {"snp_launch_measurement": []}
    assert set(requests[1]) == set(INSTALLER.FIELDS) - {"snp_launch_measurement"}
    assert requests[2] == {"snp_launch_measurement": ["a" * 96]}
    assert all("coco_tdx_profiles_v2" not in request for request in requests)


def test_failed_snp_floor_update_does_not_activate_measurements(monkeypatch):
    requests = []

    def post(message, *_):
        requests.append(json.loads(base64.b64decode(message["payload"])))
        if len(requests) == 2:
            raise OSError("simulated transport error")

    monkeypatch.setattr(INSTALLER, "post_reference", post)
    with pytest.raises(OSError):
        INSTALLER.install_references(snp_values(), "https://example.com", "ca", "token")
    assert len(requests) == 2
    assert requests[0] == {"snp_launch_measurement": []}


def test_tdx_claimant_cannot_supply_reference_path():
    values = copy.deepcopy(tdx_values())
    values["profiles"][0]["reference_path"] = "snp_launch_measurement"
    with pytest.raises(ValueError):
        INSTALLER.reference_message(values)
