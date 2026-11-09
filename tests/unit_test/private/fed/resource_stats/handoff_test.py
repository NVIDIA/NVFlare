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

import json
import os
from copy import deepcopy
from unittest.mock import Mock

import pytest

from nvflare.private.fed.resource_stats import handoff as module


def _handoff():
    return {
        "duration_seconds": 10,
        "internal_version": module.INTERNAL_HANDOFF_VERSION,
        "kind": module.INTERNAL_HANDOFF_KIND,
        "resource_capacity": {"status": "unavailable"},
        "resource_time": {"status": "unavailable"},
        "workspace_filesystem": {"capacity_bytes": 1024},
        "resource_utilization": {
            "cpu_consumed": {"status": "unavailable"},
            "retained_content": {"status": "unavailable"},
            "message_traffic": {"status": "unavailable"},
        },
        "issues": [
            {"metric": "cpu_consumed", "code": "observation_incomplete"},
            {"metric": "message_traffic", "code": "observation_incomplete"},
            {"metric": "resource_capacity", "code": "observation_incomplete"},
            {"metric": "resource_time", "code": "observation_incomplete"},
            {"metric": "retained_content", "code": "unsupported"},
        ],
    }


def _complete_handoff():
    return {
        "duration_seconds": 10,
        "internal_version": module.INTERNAL_HANDOFF_VERSION,
        "kind": module.INTERNAL_HANDOFF_KIND,
        "resource_capacity": {"cpu": {"units": 2}, "memory": {"bytes": 4096}, "gpu": {"count": 0}},
        "resource_time": {
            "cpu": {"unit_seconds": 20},
            "memory": {"byte_seconds": 40960},
            "gpu": {"instance_seconds": 0},
        },
        "workspace_filesystem": {"capacity_bytes": 1024},
        "resource_utilization": {
            "cpu_consumed": {"cpu_seconds": 2},
            "retained_content": {"bytes": 128},
            "message_traffic": {"sent_bytes": 512, "received_bytes": 128},
        },
        "issues": [],
    }


def _link(path, target, *, directory):
    try:
        path.symlink_to(target, target_is_directory=directory)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are unavailable on this platform")


@pytest.mark.parametrize("version", ["1", "2", "3"])
def test_live_handoff_rejects_prior_versions(tmp_path, version):
    handoff = _handoff()
    handoff["internal_version"] = version
    module.write_terminal_handoff(tmp_path, handoff)
    assert module.read_terminal_handoff(tmp_path) is None

    handoff["resource_time"] = handoff.pop("resource_capacity")
    module.write_terminal_handoff(tmp_path, handoff)
    assert module.read_terminal_handoff(tmp_path) is None


@pytest.mark.parametrize("fallback", [False, True])
def test_atomic_writer_round_trip_and_replacement_without_leftover_temp_files(tmp_path, monkeypatch, fallback):
    if fallback:
        monkeypatch.setattr(module, "_SECURE_HANDOFF_WRITE_DIR_FD", False)
    run = tmp_path / "run"
    path = module.write_terminal_handoff(run, _handoff())
    assert module.read_terminal_handoff(run) == _handoff()
    updated = _handoff()
    updated["workspace_filesystem"] = {"capacity_bytes": 2048}
    updated["issues"] = [issue for issue in updated["issues"] if issue["metric"] != "workspace_filesystem"] + []
    module.write_terminal_handoff(run, updated)
    assert module.read_terminal_handoff(run) == updated
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize(
    "measurement", ["resource_capacity", "cpu_consumed", "workspace_filesystem", "retained_content", "message_traffic"]
)
def test_reader_isolates_a_malformed_measurement_without_erasing_other_values(tmp_path, measurement):
    original = _complete_handoff()
    path = module.write_terminal_handoff(tmp_path, original)
    malformed = deepcopy(original)
    source = (
        malformed if measurement in {"resource_capacity", "workspace_filesystem"} else malformed["resource_utilization"]
    )
    source[measurement] = {"unexpected": "field"}
    path.write_bytes(module.canonical_json_bytes(malformed))

    result = module.read_terminal_handoff(tmp_path)

    assert result is not None
    actual = result if measurement in {"resource_capacity", "workspace_filesystem"} else result["resource_utilization"]
    assert actual[measurement] == {"status": "unavailable"}
    assert {"metric": measurement, "code": "malformed_source"} in result["issues"]
    for name, value in original.items():
        if name not in {measurement, "resource_utilization", "issues"}:
            assert result[name] == value
    for name, value in original["resource_utilization"].items():
        if name != measurement:
            assert result["resource_utilization"][name] == value


@pytest.mark.parametrize("change", ["extra_field", "missing_field", "wrong_kind", "wrong_version"])
def test_reader_still_rejects_a_malformed_envelope(tmp_path, change):
    handoff = _complete_handoff()
    path = module.write_terminal_handoff(tmp_path, handoff)
    if change == "extra_field":
        handoff["participant_name"] = "untrusted-child-name"
    elif change == "missing_field":
        handoff["resource_utilization"].pop("cpu_consumed")
    elif change == "wrong_kind":
        handoff["kind"] = "not-a-terminal-handoff"
    else:
        handoff["internal_version"] = "1"
    path.write_bytes(module.canonical_json_bytes(handoff))

    assert module.read_terminal_handoff(tmp_path) is None


def test_reader_contains_plain_decoder_value_error(tmp_path, monkeypatch):
    path = module.write_terminal_handoff(tmp_path, _complete_handoff())
    payload = path.read_text()
    # Python's integer decoder can raise ValueError independently of JSON syntax.
    # Control that boundary without depending on a process-wide digit limit.
    decoder = Mock(side_effect=ValueError("integer string conversion limit exceeded"))
    monkeypatch.setattr(module.json, "loads", decoder)

    assert module.read_terminal_handoff(tmp_path) is None
    decoder.assert_called_once()
    assert decoder.call_args.args[0] == payload


def test_reader_preserves_native_numeric_values_at_the_parent_boundary(tmp_path):
    handoff = _complete_handoff()
    path = module.write_terminal_handoff(tmp_path, handoff)
    handoff["duration_seconds"] = 10.0
    handoff["resource_capacity"]["cpu"]["units"] = 2.0
    handoff["resource_utilization"]["cpu_consumed"]["cpu_seconds"] = 2.0
    path.write_bytes(module.canonical_json_bytes(handoff))
    assert module.read_terminal_handoff(tmp_path) == _complete_handoff()


@pytest.mark.parametrize("component", ["run", "resource_stats", "staging"])
@pytest.mark.parametrize("fallback", [False, True])
def test_writer_rejects_preexisting_directory_symlink_without_outside_writes(
    tmp_path, monkeypatch, component, fallback
):
    if fallback:
        monkeypatch.setattr(module, "_SECURE_HANDOFF_WRITE_DIR_FD", False)
    run = tmp_path / "run"
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / module.TERMINAL_HANDOFF_FILE
    sentinel.write_bytes(b"outside sentinel")
    link = {
        "run": run,
        "resource_stats": run / module.RESOURCE_STATS_DIR,
        "staging": run / module.RESOURCE_STATS_DIR / module.STAGING_DIR,
    }[component]
    link.parent.mkdir(parents=True, exist_ok=True)
    _link(link, outside, directory=True)
    with pytest.raises(OSError):
        module.write_terminal_handoff(run, _handoff())
    assert sentinel.read_bytes() == b"outside sentinel"
    assert list(outside.iterdir()) == [sentinel]


@pytest.mark.parametrize("fallback", [False, True])
def test_final_file_symlink_is_replaced_not_followed(tmp_path, monkeypatch, fallback):
    if fallback:
        monkeypatch.setattr(module, "_SECURE_HANDOFF_WRITE_DIR_FD", False)
    run = tmp_path / "run"
    path = module.terminal_handoff_path(run)
    path.parent.mkdir(parents=True)
    sentinel = tmp_path / "outside.json"
    sentinel.write_bytes(b"outside sentinel")
    _link(path, sentinel, directory=False)
    module.write_terminal_handoff(run, _handoff())
    assert sentinel.read_bytes() == b"outside sentinel"
    assert not path.is_symlink()
    assert module.read_terminal_handoff(run) == _handoff()


@pytest.mark.skipif(not module._SECURE_HANDOFF_WRITE_DIR_FD, reason="requires directory-relative safe writes")
def test_open_directory_cannot_be_redirected_to_outside_after_check(tmp_path, monkeypatch):
    run = tmp_path / "run"
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / module.TERMINAL_HANDOFF_FILE
    sentinel.write_bytes(b"outside sentinel")
    open_directory = module._open_handoff_directory

    def replaced_directory(*args, **kwargs):
        descriptor, path = open_directory(*args, **kwargs)
        assert descriptor is not None
        path.rename(path.with_name("original-staging"))
        _link(path, outside, directory=True)
        return descriptor, path

    monkeypatch.setattr(module, "_open_handoff_directory", replaced_directory)
    module.write_terminal_handoff(run, _handoff())
    assert sentinel.read_bytes() == b"outside sentinel"
    assert list(outside.iterdir()) == [sentinel]
    original = run / module.RESOURCE_STATS_DIR / "original-staging"
    assert [path.name for path in original.iterdir()] == [module.TERMINAL_HANDOFF_FILE]


@pytest.mark.parametrize("fallback", [False, True])
def test_replace_failure_preserves_previous_file_and_cleans_temporary_file(tmp_path, monkeypatch, fallback):
    if fallback:
        monkeypatch.setattr(module, "_SECURE_HANDOFF_WRITE_DIR_FD", False)
    run = tmp_path / "run"
    path = module.write_terminal_handoff(run, _handoff())
    old = path.read_bytes()

    def fail_replace(*args, **kwargs):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        module.write_terminal_handoff(run, _handoff())
    assert path.read_bytes() == old
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.skipif(not module._SECURE_HANDOFF_WRITE_DIR_FD, reason="requires directory-relative safe writes")
def test_temporary_name_collisions_are_bounded_and_do_not_overwrite_existing_file(tmp_path, monkeypatch):
    run = tmp_path / "run"
    path = module.write_terminal_handoff(run, _handoff())
    collision = path.parent / ".terminal-existing.tmp"
    collision.write_bytes(b"keep existing file")
    tokens = []

    def fixed_token(_length):
        tokens.append("existing")
        return "existing"

    monkeypatch.setattr(module.secrets, "token_hex", fixed_token)
    with pytest.raises(OSError, match="could not create"):
        module.write_terminal_handoff(run, _handoff())
    assert len(tokens) == 8
    assert collision.read_bytes() == b"keep existing file"
    assert module.read_terminal_handoff(run) == _handoff()


@pytest.mark.skipif(not module._SECURE_HANDOFF_WRITE_DIR_FD, reason="requires directory-relative safe writes")
def test_fdopen_failure_closes_descriptors_and_cleans_temporary_file(tmp_path, monkeypatch):
    run = tmp_path / "run"
    path = module.write_terminal_handoff(run, _handoff())
    descriptors = []
    original_open = os.open

    def record_open(*args, **kwargs):
        descriptor = original_open(*args, **kwargs)
        descriptors.append(descriptor)
        return descriptor

    def fail_fdopen(*args, **kwargs):
        raise OSError("simulated fdopen failure")

    monkeypatch.setattr(os, "open", record_open)
    monkeypatch.setattr(os, "fdopen", fail_fdopen)
    with pytest.raises(OSError, match="simulated fdopen failure"):
        module.write_terminal_handoff(run, _handoff())
    assert descriptors
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize(
    "tags",
    [[{"metric": "message_traffic", "code": "unknown"}], [{"metric": "message_traffic", "unexpected": True}], []],
)
def test_bad_metric_tags_preserve_independent_handoff_values(tmp_path, tags):
    original = _complete_handoff()
    path = module.write_terminal_handoff(tmp_path, original)
    malformed = deepcopy(original)
    malformed["resource_utilization"]["message_traffic"]["status"] = "partial"
    malformed["issues"] = tags
    path.write_bytes(module.canonical_json_bytes(malformed))

    result = module.read_terminal_handoff(tmp_path)

    assert result["resource_utilization"]["message_traffic"] == {"status": "unavailable"}
    assert result["issues"] == [{"metric": "message_traffic", "code": "malformed_source"}]
    assert result["duration_seconds"] == original["duration_seconds"]
    assert result["resource_capacity"] == original["resource_capacity"]
    assert result["resource_time"] == original["resource_time"]
    assert result["workspace_filesystem"] == original["workspace_filesystem"]
    for name in ("cpu_consumed", "retained_content"):
        assert result["resource_utilization"][name] == original["resource_utilization"][name]


@pytest.mark.parametrize("fallback", [False, True])
def test_generation_paths_preserve_replacement_against_late_writer_and_old_cleanup(tmp_path, monkeypatch, fallback):
    if fallback:
        monkeypatch.setattr(module, "_SECURE_HANDOFF_DIR_FD", False)
    old = module.write_terminal_handoff(tmp_path, _handoff(), launch_generation="old-launch")
    current = _handoff()
    current["workspace_filesystem"] = {"capacity_bytes": 8192}
    new = module.write_terminal_handoff(tmp_path, current, launch_generation="new-launch")

    # A delayed old process cannot overwrite the current launch's report.
    module.write_terminal_handoff(tmp_path, _handoff(), launch_generation="old-launch")
    assert module.read_terminal_handoff(tmp_path, expected_generation="new-launch") == {
        **current,
        "launch_generation": "new-launch",
    }
    module.remove_terminal_handoff(tmp_path, expected_generation="old-launch")
    assert not old.exists()
    assert new.exists()
    assert module.read_terminal_handoff(tmp_path, expected_generation="new-launch")["workspace_filesystem"] == {
        "capacity_bytes": 8192
    }
    module.remove_terminal_handoff(tmp_path, expected_generation="new-launch")
    assert not new.parent.exists()


@pytest.mark.parametrize("field_generation", [None, "old-launch"])
def test_bound_read_rejects_missing_or_mismatched_envelope_generation(tmp_path, field_generation):
    path = module.terminal_handoff_path(tmp_path, launch_generation="new-launch")
    path.parent.mkdir(parents=True)
    handoff = _handoff()
    if field_generation is not None:
        handoff["launch_generation"] = field_generation
    path.write_bytes(module.canonical_json_bytes(handoff))
    assert module.read_terminal_handoff(tmp_path, expected_generation="new-launch") is None


def test_bound_runtime_helpers_never_fall_back_to_unbound_diagnostic_file(tmp_path):
    path = module.write_terminal_handoff(tmp_path, _handoff())
    assert module.read_terminal_handoff(tmp_path, expected_generation="new-launch") is None
    assert module.read_terminal_handoff(tmp_path, expected_generation="") is None
    module.remove_terminal_handoff(tmp_path, expected_generation="new-launch")
    module.remove_terminal_handoff(tmp_path, expected_generation="")
    assert path.exists()


@pytest.mark.parametrize("generation", ["", "../outside", "a/b", ".", "x" * 129, 17, "é"])
def test_generation_tokens_cannot_redirect_or_expand_private_paths(tmp_path, generation):
    with pytest.raises(module.InvalidTerminalHandoff, match="launch generation"):
        module.write_terminal_handoff(tmp_path, _handoff(), launch_generation=generation)
    assert module.read_terminal_handoff(tmp_path, expected_generation=generation) is None
    module.remove_terminal_handoff(tmp_path, expected_generation=generation)
    assert not (tmp_path / "resource_stats").exists()


def test_launch_option_keeps_exact_token_before_generic_type_coercion():
    assert (
        module.launch_generation_from_options(["resource_launch_generation=old", "resource_launch_generation=00017"])
        == "00017"
    )
    assert module.launch_generation_from_options(["cj_launch_id=00017"], option_name="cj_launch_id") == "00017"
    assert module.launch_generation_from_options([]) == ""


@pytest.mark.parametrize(
    "metric,value",
    [
        ("cpu_consumed", {"cpu_seconds": True}),
        ("cpu_consumed", {"cpu_seconds": "2"}),
        ("cpu_consumed", {"cpu_seconds": -1}),
        ("retained_content", {"bytes": 1.0}),
        ("message_traffic", {"sent_bytes": 0, "received_bytes": False}),
        ("resource_capacity", {"cpu": {"units": 0}, "memory": {"bytes": 0}, "gpu": {"count": 0}}),
        ("resource_capacity", {"cpu": {"units": 2}, "memory": {"bytes": 1.0}, "gpu": {"count": 0}}),
        ("resource_capacity", {"cpu": {"units": 2}, "memory": {"bytes": 0}, "gpu": {"count": 0.5}}),
        ("resource_capacity", {"cpu": {"units": 2}, "memory": {"bytes": 0}, "gpu": {"count": True}}),
    ],
)
def test_reader_rejects_bad_numeric_fields_without_erasing_sibling_metrics(tmp_path, metric, value):
    payload = _complete_handoff()
    target = payload if metric == "resource_capacity" else payload["resource_utilization"]
    target[metric] = value
    module.write_terminal_handoff(tmp_path, payload)

    checked = module.read_terminal_handoff(tmp_path)

    target = checked if metric == "resource_capacity" else checked["resource_utilization"]
    assert target[metric] == {"status": "unavailable"}
    assert {"metric": metric, "code": "malformed_source"} in checked["issues"]
    assert checked["resource_time"] == _complete_handoff()["resource_time"]
    assert checked["workspace_filesystem"] == {"capacity_bytes": 1024}


def test_reader_preserves_zero_capacity_and_exact_integer_bytes(tmp_path):
    payload = _complete_handoff()
    payload["resource_capacity"]["memory"]["bytes"] = 0
    payload["resource_utilization"]["retained_content"]["bytes"] = 10**50
    module.write_terminal_handoff(tmp_path, payload)

    assert module.read_terminal_handoff(tmp_path) == payload


def test_reader_contains_nonfinite_json_number_without_erasing_other_metrics(tmp_path):
    payload = _complete_handoff()
    payload["resource_utilization"]["cpu_consumed"]["cpu_seconds"] = 1.0
    path = module.write_terminal_handoff(tmp_path, payload)
    path.write_text(json.dumps(payload).replace('"cpu_seconds": 1.0', '"cpu_seconds": 1e999'))

    checked = module.read_terminal_handoff(tmp_path)

    assert checked["resource_utilization"]["cpu_consumed"] == {"status": "unavailable"}
    assert checked["resource_capacity"] == payload["resource_capacity"]


@pytest.mark.parametrize("bad", ["unknown", "unsupported"])
def test_reader_contains_invalid_partial_cause_to_its_metric(tmp_path, bad):
    payload = _complete_handoff()
    payload["resource_utilization"]["cpu_consumed"]["status"] = "partial"
    payload["issues"] = [{"metric": "cpu_consumed", "code": bad}]
    module.write_terminal_handoff(tmp_path, payload)

    checked = module.read_terminal_handoff(tmp_path)

    assert checked["resource_utilization"]["cpu_consumed"] == {"status": "unavailable"}
    assert checked["resource_capacity"] == payload["resource_capacity"]
    assert checked["issues"] == [{"metric": "cpu_consumed", "code": "malformed_source"}]
