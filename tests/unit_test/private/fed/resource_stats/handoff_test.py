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

import os
from copy import deepcopy

import pytest

from nvflare.private.fed.resource_stats import handoff as module


def _handoff():
    return {
        "duration_seconds": "10",
        "internal_version": module.INTERNAL_HANDOFF_VERSION,
        "kind": module.INTERNAL_HANDOFF_KIND,
        "resource_capacity": {"status": "unavailable", "issues": ["observation_incomplete"]},
        "resource_time": {"status": "unavailable", "issues": ["observation_incomplete"]},
        "workspace_filesystem": {"capacity_bytes": "1024"},
        "resource_utilization": {
            "cpu_consumed": {"status": "unavailable", "issues": ["observation_incomplete"]},
            "retained_content": {"status": "unavailable", "issues": ["unsupported"]},
            "message_traffic": {"status": "unavailable", "issues": ["observation_incomplete"]},
        },
    }


def _complete_handoff():
    return {
        "duration_seconds": "10",
        "internal_version": module.INTERNAL_HANDOFF_VERSION,
        "kind": module.INTERNAL_HANDOFF_KIND,
        "resource_capacity": {
            "cpu": {"groups": [{"units": "2"}]},
            "memory": {"bytes": "4096"},
            "gpu": {"groups": []},
        },
        "resource_time": {
            "cpu": {"groups": [{"unit_seconds": "20"}]},
            "memory": {"byte_seconds": "40960"},
            "gpu": {"groups": []},
        },
        "workspace_filesystem": {"capacity_bytes": "1024"},
        "resource_utilization": {
            "cpu_consumed": {"cpu_seconds": "2"},
            "retained_content": {"bytes": "128"},
            "message_traffic": {"sent_bytes": "512", "received_bytes": "128"},
        },
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
    with pytest.raises(module.InvalidTerminalHandoff, match="unsupported version"):
        module.write_terminal_handoff(tmp_path, handoff)
    assert not module.terminal_handoff_path(tmp_path).exists()

    handoff["resource_time"] = handoff.pop("resource_capacity")
    with pytest.raises(module.InvalidTerminalHandoff, match="fields"):
        module.write_terminal_handoff(tmp_path, handoff)
    assert not module.terminal_handoff_path(tmp_path).exists()


@pytest.mark.parametrize("fallback", [False, True])
def test_atomic_writer_round_trip_and_replacement_without_leftover_temp_files(tmp_path, monkeypatch, fallback):
    if fallback:
        monkeypatch.setattr(module, "_SECURE_HANDOFF_WRITE_DIR_FD", False)
    run = tmp_path / "run"
    path = module.write_terminal_handoff(run, _handoff())
    assert module.read_terminal_handoff(run) == _handoff()
    updated = _handoff()
    updated["workspace_filesystem"] = {"capacity_bytes": "2048"}
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
    assert actual[measurement] == {"status": "unavailable", "issues": ["malformed_source"]}
    for name, value in original.items():
        if name not in {measurement, "resource_utilization"}:
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


def test_reader_normalizes_numeric_values_at_the_parent_boundary(tmp_path):
    handoff = _complete_handoff()
    path = module.write_terminal_handoff(tmp_path, handoff)
    handoff["duration_seconds"] = "10.000"
    handoff["resource_capacity"]["cpu"]["groups"][0]["units"] = "2.0"
    handoff["resource_utilization"]["cpu_consumed"]["cpu_seconds"] = "2.000000000"
    handoff["resource_utilization"]["message_traffic"]["sent_bytes"] = "0512.00"
    handoff["resource_utilization"]["message_traffic"]["received_bytes"] = "0128.0"
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
