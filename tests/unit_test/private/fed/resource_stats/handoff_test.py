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
from unittest.mock import Mock

import pytest

from nvflare.private.fed.resource_stats import handoff as module


def _link(path, target, *, directory):
    try:
        path.symlink_to(target, target_is_directory=directory)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are unavailable on this platform")


def test_atomic_writer_round_trip_and_replacement_without_leftover_temp_files(make_handoff, tmp_path, monkeypatch):
    run = tmp_path / "run"
    original = make_handoff()
    original["duration_seconds"] = 1
    path = module.write_terminal_handoff(run, original)
    assert module.read_terminal_handoff(run) == original
    updated = make_handoff()
    updated["duration_seconds"] = 2
    module.write_terminal_handoff(run, updated)
    assert module.read_terminal_handoff(run) == updated
    assert list(path.parent.iterdir()) == [path]


def test_reader_contains_plain_decoder_value_error(make_handoff, tmp_path, monkeypatch):
    path = module.write_terminal_handoff(tmp_path, make_handoff(complete=True))
    payload = path.read_bytes()
    # Python's integer decoder can raise ValueError independently of JSON syntax.
    # Control that boundary without depending on a process-wide digit limit.
    decoder = Mock(side_effect=ValueError("integer string conversion limit exceeded"))
    monkeypatch.setattr(module.json, "loads", decoder)

    assert module.read_terminal_handoff(tmp_path) is None
    decoder.assert_called_once()
    assert decoder.call_args.args[0] == payload


def test_reader_preserves_native_numeric_values_at_the_parent_boundary(make_handoff, tmp_path):
    handoff = make_handoff(complete=True)
    path = module.write_terminal_handoff(tmp_path, handoff)
    handoff["duration_seconds"] = 10.0
    handoff["resource_capacity"]["cpu"]["units"] = 2.0
    handoff["resource_utilization"]["cpu_consumed"]["cpu_seconds"] = 2.0
    path.write_bytes(module.canonical_json_bytes(handoff))
    assert module.read_terminal_handoff(tmp_path) == make_handoff(complete=True)


@pytest.mark.parametrize("component", ["run", "resource_stats", "staging"])
def test_writer_rejects_preexisting_directory_symlink_without_outside_writes(
    make_handoff, tmp_path, monkeypatch, component
):
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
        module.write_terminal_handoff(run, make_handoff())
    assert sentinel.read_bytes() == b"outside sentinel"
    assert list(outside.iterdir()) == [sentinel]


def test_final_file_symlink_is_replaced_not_followed(make_handoff, tmp_path, monkeypatch):
    run = tmp_path / "run"
    path = module.terminal_handoff_path(run)
    path.parent.mkdir(parents=True)
    sentinel = tmp_path / "outside.json"
    sentinel.write_bytes(b"outside sentinel")
    _link(path, sentinel, directory=False)
    module.write_terminal_handoff(run, make_handoff())
    assert sentinel.read_bytes() == b"outside sentinel"
    assert not path.is_symlink()
    assert module.read_terminal_handoff(run) == make_handoff()


@pytest.mark.skipif(not module._SECURE_HANDOFF_WRITE_DIR_FD, reason="requires directory-relative safe writes")
def test_open_directory_cannot_be_redirected_to_outside_after_check(make_handoff, tmp_path, monkeypatch):
    run = tmp_path / "run"
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / module.TERMINAL_HANDOFF_FILE
    sentinel.write_bytes(b"outside sentinel")
    open_directory = module._open_handoff_directory

    def replaced_directory(*args, **kwargs):
        descriptor = open_directory(*args, **kwargs)
        path = module.terminal_handoff_path(run).parent
        path.rename(path.with_name("original-staging"))
        _link(path, outside, directory=True)
        return descriptor

    monkeypatch.setattr(module, "_open_handoff_directory", replaced_directory)
    module.write_terminal_handoff(run, make_handoff())
    assert sentinel.read_bytes() == b"outside sentinel"
    assert list(outside.iterdir()) == [sentinel]
    original = run / module.RESOURCE_STATS_DIR / "original-staging"
    assert [path.name for path in original.iterdir()] == [module.TERMINAL_HANDOFF_FILE]


def test_replace_failure_preserves_previous_file_and_cleans_temporary_file(make_handoff, tmp_path, monkeypatch):
    run = tmp_path / "run"
    path = module.write_terminal_handoff(run, make_handoff())
    old = path.read_bytes()

    def fail_replace(*args, **kwargs):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        module.write_terminal_handoff(run, make_handoff())
    assert path.read_bytes() == old
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.skipif(not module._SECURE_HANDOFF_WRITE_DIR_FD, reason="requires directory-relative safe writes")
def test_temporary_name_collisions_are_bounded_and_do_not_overwrite_existing_file(make_handoff, tmp_path, monkeypatch):
    run = tmp_path / "run"
    path = module.write_terminal_handoff(run, make_handoff())
    collision = path.parent / ".terminal-existing.tmp"
    collision.write_bytes(b"keep existing file")
    tokens = []

    def fixed_token(_length):
        tokens.append("existing")
        return "existing"

    monkeypatch.setattr(module.secrets, "token_hex", fixed_token)
    with pytest.raises(OSError, match="could not create"):
        module.write_terminal_handoff(run, make_handoff())
    assert len(tokens) == 8
    assert collision.read_bytes() == b"keep existing file"
    assert module.read_terminal_handoff(run) == make_handoff()


@pytest.mark.skipif(not module._SECURE_HANDOFF_WRITE_DIR_FD, reason="requires directory-relative safe writes")
def test_fdopen_failure_closes_descriptors_and_cleans_temporary_file(make_handoff, tmp_path, monkeypatch):
    run = tmp_path / "run"
    path = module.write_terminal_handoff(run, make_handoff())
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
        module.write_terminal_handoff(run, make_handoff())
    assert descriptors
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert list(path.parent.iterdir()) == [path]


def test_generation_paths_preserve_replacement_against_late_writer_and_old_cleanup(make_handoff, tmp_path, monkeypatch):
    old = module.write_terminal_handoff(tmp_path, make_handoff(), launch_generation="old-launch")
    current = make_handoff()
    current["duration_seconds"] = 3
    new = module.write_terminal_handoff(tmp_path, current, launch_generation="new-launch")

    # A delayed old process cannot overwrite the current launch's report.
    module.write_terminal_handoff(tmp_path, make_handoff(), launch_generation="old-launch")
    assert module.read_terminal_handoff(tmp_path, expected_generation="new-launch") == {
        **current,
        "launch_generation": "new-launch",
    }
    module.remove_terminal_handoff(tmp_path, expected_generation="old-launch")
    assert not old.exists()
    assert new.exists()
    assert module.read_terminal_handoff(tmp_path, expected_generation="new-launch")["duration_seconds"] == 3
    module.remove_terminal_handoff(tmp_path, expected_generation="new-launch")
    assert not new.parent.exists()


@pytest.mark.parametrize("field_generation", [None, "old-launch"])
def test_bound_read_rejects_missing_or_mismatched_envelope_generation(make_handoff, tmp_path, field_generation):
    path = module.terminal_handoff_path(tmp_path, launch_generation="new-launch")
    path.parent.mkdir(parents=True)
    handoff = make_handoff()
    if field_generation is not None:
        handoff["launch_generation"] = field_generation
    path.write_bytes(module.canonical_json_bytes(handoff))
    assert module.read_terminal_handoff(tmp_path, expected_generation="new-launch") is None


def test_bound_runtime_helpers_never_fall_back_to_unbound_diagnostic_file(make_handoff, tmp_path):
    path = module.write_terminal_handoff(tmp_path, make_handoff())
    assert module.read_terminal_handoff(tmp_path, expected_generation="new-launch") is None
    assert module.read_terminal_handoff(tmp_path, expected_generation="") is None
    module.remove_terminal_handoff(tmp_path, expected_generation="new-launch")
    module.remove_terminal_handoff(tmp_path, expected_generation="")
    assert path.exists()


@pytest.mark.parametrize("generation", ["", "../outside", "a/b", ".", "x" * 129, 17, "é"])
def test_generation_tokens_cannot_redirect_or_expand_private_paths(make_handoff, tmp_path, generation):
    with pytest.raises(module.InvalidTerminalHandoff, match="launch generation"):
        module.write_terminal_handoff(tmp_path, make_handoff(), launch_generation=generation)
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


def test_reader_preserves_zero_capacity_and_exact_integer_bytes(make_handoff, tmp_path):
    payload = make_handoff(complete=True)
    payload["resource_capacity"]["memory"]["bytes"] = 0
    payload["resource_utilization"]["message_traffic"]["sent_bytes"] = 10**50
    module.write_terminal_handoff(tmp_path, payload)

    assert module.read_terminal_handoff(tmp_path) == payload


@pytest.mark.parametrize("capability", ["_SECURE_HANDOFF_DIR_FD", "_SECURE_HANDOFF_WRITE_DIR_FD"])
def test_missing_filesystem_primitives_never_use_path_based_writes(make_handoff, tmp_path, monkeypatch, capability):
    path = module.write_terminal_handoff(tmp_path, make_handoff())
    original = path.read_bytes()
    monkeypatch.setattr(module, capability, False)
    with pytest.raises(OSError, match="directory-relative"):
        module.write_terminal_handoff(tmp_path, make_handoff())
    assert path.read_bytes() == original
    if capability == "_SECURE_HANDOFF_DIR_FD":
        assert module.read_terminal_handoff(tmp_path) is None
        module.remove_terminal_handoff(tmp_path)
        assert path.read_bytes() == original
