# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Exercise the committed plugin boundary without agents or optional SDK dependencies."""

import hashlib
import json
import stat
import zipfile

import pytest

from dev_tools.agent import package_plugin


@pytest.fixture
def bundle(tmp_path):
    package = tmp_path / "plugin"
    expected = {"plugin.json": (b"{}\n", 0o644), "skills/demo/run.sh": (b"#!/bin/sh\n", 0o755)}
    package_plugin.sync_bundle(package, expected)
    return package, expected


@pytest.mark.parametrize("mutation", ["content", "mode", "missing", "extra", "symlink", "directory-symlink"])
def test_bundle_rejects_drift(bundle, tmp_path, mutation):
    package, expected = bundle
    target = package / "skills/demo/run.sh"
    if mutation == "content":
        target.write_bytes(b"changed\n")
    elif mutation == "mode":
        target.chmod(0o644)
    elif mutation == "missing":
        target.unlink()
    elif mutation == "extra":
        (package / "unexpected.txt").write_text("not a source input")
    elif mutation == "symlink":
        target.unlink()
        target.symlink_to(tmp_path / "missing")
    else:
        (package / "linked-directory").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="drift|Symlinks"):
        package_plugin.check_bundle(package, expected)


def test_source_change_requires_bundle_refresh(bundle):
    package, expected = bundle
    updated = {**expected, "plugin.json": (b'{"version":"2.0.0"}\n', 0o644)}
    with pytest.raises(ValueError, match="drift"):
        package_plugin.check_bundle(package, updated)
    package_plugin.sync_bundle(package, updated)
    assert package_plugin.check_bundle(package, updated) == updated


@pytest.mark.parametrize("evidence", sorted(package_plugin.VERIFICATION_FILES))
def test_evidence_retained_but_blocks_changed_inputs(bundle, evidence):
    package, expected = bundle
    payload = b"retained output bytes\n"
    (package / evidence).write_bytes(payload)
    package_plugin.sync_bundle(package, expected)
    actual = package_plugin.check_bundle(package, expected)
    assert actual[evidence][0] == payload
    updated = {**expected, "plugin.json": (b'{"version":"2.0.0"}\n', 0o644)}
    with pytest.raises(ValueError, match="fresh evaluation and signing required"):
        package_plugin.sync_bundle(package, updated)
    assert (package / "plugin.json").read_bytes() == expected["plugin.json"][0]
    assert (package / evidence).read_bytes() == payload


def test_verification_exceptions_apply_only_at_plugin_root(bundle):
    package, expected = bundle
    (package / "skills/demo/eval-result.json").write_text("unexpected member evidence")
    with pytest.raises(ValueError, match="drift"):
        package_plugin.check_bundle(package, expected)


def test_sync_refuses_to_delete_unexpected_files(bundle):
    package, expected = bundle
    target = package / "notes.txt"
    target.write_text("operator notes")
    with pytest.raises(ValueError, match="Review and remove"):
        package_plugin.sync_bundle(package, expected)
    assert target.read_text() == "operator notes"


def test_bundle_parent_cannot_be_symlink(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "plugins").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlinks"):
        package_plugin.bundle_path(tmp_path)
    assert not list(outside.iterdir())


def test_archive_preserves_verification_outputs_and_records_their_hashes(bundle, tmp_path):
    package, expected = bundle
    signature = b"retained signature; no trust check performed by the packager\n"
    (package / "plugin.sig.oms").write_bytes(signature)
    (package / "plugin.sig.oms").chmod(0o644)
    files = package_plugin.check_bundle(package, expected)
    metadata = {"plugin": "demo", "version": "1.0.0", "plugin_signature_verified": False, "files": {}}
    output = tmp_path / "archive"
    archive = package_plugin.write_bundle(output, files, metadata)
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.read("demo/plugin.sig.oms") == signature
        assert stat.S_IMODE(zipped.getinfo("demo/plugin.sig.oms").external_attr >> 16) == 0o644
    assert (output / "demo/plugin.sig.oms").read_bytes() == signature
    recorded = json.loads((output / "build-metadata.json").read_text())
    assert recorded["files"]["plugin.sig.oms"] == hashlib.sha256(signature).hexdigest()
    assert set(recorded["files"]) == set(files)
    assert recorded["plugin_signature_present"] is True
    assert recorded["plugin_signature_verified"] is False
