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

"""Real download helpers with file:// fixtures; no network, sudo, or root writes."""

import hashlib
import io
import os
import shlex
import subprocess
import tarfile
from pathlib import Path

import pytest

COMMON = Path(__file__).resolve().parents[5] / "examples/devops/coco/shared/bootstrap/lib/common.sh"


def run(tmp_path, script):
    prefix = (
        "set -Eeuo pipefail\n"
        f"export COCO_STATE_DIR={shlex.quote(str(tmp_path / 'state'))}\n"
        f"source {shlex.quote(str(COMMON))}\n"
        'as_root() { "$@"; }\n'
    )
    return subprocess.run(["bash", "-c", prefix + script], capture_output=True, text=True)


def test_private_cache_and_unpredictable_download(tmp_path):
    source = tmp_path / "input"
    source.write_bytes(b"reviewed bytes")
    victim = tmp_path / "victim"
    victim.write_text("untouched")
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    result = run(
        tmp_path,
        f"""
prepare_download_dir
out="$STATE_DIR/downloads/payload"
ln -s {shlex.quote(str(victim))} "$out.partial.$$"
ensure_download_verified {shlex.quote(source.as_uri())} {sha} "$out"
ensure_download_verified unused {sha} "$out"
""",
    )
    assert result.returncode == 0, result.stderr
    assert victim.read_text() == "untouched"
    assert (tmp_path / "state/downloads").stat().st_mode & 0o777 == 0o700
    output = tmp_path / "state/downloads/payload"
    assert output.read_bytes() == source.read_bytes() and not output.is_symlink()
    assert output.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("kind", ["shared_cache", "shared_parent", "symlink_cache", "symlink_file", "hardlink_file"])
def test_unsafe_cache_is_rejected_unchanged(tmp_path, kind):
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    cache = state / "downloads"
    cache.mkdir(mode=0o700)
    victim = tmp_path / "victim"
    victim.write_text("untouched")
    if kind == "shared_cache":
        cache.chmod(0o775)
    elif kind == "shared_parent":
        state.chmod(0o775)
    elif kind == "symlink_cache":
        cache.rmdir()
        cache.symlink_to(tmp_path, target_is_directory=True)
    elif kind == "symlink_file":
        (cache / "payload").symlink_to(victim)
    else:
        os.link(victim, cache / "payload")
    result = run(tmp_path, 'ensure_download unused "$STATE_DIR/downloads/payload"')
    assert result.returncode != 0
    assert victim.read_text() == "untouched"
    if kind == "shared_cache":
        assert cache.stat().st_mode & 0o777 == 0o775


def archive(path, content):
    with tarfile.open(path, "w:gz") as output:
        member = tarfile.TarInfo("marker")
        member.size = len(content)
        output.addfile(member, io.BytesIO(content))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("replace_after_check", [False, True])
def test_privileged_consumer_rechecks_private_snapshot(tmp_path, replace_after_check):
    source, replacement = tmp_path / "good.tgz", tmp_path / "replacement.tgz"
    sha = archive(source, b"approved")
    archive(replacement, b"substituted")
    target = tmp_path / "extract"
    target.mkdir()
    substitute = f'cp {shlex.quote(str(replacement))} "$out"\n' if replace_after_check else ""
    result = run(
        tmp_path,
        f"""
out="$STATE_DIR/downloads/archive.tgz"
ensure_download_verified {shlex.quote(source.as_uri())} {sha} "$out"
{substitute}
extract_verified_archive "$out" {sha} {shlex.quote(str(target))}
""",
    )
    assert (result.returncode == 0) is (not replace_after_check), result.stderr
    if replace_after_check:
        assert not list(target.iterdir())
    else:
        assert (target / "marker").read_bytes() == b"approved"


def test_checksum_bypass_cannot_enable_bad_download(tmp_path):
    source = tmp_path / "input"
    source.write_bytes(b"wrong")
    result = run(
        tmp_path,
        f"""
IGNORE_CHECKSUM_MISMATCH=1
ensure_download_verified {shlex.quote(source.as_uri())} {'0' * 64} "$STATE_DIR/downloads/payload"
""",
    )
    assert result.returncode != 0
    assert not (tmp_path / "state/downloads/payload").exists()
    assert not list((tmp_path / "state/downloads").glob(".download.*"))
