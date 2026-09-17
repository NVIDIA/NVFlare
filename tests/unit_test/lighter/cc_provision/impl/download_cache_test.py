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
INSTALLER = COMMON.parents[1] / "10-install-kubernetes.sh"


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


@pytest.mark.parametrize("pin", [None, "", "0" * 63, "g" * 64, "0" * 65])
def test_cni_missing_or_invalid_pin_fails_before_privileged_setup(tmp_path, pin):
    source = INSTALLER.read_text()
    preflight = source[source.index("load_config\n") : source.index('tmp_dir="')]
    setup = "unset CNI_PLUGINS_SHA256\n" if pin is None else f"CNI_PLUGINS_SHA256={shlex.quote(pin)}\n"
    result = run(
        tmp_path,
        setup
        + "CONFIG_FILE=fixture.env\nload_config() { :; }\n"
        + "require_root_or_sudo() { echo PRIVILEGED; }\n"
        + preflight,
    )
    assert result.returncode != 0
    assert "Set CNI_PLUGINS_SHA256" in result.stderr
    assert "PRIVILEGED" not in result.stdout


@pytest.mark.parametrize("role", ["coco", "trusted_system/bootstrap"])
def test_cni_templates_pin_reviewed_release(role):
    config = (COMMON.parents[3] / role / "config.env.example").read_text()
    assert "CNI_PLUGINS_VERSION=v1.8.0\n" in config
    assert "CNI_PLUGINS_SHA256=ab3bda535f9d90766cccc90d3dddb5482003dd744d7f22bcf98186bf8eea8be6\n" in config


@pytest.mark.parametrize("failure", ["none", "download", "cached", "after_verification"])
def test_cni_install_uses_configured_pin_through_extraction(tmp_path, failure):
    approved, replacement = tmp_path / "approved.tgz", tmp_path / "replacement.tgz"
    sha = archive(approved, b"approved")
    archive(replacement, b"substituted")
    target = tmp_path / "cni-bin"
    payload = replacement if failure == "download" else approved
    source = INSTALLER.read_text()
    start = source.index('log "Installing CNI plugins')
    end = source.index("as_root install -d -m 0755 /etc/containerd", start)
    # Exercise the real install section and verification helpers with fixture
    # transport and a scratch extraction destination, never host CNI binaries.
    section = source[start:end].replace("/opt/cni/bin", shlex.quote(str(target)))
    cached = (
        f'install -m 0600 {shlex.quote(str(replacement))} "$STATE_DIR/downloads/cni-plugins-linux-amd64-v1.8.0.tgz"\n'
        if failure == "cached"
        else ""
    )
    substitute = (
        f'if [[ $1 == install ]]; then cp {shlex.quote(str(replacement))} "$cni_archive"; fi\n'
        if failure == "after_verification"
        else ""
    )
    result = run(
        tmp_path,
        f"""
prepare_download_dir
download_dir="$STATE_DIR/downloads"
CNI_PLUGINS_VERSION=v1.8.0
CNI_PLUGINS_SHA256={sha}
IGNORE_CHECKSUM_MISMATCH=1
curl() {{
  [[ $# == 7 && $5 == https://github.com/containernetworking/plugins/releases/download/v1.8.0/cni-plugins-linux-amd64-v1.8.0.tgz && $6 == -o ]] || return 1
  cp {shlex.quote(str(payload))} "$7"
}}
as_root() {{
  {substitute}
  "$@"
}}
{cached}
{section}
""",
    )
    if failure in ("download", "after_verification"):
        assert result.returncode != 0
        assert not (target / "marker").exists()
        if failure == "download":
            assert not target.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert (target / "marker").read_bytes() == b"approved"
