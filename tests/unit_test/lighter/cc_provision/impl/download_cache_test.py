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
import shutil
import subprocess
import tarfile
from pathlib import Path
from unittest.mock import Mock

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


@pytest.fixture(scope="module")
def signing_keys(tmp_path_factory):
    if not shutil.which("gpg") or not shutil.which("gpgconf"):
        pytest.skip("GnuPG is required for real signing-key regression tests")
    home = tmp_path_factory.mktemp("signing-keys")
    home.chmod(0o700)
    base = ["gpg", "--batch", "--no-options", "--homedir", str(home)]
    keys = []
    try:
        for name in ("approved", "unapproved"):
            try:
                result = subprocess.run(
                    base
                    + [
                        "--pinentry-mode",
                        "loopback",
                        "--passphrase",
                        "",
                        "--quick-generate-key",
                        name,
                        "ed25519",
                        "sign",
                        "1d",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env={**os.environ, "LC_ALL": "C"},
                )
            except subprocess.TimeoutExpired:
                pytest.skip("GnuPG fixture key generation timed out after 30 seconds")
            if result.returncode:
                diagnostic = result.stderr.strip() or f"exit status {result.returncode}"
                unavailable = (
                    "no agent running",
                    "can't connect to the agent",
                    "failed to start agent",
                    "invalid algorithm",
                    "unsupported algorithm",
                    "unknown elliptic curve",
                    "invalid elliptic curve",
                )
                message = diagnostic.lower()
                unsupported_option = "invalid option" in message and any(
                    option in message for option in ("--pinentry-mode", "--quick-generate-key")
                )
                if unsupported_option or any(reason in message for reason in unavailable):
                    pytest.skip(f"GnuPG cannot generate the required Ed25519 fixture key: {diagnostic}")
                pytest.fail(f"Unexpected GnuPG fixture key generation failure: {diagnostic}")
            listing = subprocess.check_output(base + ["--with-colons", "--list-keys", name], text=True)
            fingerprint = next(line.split(":")[9] for line in listing.splitlines() if line.startswith("fpr:"))
            public = subprocess.check_output(base + ["--armor", "--export", fingerprint])
            keys.append((fingerprint, public))
        yield keys
    finally:
        # Best-effort teardown must not replace a prerequisite skip with an agent error.
        try:
            subprocess.run(["gpgconf", "--homedir", str(home), "--kill", "gpg-agent"], capture_output=True, timeout=10)
        except subprocess.TimeoutExpired:
            pass


@pytest.mark.parametrize(
    "outcome,expected,reason",
    [
        ("gpg: No agent running", pytest.skip.Exception, "No agent running"),
        ('gpg: invalid option "--quick-generate-key"', pytest.skip.Exception, "invalid option"),
        ("gpg: Invalid algorithm", pytest.skip.Exception, "Invalid algorithm"),
        (subprocess.TimeoutExpired("gpg", 30), pytest.skip.Exception, "timed out after 30 seconds"),
        ("gpg: Invalid user ID", pytest.fail.Exception, "Unexpected GnuPG.*Invalid user ID"),
    ],
    ids=["agent", "option", "algorithm", "timeout", "unexpected-error"],
)
def test_signing_key_fixture_capability_failures(tmp_path_factory, monkeypatch, outcome, expected, reason):
    if isinstance(outcome, str):
        outcome = subprocess.CompletedProcess(["gpg"], 2, "", outcome)
    # Failed cleanup must preserve the diagnostic skip/failure from key generation.
    run = Mock(side_effect=[outcome, subprocess.CompletedProcess(["gpgconf"], 1)])
    monkeypatch.setattr(shutil, "which", lambda _: "/fixture/tool")
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(expected, match=reason):
        next(signing_keys.__wrapped__(tmp_path_factory))
    assert run.call_count == 2
    assert run.call_args_list[0].kwargs["timeout"] == 30
    assert run.call_args_list[0].kwargs["env"]["LC_ALL"] == "C"
    assert run.call_args.args[0][0] == "gpgconf"
    assert run.call_args.args[0][-2:] == ["--kill", "gpg-agent"]
    assert run.call_args.kwargs["timeout"] == 10


@pytest.mark.parametrize("export_fails", [False, True])
def test_signing_key_fixture_retains_success_and_export_errors(tmp_path_factory, monkeypatch, export_fails):
    fingerprint = "A" * 40
    listing = f"fpr:::::::::{fingerprint}:\n"
    outputs = (
        [listing, subprocess.CalledProcessError(2, ["gpg", "--export"])]
        if export_fails
        else [listing, b"approved", listing, b"unapproved"]
    )
    run = Mock(return_value=subprocess.CompletedProcess(["gpg"], 0, "", ""))
    monkeypatch.setattr(shutil, "which", lambda _: "/fixture/tool")
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "check_output", Mock(side_effect=outputs))
    keys = signing_keys.__wrapped__(tmp_path_factory)
    if export_fails:
        with pytest.raises(subprocess.CalledProcessError):
            next(keys)
    else:
        try:
            assert next(keys) == [(fingerprint, b"approved"), (fingerprint, b"unapproved")]
        finally:
            keys.close()
    assert run.call_args.args[0][0] == "gpgconf"
    assert run.call_args.args[0][-2:] == ["--kill", "gpg-agent"]


@pytest.mark.parametrize("kind", ["valid", "lowercase_pin", "wrong_key", "extra_key", "malformed", "wrong_pin"])
def test_apt_key_installs_only_approved_primary(tmp_path, signing_keys, kind):
    (pin, good), (_, bad) = signing_keys
    payload = {"wrong_key": bad, "extra_key": good + bad, "malformed": b"not a public key"}.get(kind, good)
    source = tmp_path / "key.asc"
    source.write_bytes(payload)
    destination = tmp_path / "installed.gpg"
    destination.write_bytes(b"previous keyring")
    if kind == "wrong_pin":
        pin = "0" * 40
    elif kind == "lowercase_pin":
        pin = pin.lower()
    result = run(
        tmp_path,
        f"""
prepare_download_dir
install -m 0600 {shlex.quote(str(source))} "$STATE_DIR/downloads/Release.key"
install_verified_apt_key "$STATE_DIR/downloads/Release.key" {pin} {shlex.quote(str(destination))}
""",
    )
    if kind in ("valid", "lowercase_pin"):
        assert result.returncode == 0, result.stderr
        expected = subprocess.check_output(
            ["gpg", "--batch", "--no-options", "--homedir", str(tmp_path), "--dearmor"], input=good
        )
        assert destination.read_bytes() == expected
        assert destination.stat().st_mode & 0o777 == 0o644
    else:
        assert result.returncode != 0
        assert destination.read_bytes() == b"previous keyring"


@pytest.mark.parametrize("pin", [None, "", "0" * 39, "g" * 40, "0" * 41])
def test_apt_missing_or_invalid_pin_fails_before_privileged_setup(tmp_path, pin):
    source = INSTALLER.read_text()
    preflight = source[source.index("load_config\n") : source.index('tmp_dir="')]
    setup = (
        "unset KUBERNETES_APT_KEY_FINGERPRINT\n"
        if pin is None
        else f"KUBERNETES_APT_KEY_FINGERPRINT={shlex.quote(pin)}\n"
    )
    result = run(
        tmp_path,
        setup
        + f"CNI_PLUGINS_SHA256={'0' * 64}\n"
        + "CONFIG_FILE=fixture.env\nload_config() { :; }\n"
        + "require_root_or_sudo() { echo PRIVILEGED; }\n"
        + preflight,
    )
    assert result.returncode != 0
    assert "Set KUBERNETES_APT_KEY_FINGERPRINT" in result.stderr
    assert "PRIVILEGED" not in result.stdout


@pytest.mark.parametrize("role", ["coco", "trusted_system/bootstrap"])
def test_apt_templates_pin_reviewed_primary(role):
    config = (COMMON.parents[3] / role / "config.env.example").read_text()
    assert "KUBERNETES_APT_KEY_FINGERPRINT=DE15B14486CD377B9E876E1A234654DA9A296436\n" in config


@pytest.mark.parametrize("valid", [True, False])
def test_bootstrap_rejects_wrong_key_before_repository_update(tmp_path, signing_keys, valid):
    (pin, approved), (_, wrong) = signing_keys
    payload = tmp_path / "key.asc"
    payload.write_bytes(approved if valid else wrong)
    keyring_dir = tmp_path / "keyrings"
    source = INSTALLER.read_text()
    start = source.index("as_root install -d -m 0755 /etc/apt/keyrings")
    end = source.index('echo "deb [signed-by=', start)
    section = source[start:end].replace("/etc/apt/keyrings", str(keyring_dir))
    result = run(
        tmp_path,
        f"""
prepare_download_dir
download_dir="$STATE_DIR/downloads"
KUBERNETES_MINOR=v1.34
KUBERNETES_APT_KEY_FINGERPRINT={pin}
install -m 0600 {shlex.quote(str(payload))} "$download_dir/kubernetes-v1.34-Release.key"
curl() {{ echo "Unexpected network request" >&2; return 99; }}
{section}
echo REPOSITORY_UPDATE_REACHED
""",
    )
    assert (result.returncode == 0) is valid, result.stderr
    assert ("REPOSITORY_UPDATE_REACHED" in result.stdout) is valid
    assert (keyring_dir / "kubernetes-apt-keyring.gpg").exists() is valid
