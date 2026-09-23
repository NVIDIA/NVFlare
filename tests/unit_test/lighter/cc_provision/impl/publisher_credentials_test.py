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

"""Offline publisher-credential installer tests; never modify the user's home."""

import os
import shlex
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from tests.unit_test.lighter.cc_provision.impl.deployment_guards_test import require_coco_bash

ROOT = Path(__file__).resolve().parents[5] / "examples/devops/coco"
PASSWORD = "fixture-password-not-for-production"


@pytest.fixture(scope="module")
def credential_bash():
    bash = require_coco_bash()
    # The deployment target is Ubuntu; do not confuse absent GNU host tools
    # on a developer's machine with a failure of the credential installer.
    for tool in ("realpath", "find", "install", "sort", "tr", "hostname", "mkdir", "chmod"):
        if not shutil.which(tool):
            pytest.skip(f"Publisher credential tests require {tool}")
    probe = subprocess.run(
        [bash, "-c", 'realpath -e -- . >/dev/null && find . -maxdepth 0 -printf "%f\\n" >/dev/null'],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if probe.returncode:
        pytest.skip(f"Publisher credential tests require GNU realpath/find: {probe.stderr.strip()}")
    return bash


@pytest.fixture
def credential_kit(tmp_path, credential_bash):
    kit = tmp_path / "admin"
    # Copy the real admin/shared sources so the test exercises configuration
    # loading and path derivation, not a replacement platform helper.
    shutil.copytree(ROOT / "admin", kit)
    shutil.copytree(ROOT / "shared", tmp_path / "shared")
    work_root = tmp_path / "custom state"
    hostname = subprocess.check_output(["hostname", "-f"], text=True).strip()
    config = (ROOT / "admin/platform.env.example").read_text() + (
        f"EXPECTED_HOSTNAME={shlex.quote(hostname)}\n"
        f"WORK_ROOT={shlex.quote(str(work_root))}\n"
        'REGISTRY_HOST="secure.nvflare.local"\n'
        'REGISTRY_PORT="5000"\n'
        'REGISTRY_USERNAME="coco-publisher"\n'
        'KBS_URL="https://secure.nvflare.local:8443"\n'
    )
    (kit / "platform.env").write_text(config)
    received = tmp_path / "received credential"
    received.mkdir()
    (received / "username").write_text("coco-publisher\n")
    (received / "password").write_text(PASSWORD)
    (kit / "public").mkdir(exist_ok=True)
    for certificate in ("trustee.crt", "registry-ca.crt"):
        (kit / "public" / certificate).write_text("fixture-not-a-real-certificate\n")
    context = tmp_path / "build"
    context.mkdir()
    (context / "Dockerfile").write_text("FROM scratch\n")
    (kit / "workload.env").write_text(
        "RELEASE_NAME=fixture-v1\n"
        f"BUILD_CONTEXT={shlex.quote(str(context))}\n"
        f"DOCKERFILE={shlex.quote(str(context / 'Dockerfile'))}\n"
        "REGISTRY_REPOSITORY=workloads/fixture\n"
        "APP_COMMAND_JSON='[\"/coco-app\"]'\n"
        "APP_UID=65532\nAPP_GID=65532\n"
    )
    return kit, work_root, received


def run_installer(credential_bash, credential_kit, *, source=None, dry_run=False, verify_release=False):
    kit, work_root, received = credential_kit
    # Guard every filesystem-mutating command used by this script/helper.
    # Never reassign HOME or touch real ~/.local/bin or default state; the
    # default-root case records requested install destinations without writing.
    prelude = r"""
mkdir() {
    local item
    for item in "$@"; do
        [[ $item == -* || $item == "$HOME/.local/bin" ]] && continue
        if [[ $DRY_RUN == 1 ]]; then continue; fi
        [[ $item == "$TEST_ROOT/"* ]] || { echo "write outside test root: $item" >&2; return 99; }
        command mkdir -p -- "$item"
    done
}
chmod() {
    [[ $DRY_RUN == 1 ]] && return 0
    local item
    for item in "${@:2}"; do
        [[ $item == "$TEST_ROOT/"* ]] || { echo "write outside test root: $item" >&2; return 99; }
    done
    command chmod "$@"
}
install() {
    local destination=${@: -1}
    printf 'INSTALL_DESTINATION=%s\n' "$destination"
    [[ $DRY_RUN == 1 ]] && return 0
    [[ $destination == "$TEST_ROOT/"* ]] || { echo "write outside test root: $destination" >&2; return 99; }
    command install "$@"
}
source "$INSTALLER" "$RECEIVED"
if [[ $VERIFY_RELEASE == 1 ]]; then
    source "$SCRIPT_DIR/lib/release.sh"
    need_file "$REGISTRY_USERNAME_PATH"
    need_file "$REGISTRY_PASSWORD_PATH"
    printf 'RELEASE_USERNAME_PATH=%s\nRELEASE_PASSWORD_PATH=%s\n' "$REGISTRY_USERNAME_PATH" "$REGISTRY_PASSWORD_PATH"
fi
"""
    env = dict(
        os.environ,
        TEST_ROOT=str(kit.parent),
        DRY_RUN=str(int(dry_run)),
        INSTALLER=str(kit / "05-install-publisher-credential.sh"),
        RECEIVED=str(source or received),
        VERIFY_RELEASE=str(int(verify_release)),
        OWNER_CONFIG=str(kit / "workload.env"),
        PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
    )
    return subprocess.run([credential_bash, "-c", prelude], env=env, capture_output=True, text=True, timeout=10)


def test_installs_credential_in_configured_work_root(credential_bash, credential_kit):
    _, work_root, _ = credential_kit
    result = run_installer(credential_bash, credential_kit, verify_release=True)
    assert result.returncode == 0, result.stdout + result.stderr
    destination = work_root / "secrets/registry"
    assert (destination / "username").read_text() == "coco-publisher\n"
    assert (destination / "password").read_text() == PASSWORD
    assert stat.S_IMODE(destination.stat().st_mode) == 0o700
    assert stat.S_IMODE((destination / "username").stat().st_mode) == 0o600
    assert stat.S_IMODE((destination / "password").stat().st_mode) == 0o600
    assert f"RELEASE_USERNAME_PATH={destination / 'username'}" in result.stdout
    assert f"RELEASE_PASSWORD_PATH={destination / 'password'}" in result.stdout
    assert PASSWORD not in result.stdout + result.stderr


def test_default_work_root_is_still_supported_without_writing_home(credential_bash, credential_kit):
    kit, _, _ = credential_kit
    config = kit / "platform.env"
    lines = config.read_text().splitlines()
    config.write_text(
        "\n".join(
            'WORK_ROOT="${HOME}/coco-workload-owner"' if line.startswith("WORK_ROOT=") else line for line in lines
        )
        + "\n"
    )
    default = Path.home() / "coco-workload-owner/secrets/registry"
    if any((default / name).exists() or (default / name).is_symlink() for name in ("username", "password")):
        pytest.skip("Existing real default publisher credentials must not be accessed by this dry-run test")
    result = run_installer(credential_bash, credential_kit, dry_run=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"INSTALL_DESTINATION={default / 'username'}" in result.stdout
    assert f"INSTALL_DESTINATION={default / 'password'}" in result.stdout
    assert PASSWORD not in result.stdout + result.stderr


@pytest.mark.parametrize("invalid", ["extra", "missing", "empty", "symlink", "directory", "username", "source_symlink"])
def test_invalid_credential_handoff_is_rejected(credential_bash, credential_kit, invalid):
    kit, work_root, received = credential_kit
    password = received / "password"
    source = received
    if invalid == "extra":
        (received / "extra").touch()
    elif invalid == "missing":
        password.unlink()
    elif invalid == "empty":
        password.write_text("")
    elif invalid == "symlink":
        password.unlink()
        password.symlink_to(received / "username")
    elif invalid == "directory":
        password.unlink()
        password.mkdir()
    elif invalid == "username":
        (received / "username").write_text("registry-admin\n")
    else:
        source = kit.parent / "source-link"
        source.symlink_to(received, target_is_directory=True)
    result = run_installer(credential_bash, credential_kit, source=source)
    assert result.returncode != 0
    assert "INSTALL_DESTINATION=" not in result.stdout
    assert not (work_root / "secrets/registry/password").exists()
    assert PASSWORD not in result.stdout + result.stderr


@pytest.mark.parametrize("name", ["username", "password"])
@pytest.mark.parametrize("existing", ["regular", "dangling_symlink"])
def test_existing_credentials_are_never_overwritten(credential_bash, credential_kit, name, existing):
    _, work_root, _ = credential_kit
    destination = work_root / "secrets/registry"
    destination.mkdir(parents=True)
    target = destination / name
    if existing == "regular":
        target.write_text("existing-value")
    else:
        target.symlink_to(destination / "must-not-create")
    result = run_installer(credential_bash, credential_kit)
    assert result.returncode != 0
    assert "refusing to overwrite" in result.stderr
    assert "INSTALL_DESTINATION=" not in result.stdout
    if existing == "regular":
        assert target.read_text() == "existing-value"
    else:
        assert target.is_symlink()
        assert not (destination / "must-not-create").exists()


@pytest.mark.parametrize("invalid", ["missing", "hostname", "root"])
def test_installer_requires_reviewed_platform_configuration(credential_bash, credential_kit, invalid):
    kit, _, _ = credential_kit
    config = kit / "platform.env"
    if invalid == "missing":
        config.unlink()
    elif invalid == "hostname":
        config.write_text(config.read_text() + "EXPECTED_HOSTNAME=wrong-machine\n")
    else:
        config.write_text(config.read_text() + "WORK_ROOT=/tmp\n")
    result = run_installer(credential_bash, credential_kit)
    assert result.returncode != 0
    assert "INSTALL_DESTINATION=" not in result.stdout
    assert PASSWORD not in result.stdout + result.stderr
