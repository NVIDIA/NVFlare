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

"""Run configuration-only reference migration without host or network operations."""

import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

COCO = Path(__file__).resolve().parents[5] / "examples/devops/coco"
STAGE = "02-install-platform-reference-values.sh"
FIELDS = ("mr_td", "rtmr_0", "rtmr_1", "rtmr_2", "rtmr_3", "xfam", "tdvfkernel", "tdvfkernelparams")
STUB_COMMON = """
source "$SCRIPT_DIR/platform.env"
printf 'source common\\n' >>"$TEST_CALLS"
die() { printf '%s\\n' "$*" >&2; exit 1; }
lock_platform_reference_update() { printf 'lock references\\n' >>"$TEST_CALLS"; }
require_sudo() { die 'Unexpected privileged operation'; }
install_platform_references() { die 'Unexpected network operation'; }
"""


def reference_values(*missing):
    profile = {"id": "approved-tdx"}
    profile.update({key: "a" * (16 if key == "xfam" else 96) for key in FIELDS if key not in missing})
    return {"schema": "coco-platform-reference-values/v2", "tee": "tdx", "profiles": [profile]}


@pytest.fixture
def role_kit(tmp_path):
    role = tmp_path / "service"
    library = role / "lib"
    library.mkdir(parents=True)
    shutil.copyfile(COCO / "service" / STAGE, role / STAGE)
    shutil.copyfile(COCO / "service/lib/platform-reference-values.py", library / "platform-reference-values.py")
    shutil.copyfile(COCO / "shared/platform-reference-values.py", library / "platform-reference-schema.py")
    (library / "common.sh").write_text(STUB_COMMON)
    snapshot = role / "approved-platform-reference-values.json"
    snapshot.write_text(json.dumps(reference_values("rtmr_0", "rtmr_3"), indent=4) + "\n")
    snapshot.chmod(0o600)
    (role / "platform.env").write_text(
        f'SNP_LAUNCH_MEASUREMENT="unchanged"\nPLATFORM_REFERENCE_VALUES_FILE="{snapshot}"\n'
    )
    (role / "platform.env").chmod(0o600)
    tools = tmp_path / "bin"
    tools.mkdir()
    (tools / "python3").symlink_to(sys.executable)
    calls = tmp_path / "calls"
    env = {
        **os.environ,
        "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TEST_CALLS": str(calls),
    }
    return role, env, calls


def configure(role, env, values):
    return subprocess.run(
        ["bash", str(role / STAGE), str(values), "--approve-platform-reference-values", "--configure-only"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_configure_only_archives_six_field_snapshot_before_migration(tmp_path, role_kit):
    role, env, calls = role_kit
    snapshot = role / "approved-platform-reference-values.json"
    old_snapshot = snapshot.read_bytes()
    old_env = (role / "platform.env").read_bytes()
    values = tmp_path / "reviewed-values.json"
    values.write_text(json.dumps(reference_values()))

    result = configure(role, env, values)

    assert result.returncode == 0, result.stderr
    assert "Configuration-only" in result.stdout
    assert calls.read_text().splitlines() == ["source common", "lock references"]
    backups = list(role.glob("platform.env.before-values.*.references.json"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == old_snapshot
    assert stat.S_IMODE(backups[0].stat().st_mode) == 0o600
    env_backup = backups[0].with_name(backups[0].name.removesuffix(".references.json"))
    assert env_backup.read_bytes() == old_env
    assert stat.S_IMODE(env_backup.stat().st_mode) == 0o600
    assert json.loads(snapshot.read_text()) == reference_values()
    assert stat.S_IMODE(snapshot.stat().st_mode) == 0o600
    assert (role / "platform.env").read_text() == (
        f'SNP_LAUNCH_MEASUREMENT="unchanged"\nPLATFORM_REFERENCE_VALUES_FILE={shlex.quote(str(snapshot))}\n'
    )
    assert stat.S_IMODE((role / "platform.env").stat().st_mode) == 0o600


@pytest.mark.parametrize("missing", [("rtmr_0",), ("rtmr_3",), ("rtmr_0", "rtmr_3")])
def test_configure_only_rejects_incomplete_new_values_before_mutation(tmp_path, role_kit, missing):
    role, env, calls = role_kit
    before = {
        path.relative_to(role): (path.read_bytes(), stat.S_IMODE(path.stat().st_mode))
        for path in role.rglob("*")
        if path.is_file()
    }
    values = tmp_path / "incomplete-values.json"
    values.write_text(json.dumps(reference_values(*missing)))

    result = configure(role, env, values)

    assert result.returncode != 0
    assert "all eight measurement fields" in result.stderr
    assert not calls.exists()
    after = {
        path.relative_to(role): (path.read_bytes(), stat.S_IMODE(path.stat().st_mode))
        for path in role.rglob("*")
        if path.is_file()
    }
    assert after == before
