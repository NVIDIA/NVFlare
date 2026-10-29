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

"""Build-wrapper checks; these do not replace the native Rust or hardware tests."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

BUILD = Path(__file__).resolve().parents[5] / "examples/devops/coco/trusted_system/tdx-verifier/build.sh"
VERIFY_INTEL_KEY = BUILD.with_name("verify-intel-key.sh")
IMAGE_ID = "sha256:" + "a" * 64


@pytest.fixture
def build_env(tmp_path):
    tools = tmp_path / "bin"
    tools.mkdir()
    calls = tmp_path / "docker-calls.jsonl"
    docker = tools / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['DOCKER_TEST_CALLS'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1:3] == ['image', 'inspect']:\n"
        f"    print({IMAGE_ID!r})\n"
        "elif sys.argv[1] == 'run':\n"
        "    print('tdx-evidence-verify 0.1.0')\n"
    )
    docker.chmod(0o700)
    (tools / "python3").symlink_to(sys.executable)
    env = {**os.environ, "PATH": f"{tools}:{os.environ['PATH']}", "DOCKER_TEST_CALLS": str(calls)}
    env.pop("TDX_VERIFIER_BUILD_NETWORK", None)
    return env, calls


@pytest.mark.parametrize("network", [None, "default", "host"])
def test_build_network_is_explicit_and_preserves_image_pin(tmp_path, build_env, network):
    env, calls = build_env
    if network is not None:
        env["TDX_VERIFIER_BUILD_NETWORK"] = network
    output = tmp_path / "output"
    result = subprocess.run(["bash", str(BUILD), str(output)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    build = next(command for command in commands if command[0] == "build")
    assert build[1:5] == ["--network", network or "default", "--pull", "--platform"]
    assert (output / "tdx-evidence-verify.image-id").read_text().strip() == IMAGE_ID
    assert f"IMAGE_ID='{IMAGE_ID}'" in (output / "tdx-evidence-verify").read_text()


def test_unknown_build_network_is_rejected_before_build(tmp_path, build_env):
    env, calls = build_env
    env["TDX_VERIFIER_BUILD_NETWORK"] = "unreviewed-network"
    result = subprocess.run(["bash", str(BUILD), str(tmp_path / "output")], env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "must be default or host" in result.stderr
    assert not any(json.loads(line)[0] == "build" for line in calls.read_text().splitlines())


@pytest.mark.parametrize(("failure", "returncode"), [("", 0), ("fingerprint", 1), ("inspect", 1), ("dearmor", 23)])
def test_intel_key_verification_cleans_up_gnupg_home(tmp_path, failure, returncode):
    tools = tmp_path / "bin"
    tools.mkdir()
    temporary_homes = tmp_path / "gnupg"
    temporary_homes.mkdir()
    mktemp = tools / "mktemp"
    mktemp.write_text(
        f"#!{sys.executable}\n"
        "import os, tempfile\n"
        "print(tempfile.mkdtemp(prefix='coco-intel-key.', dir=os.environ['INTEL_KEY_TEST_ROOT']))\n"
    )
    mktemp.chmod(0o700)
    calls = tmp_path / "gpg-calls.jsonl"
    gpg = tools / "gpg"
    gpg.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "home = Path(sys.argv[sys.argv.index('--homedir') + 1])\n"
        "with open(os.environ['INTEL_KEY_TEST_CALLS'], 'a') as stream:\n"
        "    stream.write(json.dumps(str(home)) + '\\n')\n"
        "(home / 'private-keys-v1.d').mkdir(exist_ok=True)\n"
        "(home / 'private-keys-v1.d' / 'sentinel').write_text('temporary key data')\n"
        "failure = os.environ['INTEL_KEY_TEST_FAILURE']\n"
        "if '--show-keys' in sys.argv:\n"
        "    if failure == 'inspect':\n"
        "        sys.exit(17)\n"
        "    fingerprint = 'invalid' if failure == 'fingerprint' else '150434D1488BF80308B69398E5C7F0FA1C6C6C3C'\n"
        "    print('pub:-:3072:1:key:0:0::::')\n"
        "    print('fpr:::::::::' + fingerprint + ':')\n"
        "elif '--dearmor' in sys.argv:\n"
        "    if failure == 'dearmor':\n"
        "        sys.exit(23)\n"
        "    Path(sys.argv[sys.argv.index('--output') + 1]).write_bytes(b'test keyring')\n"
        "else:\n"
        "    sys.exit(99)\n"
    )
    gpg.chmod(0o700)
    env = {
        **os.environ,
        "PATH": f"{tools}:{os.environ['PATH']}",
        "INTEL_KEY_TEST_ROOT": str(temporary_homes),
        "INTEL_KEY_TEST_CALLS": str(calls),
        "INTEL_KEY_TEST_FAILURE": failure,
    }
    key_input = tmp_path / "intel.asc"
    key_input.write_text("test public key")
    key_output = tmp_path / "intel.gpg"
    result = subprocess.run(
        ["bash", str(VERIFY_INTEL_KEY), str(key_input), str(key_output)], env=env, capture_output=True, text=True
    )
    assert result.returncode == returncode, result.stderr
    homes = [Path(json.loads(line)) for line in calls.read_text().splitlines()]
    assert len(homes) == (1 if failure in {"fingerprint", "inspect"} else 2)
    assert len(set(homes)) == 1
    assert all(home.parent == temporary_homes and not home.exists() for home in homes)
    assert not list(temporary_homes.iterdir())
    if returncode == 0:
        assert key_output.read_bytes() == b"test keyring"
        assert key_output.stat().st_mode & 0o777 == 0o644
    else:
        assert not key_output.exists()
