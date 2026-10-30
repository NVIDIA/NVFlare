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
        "from pathlib import Path\n"
        "engine = Path(sys.argv[0]).name\n"
        "with open(os.environ['DOCKER_TEST_CALLS'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "with open(os.environ['DOCKER_TEST_CALLS'] + '.engines', 'a') as stream:\n"
        "    stream.write(engine + '\\n')\n"
        "if sys.argv[1:3] == ['image', 'inspect']:\n"
        f"    print(os.environ.get('DOCKER_TEST_IMAGE_ID', {IMAGE_ID!r}))\n"
        "elif sys.argv[1] == 'info':\n"
        "    if os.environ.get('DOCKER_TEST_INFO_FAILURE') == engine:\n"
        "        sys.exit(1)\n"
        "    if engine == 'podman':\n"
        "        print(os.environ.get('PODMAN_TEST_ROOTLESS', 'true'))\n"
        "    elif os.environ.get('DOCKER_TEST_NEEDS_SUDO') and not os.environ.get('DOCKER_TEST_SUDO'):\n"
        "        sys.exit(1)\n"
        "elif sys.argv[1] == 'run':\n"
        "    for argument in sys.argv:\n"
        "        if argument.startswith('type=bind,source='):\n"
        "            source = Path(argument.split('source=', 1)[1].split(',', 1)[0])\n"
        "            assert source.read_bytes()\n"
        "    print('tdx-evidence-verify 0.1.0')\n"
    )
    docker.chmod(0o700)
    (tools / "podman").symlink_to(docker)
    sudo = tools / "sudo"
    sudo.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "assert sys.argv[1:3] == ['-n', 'docker']\n"
        "os.environ['DOCKER_TEST_SUDO'] = '1'\n"
        "os.execvp('docker', sys.argv[2:])\n"
    )
    sudo.chmod(0o700)
    (tools / "python3").symlink_to(sys.executable)
    env = {**os.environ, "PATH": f"{tools}:{os.environ['PATH']}", "DOCKER_TEST_CALLS": str(calls)}
    env.pop("TDX_VERIFIER_BUILD_NETWORK", None)
    env.pop("TDX_VERIFIER_ENGINE", None)
    return env, calls


@pytest.mark.parametrize("engine", ["docker", "podman"])
@pytest.mark.parametrize("network", [None, "default", "host"])
def test_build_network_is_explicit_and_preserves_image_pin(tmp_path, build_env, network, engine):
    env, calls = build_env
    env["TDX_VERIFIER_ENGINE"] = engine
    if network is not None:
        env["TDX_VERIFIER_BUILD_NETWORK"] = network
    output = tmp_path / "output"
    result = subprocess.run(["bash", str(BUILD), str(output)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    build = next(command for command in commands if command[0] == "build")
    expected_network = "host" if network == "host" else ("private" if engine == "podman" else "default")
    expected_pull = "--pull=always" if engine == "podman" else "--pull"
    assert build[1:5] == ["--network", expected_network, expected_pull, "--platform"]
    assert (output / "tdx-evidence-verify.image-id").read_text().strip() == IMAGE_ID
    assert f"IMAGE_ID='{IMAGE_ID}'" in (output / "tdx-evidence-verify").read_text()


def test_unknown_build_network_is_rejected_before_build(tmp_path, build_env):
    env, calls = build_env
    env["TDX_VERIFIER_BUILD_NETWORK"] = "unreviewed-network"
    result = subprocess.run(["bash", str(BUILD), str(tmp_path / "output")], env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "must be default or host" in result.stderr
    assert not calls.exists()


@pytest.mark.parametrize("engine", ["", "unknown"])
def test_unknown_engine_is_rejected_before_side_effects(tmp_path, build_env, engine):
    env, calls = build_env
    env["TDX_VERIFIER_ENGINE"] = engine
    output = tmp_path / "output"
    result = subprocess.run(["bash", str(BUILD), str(output)], env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "must be docker or podman" in result.stderr
    assert not output.exists()
    assert not calls.exists()


@pytest.mark.parametrize("image_id", [IMAGE_ID, IMAGE_ID.removeprefix("sha256:")])
def test_podman_normalizes_config_id_and_pins_engine_for_private_evidence(tmp_path, build_env, image_id):
    env, calls = build_env
    env.update(TDX_VERIFIER_ENGINE="podman", DOCKER_TEST_IMAGE_ID=image_id)
    output = tmp_path / "output"
    result = subprocess.run(["bash", str(BUILD), str(output)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    build = next(command for command in commands if command[0] == "build")
    assert "--pull=always" in build
    assert "--pull" not in build
    assert (output / "tdx-evidence-verify.image-id").read_text().strip() == IMAGE_ID
    wrapper = output / "tdx-evidence-verify"
    assert "ENGINE_NAME='podman'" in wrapper.read_text()
    assert f"BUILD_UID='{os.getuid()}'" in wrapper.read_text()
    inputs = []
    for filename in ("evidence.json", "challenge.bin", "initdata.toml"):
        path = tmp_path / filename
        path.write_bytes(b"private evidence")
        path.chmod(0o600)
        inputs.append(str(path))
    # The selected engine belongs to the installed wrapper, not its invocation environment.
    env["TDX_VERIFIER_ENGINE"] = "docker"
    result = subprocess.run(["bash", str(wrapper), *inputs], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    run = commands[-1]
    assert run[0] == "run"
    assert "--userns=keep-id" in run
    assert run[run.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert "--read-only" in run and "--cap-drop" in run and "no-new-privileges" in run
    assert IMAGE_ID in run
    assert "--network" not in run  # Quote verification can retrieve PCS collateral.
    assert sum(argument.startswith("type=bind,source=") and argument.endswith(",readonly") for argument in run) == 3
    assert set(Path(str(calls) + ".engines").read_text().splitlines()) == {"podman"}
    assert all(Path(path).stat().st_mode & 0o777 == 0o600 for path in inputs)


@pytest.mark.parametrize("image_id", ["bad", "b" * 63, "sha256:" + "g" * 64])
def test_podman_rejects_invalid_image_id(tmp_path, build_env, image_id):
    env, _ = build_env
    env.update(TDX_VERIFIER_ENGINE="podman", DOCKER_TEST_IMAGE_ID=image_id)
    output = tmp_path / "output"
    result = subprocess.run(["bash", str(BUILD), str(output)], env=env, capture_output=True, text=True)
    assert result.returncode == 1
    assert "invalid verifier image ID" in result.stderr
    assert not (output / "tdx-evidence-verify").exists()


@pytest.mark.parametrize("setting", ["DOCKER_TEST_INFO_FAILURE", "PODMAN_TEST_ROOTLESS"])
def test_podman_never_falls_back_to_docker_or_sudo(tmp_path, build_env, setting):
    env, calls = build_env
    env.update(TDX_VERIFIER_ENGINE="podman")
    env[setting] = "podman" if setting == "DOCKER_TEST_INFO_FAILURE" else "false"
    result = subprocess.run(["bash", str(BUILD), str(tmp_path / "output")], env=env, capture_output=True, text=True)
    assert result.returncode == 1
    assert "rootless Podman" in result.stderr
    assert set(Path(str(calls) + ".engines").read_text().splitlines()) == {"podman"}
    assert not any(json.loads(line)[0] == "build" for line in calls.read_text().splitlines())


def test_podman_wrapper_rejects_different_user(tmp_path, build_env):
    env, calls = build_env
    env["TDX_VERIFIER_ENGINE"] = "podman"
    output = tmp_path / "output"
    result = subprocess.run(["bash", str(BUILD), str(output)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    wrapper = output / "tdx-evidence-verify"
    wrapper.write_text(wrapper.read_text().replace(f"BUILD_UID='{os.getuid()}'", f"BUILD_UID='{os.getuid() + 1}'"))
    calls.unlink()
    result = subprocess.run(["bash", str(wrapper), "--version"], env=env, capture_output=True, text=True)
    assert result.returncode == 1
    assert "same user" in result.stderr
    assert not calls.exists()


def test_docker_preserves_noninteractive_sudo_fallback(tmp_path, build_env):
    env, calls = build_env
    env["DOCKER_TEST_NEEDS_SUDO"] = "1"
    output = tmp_path / "output"
    result = subprocess.run(["bash", str(BUILD), str(output)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run(
        ["bash", str(output / "tdx-evidence-verify"), "--version"], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    assert "--userns=keep-id" not in commands[-1]
    assert commands[-1][-2:] == [IMAGE_ID, "--version"]


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
