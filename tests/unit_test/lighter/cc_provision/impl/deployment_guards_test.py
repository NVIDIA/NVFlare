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

"""Offline checks of deployment snippets; no Docker, network, sudo or cluster access."""

import ast
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[5] / "examples/devops/coco"
SERVICE = "service/12-install-trusted-service-handoff.sh"
LAUNCH = "coco/50-launch-handoff.sh"
PREFIX = 'set -Eeuo pipefail\ndie() { echo "$*"; exit 2; }\n'


def section(name, start, end):
    source = (ROOT / name).read_text()
    return source[source.index(start) : source.index(end, source.index(start))]


@pytest.mark.parametrize("quantity", [1, "1"])
def test_profile_resource_quantities_match_kubernetes(quantity):
    source = section("trusted_system/05-define-approved-launch-profile.py", "resources =", "with config_path.open")
    resources = {"limits": {"nvidia.com/pgpu": quantity}}
    namespace = {"spec": {"containers": [{"resources": resources}]}}
    exec(compile(ast.parse(source), "profile-resource-normalization", "exec"), namespace)
    assert namespace["resources"] == {"limits": {"nvidia.com/pgpu": "1"}, "requests": {"nvidia.com/pgpu": "1"}}


@pytest.mark.parametrize("cached,pull_status", [(True, 0), (False, 0), (False, 1)])
def test_setup_image_is_available_before_deployment(cached, pull_status):
    source = section("service/05-deploy-trustee.sh", 'sudo docker image inspect "${SETUP_IMAGE}"', "install -d")
    stubs = (
        'SETUP_IMAGE="alpine/openssl@sha256:fixture"\n'
        'sudo() { "$@"; }\n'
        'docker() { if [[ "$1" == image ]]; then return '
        + str(0 if cached else 1)
        + '; fi; printf "PULL %s\\n" "$2"; return '
        + str(pull_status)
        + "; }\n"
    )
    result = subprocess.run(["bash", "-c", PREFIX + stubs + source + "echo CONTINUE"], capture_output=True, text=True)
    assert ("PULL alpine/openssl@sha256:fixture" in result.stdout) is not cached
    assert (result.returncode == 0) == (cached or pull_status == 0)
    assert ("CONTINUE" in result.stdout) == (result.returncode == 0)


@pytest.mark.parametrize("failure", ["none", "hosts.toml", "ca.crt", "curl"])
def test_registry_preflight_fails_before_launch(failure):
    source = section(LAUNCH, "need_cmd curl", "kctl apply --dry-run=server")
    stubs = (
        'REGISTRY_HOST="secure.example.invalid:5000"\n'
        f'FAILURE="{failure}"\n'
        'need_cmd() { [[ "$1" == curl ]]; }\n'
        'need_file() { [[ "$1" != *"/$FAILURE" ]] || die "missing $1"; }\n'
        'curl() { printf "%s\\n" "$@"; [[ "$FAILURE" != curl ]]; }\n'
    )
    result = subprocess.run(["bash", "-c", PREFIX + stubs + source + "echo CONTINUE"], capture_output=True, text=True)
    assert (result.returncode == 0) == (failure == "none")
    assert ("CONTINUE" in result.stdout) == (failure == "none")
    if failure == "none":
        assert "/etc/containerd/certs.d/secure.example.invalid:5000/ca.crt" in result.stdout
        assert "https://secure.example.invalid:5000/v2/" in result.stdout
        assert "--max-time\n15" in result.stdout


@pytest.mark.parametrize("role", ["release", "launch", "remove"])
@pytest.mark.parametrize(
    "answer,close_input", [("correct\n", True), ("wrong\n", True), ("", True), ("", False), ("correct", False)]
)
def test_bounded_approval_and_safe_key_retention(role, answer, close_input):
    if role == "release":
        source = section(SERVICE, "read -r -t 120", "# Some OS images")
        expected, setup = "release-v1", 'RELEASE_NAME="release-v1"\n'
    elif role == "launch":
        source = section(LAUNCH, "read -r -t 120", 'kctl apply -f "${POD_FILE}"')
        expected, setup = "APPLY", ""
    else:
        source = section(SERVICE, "read -r -t 120 -p 'Type REMOVE", 'if [[ "${REMOVE_KEY}"')
        source += 'printf "KEY_CHOICE=%s\\n" "$REMOVE_KEY"\n'
        expected, setup = "REMOVE", ""
    # Exercise Bash timeout behavior without waiting two minutes per case.
    source = source.replace("-t 120", "-t 0.05")
    script = PREFIX + setup + source + "echo CONTINUE"
    answer = answer.replace("correct", expected)
    with subprocess.Popen(
        ["bash", "-c", script], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ) as process:
        if close_input:
            output, _ = process.communicate(answer, timeout=5)
        else:
            process.stdin.write(answer)
            process.stdin.flush()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                raise
            output, _ = process.communicate()
        approved = close_input and answer == expected + "\n"
        assert (process.returncode == 0) == (role == "remove" or approved)
        if role == "remove":
            assert "KEY_CHOICE=" + (expected if approved else ("wrong" if answer == "wrong\n" else "")) + "\n" in output
        else:
            assert ("CONTINUE" in output) == approved
