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
import shlex
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


@pytest.fixture
def rehearsal_preflight(tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()
    base, approval, source = tmp_path / "base.env", profile / "platform-approval.env", tmp_path / "source.yaml"
    source.write_text("kind: Pod\n")
    approved = profile / "approved-launch-profile.json"
    approved.write_text("{}\n")
    base.write_text(
        f"PLATFORM_WORK_ROOT={shlex.quote(str(tmp_path))}\n"
        "PLATFORM_PROFILE=profile\nRUNTIME_CLASS=kata-qemu-nvidia-gpu-snp\n"
        f"REHEARSAL_WORKLOAD_YAML={shlex.quote(str(source))}\n"
    )
    approval.write_text(
        "\n".join(f"SNP_MIN_REPORTED_TCB_{name}=0" for name in ("BOOTLOADER", "TEE", "SNP", "MICROCODE"))
    )
    return base, approval, source, approved, profile


def run_rehearsal_preflight(inputs, missing_commands=()):
    base, approval, _, _, _ = inputs
    script = section("trusted_system/07-run-snp-rehearsal.sh", "die()", "KUBECONFIG_PATH=")
    # Exercise the actual local preflight with deterministic command discovery;
    # no Docker, kubectl, network access or sudo is permitted in these tests.
    setup = (
        "set -Eeuo pipefail\n"
        'BASE_CONFIG="$1"; APPROVAL_ENV="$2"; SCRIPT_DIR="$3"\n'
        f"missing_commands=({' '.join(map(shlex.quote, missing_commands))})\n"
        'command() { if [[ "$1" == -v ]]; then '
        'for missing in "${missing_commands[@]}"; do [[ "$2" != "$missing" ]] || return 1; done; '
        'return 0; else builtin command "$@"; fi; }\n'
        "sudo() { echo UNEXPECTED_SUDO; return 99; }\n"
        "kubectl() { echo UNEXPECTED_KUBECTL; return 99; }\n"
        "docker() { echo UNEXPECTED_DOCKER; return 99; }\n"
        "curl() { echo UNEXPECTED_CURL; return 99; }\n"
    )
    return subprocess.run(
        [
            "bash",
            "-c",
            setup + script + "echo PREFLIGHT_OK",
            "preflight",
            str(base),
            str(approval),
            str(ROOT / "trusted_system"),
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )


def test_rehearsal_preflight_collects_all_known_conflicts(rehearsal_preflight):
    _, approval, source, approved, profile = rehearsal_preflight
    approval.write_text(
        "\n".join(f"SNP_MIN_REPORTED_TCB_{name}=''" for name in ("BOOTLOADER", "TEE", "SNP", "MICROCODE"))
    )
    source.unlink()
    approved.unlink()
    for name in ("rehearsal-collector-build", "reported-tcb-evidence"):
        path = profile / name
        path.mkdir()
        (path / "retained").write_bytes(b"retain original evidence")
    evidence = profile / "rehearsal-evidence.txt"
    evidence.write_bytes(b"retain original summary")
    before = {str(p): p.read_bytes() for p in profile.rglob("*") if p.is_file()}
    result = run_rehearsal_preflight(rehearsal_preflight, ("curl", "crictl"))
    assert result.returncode == 1
    assert "11 problem(s)" in result.stderr
    for expected in (
        "curl",
        "crictl",
        "source Pod YAML",
        "approved-launch-profile.json",
        "rehearsal-collector-build",
        "reported-tcb-evidence",
        "rehearsal-evidence.txt",
        "BOOTLOADER",
        "TEE",
        "SNP",
        "MICROCODE",
    ):
        assert expected in result.stderr
    assert "PREFLIGHT_OK" not in result.stdout and "UNEXPECTED" not in result.stdout
    assert before == {str(p): p.read_bytes() for p in profile.rglob("*") if p.is_file()}


def test_rehearsal_preflight_accepts_complete_inputs_without_writes(rehearsal_preflight):
    *_, profile = rehearsal_preflight
    before = sorted(profile.iterdir())
    result = run_rehearsal_preflight(rehearsal_preflight)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "PREFLIGHT_OK\n"
    assert sorted(profile.iterdir()) == before


@pytest.mark.parametrize("value", ["", "256", "-1", "01", "not-a-floor"])
def test_rehearsal_preflight_rejects_invalid_tcb_floor(rehearsal_preflight, value):
    _, approval, *_ = rehearsal_preflight
    approval.write_text(
        approval.read_text().replace("SNP_MIN_REPORTED_TCB_MICROCODE=0", f"SNP_MIN_REPORTED_TCB_MICROCODE='{value}'")
    )
    result = run_rehearsal_preflight(rehearsal_preflight)
    assert result.returncode == 1
    assert "1 problem(s)" in result.stderr
    assert "SNP_MIN_REPORTED_TCB_MICROCODE" in result.stderr


@pytest.mark.parametrize("name", ["rehearsal-collector-build", "reported-tcb-evidence", "rehearsal-evidence.txt"])
def test_rehearsal_preflight_rejects_dangling_output_symlinks(rehearsal_preflight, name):
    *_, profile = rehearsal_preflight
    link = profile / name
    link.symlink_to("absent-target")
    result = run_rehearsal_preflight(rehearsal_preflight)
    assert result.returncode == 1
    assert f"refusing to overwrite: {link}" in result.stderr
    assert link.is_symlink() and not link.exists()


def test_rehearsal_preflight_requires_source_path(rehearsal_preflight):
    base, *_ = rehearsal_preflight
    base.write_text(base.read_text() + "\nunset REHEARSAL_WORKLOAD_YAML\n")
    result = run_rehearsal_preflight(rehearsal_preflight)
    assert result.returncode == 1
    assert "REHEARSAL_WORKLOAD_YAML is required" in result.stderr


def test_rehearsal_preflight_reports_both_missing_configs(rehearsal_preflight):
    base, approval, *_ = rehearsal_preflight
    base.unlink()
    approval.unlink()
    result = run_rehearsal_preflight(rehearsal_preflight, ("docker",))
    assert result.returncode == 1
    assert "3 problem(s)" in result.stderr
    assert str(base) in result.stderr and str(approval) in result.stderr


def test_rehearsal_preflight_runs_before_cluster_and_cleanup():
    text = (ROOT / "trusted_system/07-run-snp-rehearsal.sh").read_text()
    assert (
        text.rindex("\nfinish_preflight\n")
        < text.index('"${KCTL[@]}" get runtimeclass')
        < text.index("trap cleanup EXIT")
    )
    assert '[[ ! -e "${COLLECTOR_DIR}" && ! -L "${COLLECTOR_DIR}" ]]' in text
    assert '[[ ! -e "${REHEARSAL_EVIDENCE}" && ! -L "${REHEARSAL_EVIDENCE}" ]]' in text
