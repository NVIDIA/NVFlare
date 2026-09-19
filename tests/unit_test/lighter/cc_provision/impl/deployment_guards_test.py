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

"""Offline tests of callable deployment helpers; no external operations."""

import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from nvflare.lighter.cc_provision.workload_security import normalize_resources

ROOT = Path(__file__).resolve().parents[5] / "examples/devops/coco"
SERVICE = "service/12-install-trusted-service-handoff.sh"
LAUNCH = "coco/50-launch-handoff.sh"
PREFIX = "set -Eeuo pipefail\n" + f'source {shlex.quote(str(ROOT / "shared/lib/common-base.sh"))}\n'


def require_coco_bash():
    """Check shell prerequisites without skipping the Python/static tests."""
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("CoCo shell tests require bash on PATH")
    # Lowercase expansion and fractional read timeouts need Bash 4; empty
    # arrays under nounset also require the Bash 4.4 behavior used by the harness.
    probe = (
        "set -eu\n"
        "printf '%s\\n' \"$BASH_VERSION\"\n"
        "value=ABC; [[ ${value,,} == abc ]]\n"
        'items=(); for item in "${items[@]}"; do :; done\n'
        'read -r -t 0.05 item <<< ready; [[ "$item" == ready ]]\n'
    )
    try:
        result = subprocess.run([bash, "-c", probe], capture_output=True, text=True, timeout=5)
    except subprocess.TimeoutExpired:
        pytest.skip("CoCo shell prerequisite probe timed out")
    if result.returncode:
        pytest.skip(
            "CoCo shell tests require lowercase expansion, fractional read timeouts and nounset-safe empty arrays "
            f"(Bash 4.4+; macOS system Bash 3.2 is insufficient): {result.stdout.strip()} {result.stderr.strip()}"
        )
    return bash


@pytest.fixture(scope="module")
def coco_bash():
    return require_coco_bash()


@pytest.mark.parametrize("available,returncode", [(False, 0), (True, 1), (True, 0)])
def test_shell_prerequisite_detection(monkeypatch, available, returncode):
    monkeypatch.setattr(shutil, "which", lambda _: "/test/bash" if available else None)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["timeout"] == 5
        return subprocess.CompletedProcess(command, returncode, "test version", "unsupported shell feature")

    monkeypatch.setattr(subprocess, "run", run)
    if available and returncode == 0:
        assert require_coco_bash() == "/test/bash"
    else:
        with pytest.raises(pytest.skip.Exception, match="CoCo shell tests require"):
            require_coco_bash()
    assert len(calls) == int(available)


def test_shell_prerequisite_probe_is_bounded(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: "/test/bash")

    def run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(pytest.skip.Exception, match="prerequisite probe timed out"):
        require_coco_bash()


@pytest.mark.parametrize("quantity", [1, "1"])
def test_profile_resource_quantities_match_kubernetes(quantity):
    resources = {"limits": {"nvidia.com/pgpu": quantity}}
    assert normalize_resources(resources) == {"limits": {"nvidia.com/pgpu": "1"}, "requests": {"nvidia.com/pgpu": "1"}}


@pytest.mark.parametrize("cached,pull_status", [(True, 0), (False, 0), (False, 1)])
def test_setup_image_is_available_before_deployment(cached, pull_status, coco_bash):
    source = 'ensure_setup_image "$SETUP_IMAGE"\n'
    stubs = (
        'SETUP_IMAGE="alpine/openssl@sha256:fixture"\n'
        'sudo() { "$@"; }\n'
        'docker() { if [[ "$1" == image ]]; then return '
        + str(0 if cached else 1)
        + '; fi; printf "PULL %s\\n" "$2"; return '
        + str(pull_status)
        + "; }\n"
    )
    result = subprocess.run(
        [coco_bash, "-c", PREFIX + stubs + source + "echo CONTINUE"], capture_output=True, text=True
    )
    assert ("PULL alpine/openssl@sha256:fixture" in result.stdout) is not cached
    assert (result.returncode == 0) == (cached or pull_status == 0)
    assert ("CONTINUE" in result.stdout) == (result.returncode == 0)


@pytest.mark.parametrize("failure", ["none", "hosts.toml", "ca.crt", "curl"])
def test_registry_preflight_fails_before_launch(failure, coco_bash):
    source = 'check_registry_trust "$REGISTRY_HOST"\n'
    stubs = (
        'REGISTRY_HOST="secure.example.invalid:5000"\n'
        f'FAILURE="{failure}"\n'
        'need_cmd() { [[ "$1" == curl ]]; }\n'
        'need_file() { [[ "$1" != *"/$FAILURE" ]] || die "missing $1"; }\n'
        'curl() { printf "%s\\n" "$@"; [[ "$FAILURE" != curl ]]; }\n'
    )
    result = subprocess.run(
        [coco_bash, "-c", PREFIX + stubs + source + "echo CONTINUE"], capture_output=True, text=True
    )
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
def test_bounded_approval_and_safe_key_retention(role, answer, close_input, coco_bash):
    expected = {"release": "release-v1", "launch": "APPLY", "remove": "REMOVE"}[role]
    call = f'confirm_action {shlex.quote(expected)} "Approve: " 0.05'
    if role == "remove":
        source = f"if {call}; then echo KEY_CHOICE=REMOVE; else echo KEY_CHOICE=; fi\n"
    else:
        source = call + " || die 'approval failed'\n"
    script = PREFIX + source + "echo CONTINUE"
    answer = answer.replace("correct", expected)
    with subprocess.Popen(
        [coco_bash, "-c", script], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
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
            assert "KEY_CHOICE=" + (expected if approved else "") + "\n" in output
        else:
            assert ("CONTINUE" in output) == approved


@pytest.fixture
def rehearsal_preflight(tmp_path, coco_bash):
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
    script = f'source {shlex.quote(str(ROOT / "trusted_system/lib/rehearsal-preflight.sh"))}\nrehearsal_preflight\n'
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
        text.index("\nrehearsal_preflight\n")
        < text.index('"${KCTL[@]}" get runtimeclass')
        < text.index("trap cleanup EXIT")
    )
    assert '[[ ! -e "${COLLECTOR_DIR}" && ! -L "${COLLECTOR_DIR}" ]]' in text
    assert '[[ ! -e "${REHEARSAL_EVIDENCE}" && ! -L "${REHEARSAL_EVIDENCE}" ]]' in text
