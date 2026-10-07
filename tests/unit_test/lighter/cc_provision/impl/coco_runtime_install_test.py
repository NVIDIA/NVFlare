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

"""Exercise target selection in the real installer without modifying a host."""

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from nvflare.lighter.cc_provision.kata_runtime_profile import RUNTIME_TARGETS, runtime_target
from tests.unit_test.lighter.cc_provision.impl.deployment_guards_test import require_coco_bash

ROOT = Path(__file__).resolve().parents[5]
ROLE = ROOT / "examples/devops/coco/coco"
HELPER = ROOT / "nvflare/lighter/cc_provision/kata_runtime_profile.py"

STUB_COMMON = r"""
load_config() {
  RUNTIME_CLASS="$TEST_RUNTIME"
  TEE_NAME=fixture
  TEE_NODE_LABEL_KEY=fixture.tee
  KATA_VERSION=3.29.0
  GPU_OPERATOR_VERSION=v26.3.1
  GPU_RESOURCE=nvidia.com/pgpu
}
record() { printf '%s\n' "$*" >>"$TEST_CALLS"; }
die() { printf '%s\n' "$*" >&2; exit 1; }
log() { record "log $*"; }
require_root_or_sudo() { :; }
need() { record "need $*"; }
wait_for() { record "wait $1"; shift 2; "$@"; }
helmctl() { record "helm $*"; }
validate_runtime_prerequisites() { record "prerequisites $*"; }
kctl() {
  record "kubectl $*"
  case "$*" in
    *metadata.name*) printf 'fixture-node\n';;
    *cc*ready*) printf 'true\n';;
    *allocatable*) printf '1\n';;
  esac
}
as_root() {
  record "root $*"
  case "$1" in
    test|grep|systemctl|python3) return 0;;
    awk) printf '/opt/kata/fixture.toml\n';;
    containerd) printf '[plugins.cri.containerd.runtimes.%s]\n' "$RUNTIME_CLASS";;
    ctr) printf 'io.containerd.snapshotter.v1 nydus-for-kata-tee - ok\n';;
    *) die "unexpected privileged invocation: $*";;
  esac
}
lspci() { record "lspci $*"; printf '0000:01:00.0 0302: 10de:2330\n'; }
readlink() { printf '/sys/bus/pci/drivers/vfio-pci\n'; }
"""


@pytest.mark.parametrize("runtime", RUNTIME_TARGETS)
def test_install_selects_target_without_gpu_dependencies_on_cpu_only(tmp_path, runtime):
    bash = require_coco_bash()
    bootstrap = tmp_path / "coco/bootstrap"
    for directory in (bootstrap / "lib", tmp_path / "coco/lib", tmp_path / "coco/public", tmp_path / "bin"):
        directory.mkdir(parents=True)
    script = bootstrap / "20-install-coco-gpu.sh"
    shutil.copyfile(ROLE / "bootstrap/20-install-coco-gpu.sh", script)
    shutil.copyfile(HELPER, tmp_path / "coco/lib/kata-runtime-profile.py")
    (bootstrap / "lib/common.sh").write_text(STUB_COMMON)
    (tmp_path / "bin/python3").symlink_to(sys.executable)
    chart = tmp_path / "coco/public/kata-deploy-3.29.0.tgz"
    chart.write_bytes(b"pinned chart fixture")
    image = "quay.io/fixture/kata@sha256:" + "a" * 64
    (chart.parent / "kata-platform.env").write_text(
        'RUNTIME_CLASS="kata-qemu-nvidia-gpu-snp"\n'
        f'KATA_CHART_TGZ_SHA256="{hashlib.sha256(chart.read_bytes()).hexdigest()}"\n'
        f'KATA_DEPLOY_AMD64="{image}"\n'
    )
    calls = tmp_path / "calls"
    result = subprocess.run(
        [bash, str(script)],
        env={
            **os.environ,
            "PATH": str(tmp_path / "bin") + os.pathsep + os.environ["PATH"],
            "TEST_RUNTIME": runtime,
            "TEST_CALLS": str(calls),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    commands = calls.read_text()
    assert f"image.reference={image}" in commands
    assert f"wait RuntimeClass {runtime}" in commands
    assert f"check-target {runtime} /opt/kata/fixture.toml" in commands
    assert f"prerequisites {runtime} /opt/kata/fixture.toml" in commands
    assert "check /opt/kata/fixture.toml --runtime /opt/kata/bin/kata-runtime" in commands
    has_gpu = bool(runtime_target(runtime)["gpu_count"])
    assert ("helm upgrade --install gpu-operator" in commands) is has_gpu
    assert ("need lspci" in commands) is has_gpu
    assert ("nvidia.com/gpu.workload.config=vm-passthrough" in commands) is has_gpu
    assert ("wait NVIDIA confidential-computing readiness label" in commands) is has_gpu
    if has_gpu:
        assert "ccManager.defaultMode=on" in commands
        assert "vfioManager.enabled=true" in commands
        assert "bound to vfio-pci" in commands


@pytest.mark.parametrize("script", ["35-repin-kata-deployment.sh", "60-verify-platform.sh"])
@pytest.mark.parametrize("runtime", RUNTIME_TARGETS)
def test_repin_and_verify_resolve_runtime_config_from_canonical_mapping(tmp_path, script, runtime):
    source = (ROLE / script).read_text()
    line = next(line for line in source.splitlines() if line.startswith("config_name="))
    code = f'set -e\nSCRIPT_DIR={str(ROLE)!r}\nRUNTIME_CLASS={runtime!r}\n{line}\nprintf "%s" "$config_name"\n'
    result = subprocess.run(
        [require_coco_bash(), "-c", code],
        env={**os.environ, "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]},
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == runtime_target(runtime)["config_name"]
    assert 'check-target "$RUNTIME_CLASS"' in source
    assert 'validate_runtime_prerequisites "$RUNTIME_CLASS"' in source
    assert "configuration-qemu-nvidia-gpu-snp.toml" not in source


@pytest.mark.parametrize("gpu_count", [0, 1])
def test_host_gpu_preflight_is_conditional(tmp_path, gpu_count):
    source = (ROLE / "bootstrap/00-verify-host.sh").read_text()
    start = source.index("if ((gpu_count > 0)); then")
    stop = source.index("grep -qw swap", start)
    # CPU-only must not inspect GPU devices, IOMMU state or installed drivers.
    code = (
        f"gpu_count={gpu_count}\n"
        "pass() { printf 'pass %s\\n' \"$*\"; }\n"
        "fail() { printf 'fail %s\\n' \"$*\"; }\n"
        "warn() { :; }\n"
        "has() { return 1; }\n"
        "lsmod() { :; }\n" + source[start:stop]
    )
    result = subprocess.run([require_coco_bash(), "-c", code], text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    if not gpu_count:
        assert "CPU-only target" in result.stdout
        assert "fail " not in result.stdout
    else:
        assert "NVIDIA GPU not detected" in result.stdout
