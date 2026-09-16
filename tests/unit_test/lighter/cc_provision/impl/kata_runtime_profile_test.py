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

"""Offline tests for the reviewed Kata configuration derivation; no host installation."""

import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

# NVFlare supports Python 3.10, but these deployment helpers require Python 3.11+.
# Skip before loading the helpers; do not add a third-party TOML backport for tests.
tomllib = pytest.importorskip("tomllib", reason="CoCo deployment helpers require Python 3.11+")

ROOT = Path(__file__).resolve().parents[5] / "examples/devops/coco"
HELPER = ROOT / "shared/kata-runtime-profile.py"
API = runpy.run_path(str(HELPER))
ADMIN = runpy.run_path(str(ROOT / "admin/lib/workload-launch-profile.py"))
CONFIG = '[hypervisor.qemu]\n# Preserve this comment\nkernel_params = "pci=one pci=two quiet" # tail\ndefault_vcpus = 1\n[agent.kata]\ndebug = false\n'


def test_only_reviewed_setting_changes():
    derived = API["derive"](CONFIG)
    assert API["derive"](derived) == derived
    before, after = tomllib.loads(CONFIG), tomllib.loads(derived)
    before["hypervisor"]["qemu"]["kernel_params"] += " " + API["REQUIRED"]
    assert before == after
    assert "pci=one pci=two quiet" in derived
    assert "# Preserve this comment" in derived and "# tail" in derived


@pytest.mark.parametrize("old", ["resource", "attestation", "all"])
def test_existing_single_setting(old):
    source = CONFIG.replace("quiet", "quiet agent.guest_components_rest_api=" + old)
    result = API["derive"](source)
    API["require_token_api"](tomllib.loads(result)["hypervisor"]["qemu"]["kernel_params"])


@pytest.mark.parametrize(
    "value",
    [
        "",
        "agent.guest_components_rest_api",
        "agent.guest_components_rest_api=resource",
        "agent.guest_components_rest_api=all agent.guest_components_rest_api=all",
    ],
)
def test_missing_disabled_duplicate_launch_option(value):
    with pytest.raises(ValueError):
        API["require_token_api"](value)


def test_ambiguous_derivation_rejected():
    for params in [
        "agent.guest_components_rest_api=all agent.guest_components_rest_api=resource",
        "agent.guest_components_rest_api=unknown",
        "agent.guest_components_rest_api",
    ]:
        with pytest.raises(ValueError):
            API["derive"](CONFIG.replace("quiet", params))


def test_provenance_installed_and_launch_checks(tmp_path):
    upstream, approved, record, installed, launch = [
        tmp_path / n for n in ("upstream", "approved", "record", "installed", "launch")
    ]
    upstream.write_text(CONFIG)
    subprocess.run([sys.executable, str(HELPER), "derive", str(upstream), str(approved), str(record)], check=True)
    assert upstream.read_text() == CONFIG
    installed.write_bytes(approved.read_bytes())
    captured = {
        "launch_inputs": {"kernel_command_line": API["REQUIRED"]},
        "artifacts": {"kata_config": {"sha256": hashlib.sha256(approved.read_bytes()).hexdigest()}},
    }
    launch.write_text(json.dumps(captured))
    API["verify"](upstream, approved, record, installed, launch)
    installed.write_text(CONFIG)
    with pytest.raises(ValueError, match="Installed"):
        API["verify"](upstream, approved, record, installed, launch)
    API["enable"](installed)
    API["enable"](installed)
    assert installed.read_bytes() == approved.read_bytes()
    captured["launch_inputs"]["kernel_command_line"] = "quiet"
    launch.write_text(json.dumps(captured))
    with pytest.raises(ValueError, match="Require exactly"):
        API["verify"](upstream, approved, record, installed, launch)
    approved.write_text(approved.read_text().replace("default_vcpus = 1", "default_vcpus = 2"))
    with pytest.raises(ValueError, match="reviewed upstream derivation"):
        API["verify"](upstream, approved, record)


def test_reference_symlink_rejected(tmp_path):
    source = tmp_path / "source"
    source.write_text(CONFIG)
    alias = tmp_path / "alias"
    alias.symlink_to(source)
    with pytest.raises(ValueError, match="regular"):
        API["read_config"](alias)


@pytest.fixture
def deployed_profile(tmp_path):
    upstream, approved, record = [tmp_path / n for n in ("upstream.toml", "approved.toml", "record.json")]
    upstream.write_text(CONFIG)
    subprocess.run([sys.executable, str(HELPER), "derive", str(upstream), str(approved), str(record)], check=True)
    config_dir = tmp_path / "kata-containers"
    target = config_dir / "runtimes/qemu-nvidia-gpu-snp/configuration.toml"
    target.parent.mkdir(parents=True)
    target.write_text("# Managed by kata-deploy\n# Do not remove this header\n\n" + CONFIG)
    target.chmod(0o640)
    installed = config_dir / "configuration-qemu-nvidia-gpu-snp.toml"
    installed.symlink_to(target.relative_to(config_dir))
    return upstream, approved, record, installed, target


def test_install_stock_header_and_runtime_symlink(deployed_profile, tmp_path):
    upstream, approved, record, installed, target = deployed_profile
    original = target.read_text()
    reference_bytes = [p.read_bytes() for p in (upstream, approved, record)]
    stat = target.stat()
    command = [
        sys.executable,
        str(HELPER),
        "install",
        str(upstream),
        str(approved),
        str(record),
        "--installed",
        str(installed),
    ]
    subprocess.run(command, check=True)
    assert installed.is_symlink() and installed.resolve() == target
    assert target.read_text() == API["derive"](original)
    assert target.stat().st_mode == stat.st_mode
    assert (target.stat().st_uid, target.stat().st_gid) == (stat.st_uid, stat.st_gid)
    assert reference_bytes == [p.read_bytes() for p in (upstream, approved, record)]
    assert target.read_bytes() != approved.read_bytes()
    assert API["settings"](target.read_text()) == API["settings"](approved.read_text())
    inode = target.stat().st_ino
    subprocess.run(command, check=True)
    assert target.stat().st_ino == inode  # Idempotent: do not even replace the file.
    for role in ("trusted_system", "coco"):
        helper = ROOT / role / "lib/kata-runtime-profile.py"
        subprocess.run([sys.executable, str(helper), "enable", str(installed)], check=True)
        subprocess.run([sys.executable, str(helper), "check", str(installed)], check=True)
    captured = {
        "launch_inputs": {"kernel_command_line": API["REQUIRED"]},
        "artifacts": {"kata_config": {"sha256": hashlib.sha256(target.read_bytes()).hexdigest()}},
    }
    launch = tmp_path / "launch.json"
    launch.write_text(json.dumps(captured))
    API["verify"](upstream, approved, record, installed, launch)
    with pytest.raises(ValueError, match="requires the installed"):
        API["verify"](upstream, approved, record, launch=launch)
    # Comments are not settings, but captured evidence must bind the exact installed bytes.
    target.write_text(target.read_text() + "# changed after capture\n")
    with pytest.raises(ValueError, match="Captured Kata configuration"):
        API["verify"](upstream, approved, record, installed, launch)
    captured["artifacts"]["kata_config"]["sha256"] = hashlib.sha256(approved.read_bytes()).hexdigest()
    launch.write_text(json.dumps(captured))
    with pytest.raises(ValueError, match="Captured Kata configuration"):
        API["verify"](upstream, approved, record, installed, launch)


@pytest.mark.parametrize("value", ["2", "true", "1.0"])
def test_install_and_verify_reject_changed_settings(deployed_profile, value):
    upstream, approved, record, installed, target = deployed_profile
    for text in (CONFIG, API["derive"](CONFIG)):
        changed = text.replace("default_vcpus = 1", "default_vcpus = " + value)
        target.write_text(changed)
        with pytest.raises(ValueError, match="differ from both"):
            API["install"](upstream, approved, record, installed)
        assert target.read_text() == changed
        with pytest.raises(ValueError, match="Installed Kata configuration"):
            API["verify"](upstream, approved, record, installed)


@pytest.mark.parametrize("kind", ["outside", "broken", "directory"])
def test_unsafe_runtime_links_rejected(tmp_path, kind):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    outside = tmp_path / "outside.toml"
    outside.write_text(CONFIG)
    target = {"outside": outside, "broken": config_dir / "missing", "directory": config_dir}[kind]
    alias = config_dir / "configuration.toml"
    alias.symlink_to(target)
    with pytest.raises((ValueError, FileNotFoundError)):
        API["enable"](alias)
    assert outside.read_text() == CONFIG
    assert alias.is_symlink()


def test_source_and_isolated_role_helpers(tmp_path):
    for role in ("trusted_system", "coco"):
        source = runpy.run_path(str(ROOT / role / "lib/kata-runtime-profile.py"))
        assert source["derive"](CONFIG) == API["derive"](CONFIG)
    isolated = tmp_path / "kit/lib/kata-runtime-profile.py"
    isolated.parent.mkdir(parents=True)
    isolated.write_bytes(HELPER.read_bytes())
    config = tmp_path / "runtime.toml"
    config.write_text(API["derive"](CONFIG))
    subprocess.run([sys.executable, str(isolated), "check", str(config)], check=True)


def test_admin_rejects_old_or_missing_capability(tmp_path):
    path = tmp_path / "contract.json"
    contract = {
        "schema": "coco-approved-workload-launch/v2",
        "profile_id": "reviewed",
        "runtime_class": "kata-qemu-nvidia-gpu-snp",
        "kata_version": "3.29.0",
        "guest_token_api": API["CAPABILITY"],
        "kata_deploy_image": "registry/runtime@sha256:" + "a" * 64,
        "kata_config_sha256": "b" * 64,
        "launch_inputs_sha256": "c" * 64,
        "vm_defaults": {"vcpus": 1, "memory_mib": 8192},
        "pod_constraints": {
            "container_count": 1,
            "gpu_resource": "nvidia.com/pgpu",
            "gpu_count": 1,
            "cpu_memory_resources": "omitted",
            "host_namespaces": False,
            "allowed_annotations": ["io.katacontainers.config.hypervisor.cc_init_data"],
        },
    }

    def load(value):
        path.write_text(json.dumps(value))
        return ADMIN["load_profile"](
            path, hashlib.sha256(path.read_bytes()).hexdigest(), contract["runtime_class"], "3.29.0"
        )

    assert load(contract)["guest_token_api"] == API["CAPABILITY"]
    with pytest.raises(ValueError):
        load({**contract, "schema": "coco-approved-workload-launch/v1"})
    with pytest.raises(ValueError):
        load({k: v for k, v in contract.items() if k != "guest_token_api"})
    with pytest.raises(ValueError):
        load({**contract, "guest_token_api": "resource-only"})


def test_exported_contract_and_measured_option(tmp_path):
    runtime = "kata-qemu-nvidia-gpu-snp"
    image = "registry/runtime@sha256:" + "a" * 64
    resources = {"limits": {"nvidia.com/pgpu": 1}, "requests": {"nvidia.com/pgpu": 1}}
    approved = API["derive"](CONFIG)
    profile = {
        "guest_token_api": API["CAPABILITY"],
        "kata_config": tomllib.loads(approved),
        "kata_config_sha256": hashlib.sha256(approved.encode()).hexdigest(),
        "runtime_class": runtime,
        "pod_resources": resources,
        "runtime_default_vcpus": 1,
        "runtime_default_memory_mib": 8192,
    }
    actual = {
        "launch_inputs": {"kernel_command_line": API["REQUIRED"], "smp": "1", "memory": "8192M"},
        "pod_resources": [resources],
        "artifacts": {"kata_config": {"sha256": profile["kata_config_sha256"]}},
    }
    actual_path = tmp_path / "rehearsal-collector-build/actual-launch.json"
    actual_path.parent.mkdir()
    profile_path = tmp_path / "approved-launch-profile.json"

    def export():
        actual["launch_inputs_sha256"] = hashlib.sha256(
            json.dumps(actual["launch_inputs"], sort_keys=True).encode()
        ).hexdigest()
        profile_path.write_text(json.dumps(profile))
        actual_path.write_text(json.dumps(actual))
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "trusted_system/export-workload-launch-profile.py"),
                str(tmp_path),
                "reviewed",
                "3.29.0",
                runtime,
                image,
                hashlib.sha256(profile_path.read_bytes()).hexdigest(),
                hashlib.sha256(actual_path.read_bytes()).hexdigest(),
            ],
            capture_output=True,
            text=True,
        )

    result = export()
    assert result.returncode == 0, result.stderr
    contract = tmp_path / "contract.json"
    contract.write_text(result.stdout)
    assert ADMIN["load_profile"](contract, hashlib.sha256(contract.read_bytes()).hexdigest(), runtime, "3.29.0")
    actual["launch_inputs"]["kernel_command_line"] = "quiet"
    assert export().returncode != 0
    actual["launch_inputs"]["kernel_command_line"] = API["REQUIRED"]
    del profile["guest_token_api"]
    assert export().returncode != 0
