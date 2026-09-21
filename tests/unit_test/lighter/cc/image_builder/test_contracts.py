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

"""Security and application-neutral contracts, runnable without a guest."""

import base64
import hashlib
import json
import os
import stat
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils
from cvm.build import config
from cvm.build.cvm import kernel_command_line
from cvm.build.provisioning import (
    HARDENING_SYSCTL,
    chrony_configuration,
    docker_configuration,
    install_files,
    load_config,
    mask,
    scrub_identity,
    sudoers_files_for,
    suppress_service_starts,
    validate_time_servers,
)
from cvm.common import measurements as report_measurements
from cvm.common.contracts import (
    DISK_ROLES,
    HEADER_BYTES,
    binding,
    binding_id,
    qemu_binding,
    resource_path,
    validate_resource,
)
from cvm.common.errors import BuildError
from cvm.common.evidence import serial_evidence, serial_frames
from cvm.common.firewall import firewall_rules
from cvm.common.io import canonical
from cvm.common.linux import memory_file, validate_core_policy
from cvm.common.luks import validate_luks_metadata, validate_mapping
from cvm.common.measurements import measurements, validate_measurements
from cvm.common.services import validate_service
from cvm.common.validation import runtime_config
from cvm.host import platforms
from cvm.host.launcher import qemu_command
from cvm.runtime import bootstrap as runtime
from cvm.runtime import gpu
from cvm.runtime import platforms as guest_platforms
from cvm.runtime.attestation import ATTESTATION_BUDGET_SECONDS, authorized_key, validate_token
from cvm.runtime.integrity import healthy_status
from cvm.runtime.storage import disk_device
from cvm.trustee.admin import verify_readback


class BindingTests(unittest.TestCase):
    def test_piped_core_collectors_are_rejected_for_secret_children(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "core_pattern"
            for value in ("|/usr/share/apport/apport", " |/usr/lib/systemd/systemd-coredump\n"):
                path.write_text(value)
                with self.assertRaises(BuildError):
                    validate_core_policy(path)
            for value in ("core\n", "/dev/null\n", "\n"):
                path.write_text(value)
                validate_core_policy(path)

    def test_large_reference_survives_console_framing(self):
        evidence = {"ccel": base64.b64encode(os.urandom(65536) + bytes(65536)).decode()}
        frames = serial_frames(evidence)
        self.assertLessEqual(max(map(len, frames)), 1060)
        self.assertIsNone(serial_evidence("\n".join(frames[:-1])))
        self.assertIsNone(serial_evidence("boot noise\n" + "\n".join(frames)[:-1]))
        self.assertEqual(serial_evidence("boot noise\n" + "\n".join(frames) + "\n"), evidence)

    def test_reference_survives_journal_console_prefix(self):
        evidence = {"ccel": base64.b64encode(os.urandom(4096)).decode()}
        frames = serial_frames(evidence)
        output = "boot noise\r\n" + "\r\n".join(f"[   14.977856] python3[741]: {frame}" for frame in frames)
        self.assertIsNone(serial_evidence(output))
        self.assertEqual(serial_evidence(output + "\r\n"), evidence)
        with self.assertRaises(BuildError):
            serial_evidence(output + "\r\n[   15.021668] python3[741]: CVM_REFERENCE_V2 1/99 AAAA\r\n")

    def test_domain_and_entire_header(self):
        header = bytes(HEADER_BYTES)
        self.assertEqual(binding(header), hashlib.sha256(b"nvflare-vault-v2\0" + header).digest())
        for index in (0, 16384, 32768, HEADER_BYTES - 1):
            modified = bytearray(header)
            modified[index] = 1
            self.assertNotEqual(binding(modified), binding(header))

    def test_truncated_header_denied(self):
        for length in (0, 1024, HEADER_BYTES - 1, HEADER_BYTES + 1):
            with self.subTest(length=length), self.assertRaises(BuildError):
                binding(bytes(length))

    def test_canonical_platform_encodings(self):
        value = bytes.fromhex("fbff" * 16)
        self.assertEqual(binding_id("amd_sev_snp", value), value.hex())
        self.assertNotIn("/", binding_id("amd_sev_snp", value))
        self.assertEqual(binding_id("intel_tdx", value), value.hex() + "0" * 32)
        self.assertEqual(base64.b64decode(qemu_binding("intel_tdx", value)), value + bytes(16))
        for platform in platforms.PLATFORMS:
            self.assertEqual(validate_resource(resource_path("bundle-1", platform, value))[1], "bundle-1")

    def test_path_aliases_denied(self):
        snp = binding_id("amd_sev_snp", bytes(32))
        for value in (
            "keys/../" + snp,
            "resource/keys/a/" + snp,
            "keys/a/" + snp + "=",
            "keys/a/" + base64.urlsafe_b64encode(bytes(32)).decode().rstrip("="),
            "keys/a/" + snp[:-1] + "B",
            "keys/a/" + "a" * 96,
            "keys/a/" + "A" * 64 + "0" * 32,
            "keys/a/" + snp + "/extra",
        ):
            with self.subTest(path=value), self.assertRaises(BuildError):
                validate_resource(value)

    def test_snp_measurement_matches_upstream_hex_claim(self):
        report = bytearray(1184)
        launch = bytes.fromhex("fbff" * 24)
        report[144:192] = launch
        expected = {"snp.measurement": launch.hex()}
        self.assertEqual(measurements("amd_sev_snp", report), expected)
        validate_measurements("amd_sev_snp", expected)
        for value in (launch.hex().upper(), launch.hex()[:-1], base64.b64encode(launch).decode()):
            with self.subTest(value=value), self.assertRaises(BuildError):
                validate_measurements("amd_sev_snp", {"snp.measurement": value})

    def test_frozen_memfd_cannot_change(self):
        with memory_file(b"verified header", sealed=True) as fd:
            with self.assertRaises(OSError):
                os.write(fd, b"x")
            with self.assertRaises(OSError):
                os.ftruncate(fd, 0)

    def test_tdx_report_and_nonce(self):
        nonce = os.urandom(64)
        report = bytearray(1024)
        report[0] = 0x81
        report[128:192] = nonce
        report[576:624] = bytes(range(48))
        self.assertEqual(report_measurements.parse_tdx_report(report, nonce), bytes(range(48)))
        with self.assertRaises(BuildError):
            report_measurements.parse_tdx_report(report, bytes(64))
        with self.assertRaises(BuildError):
            report_measurements.parse_tdx_report(report[:-1], nonce)

    def test_tdx_measurements_include_rtmr0(self):
        report = bytearray(1024)
        for offset, value in ((528, 1), (720, 2), (768, 3), (816, 4)):
            report[offset : offset + 48] = bytes([value]) * 48
        self.assertEqual(
            report_measurements.measurements("intel_tdx", report),
            {
                "mr_td": (bytes([1]) * 48).hex(),
                "rtmr_0": (bytes([2]) * 48).hex(),
                "rtmr_1": (bytes([3]) * 48).hex(),
                "rtmr_2": (bytes([4]) * 48).hex(),
            },
        )

    def test_snp_report_and_nonce(self):
        nonce = os.urandom(64)
        report = bytearray(1184)
        struct.pack_into("<I", report, 0, 3)
        report[80:144] = nonce
        report[192:224] = bytes(range(32))
        self.assertEqual(report_measurements.parse_snp_report(report, nonce), bytes(range(32)))
        with self.assertRaises(BuildError):
            report_measurements.parse_snp_report(report, bytes(64))

    def test_tdx_nonzero_padding_denied(self):
        with patch("cvm.runtime.platforms.local_binding", return_value=bytes(47) + b"x"), self.assertRaises(BuildError):
            guest_platforms.verify_local_binding("intel_tdx", bytes(32))


class PlatformTests(unittest.TestCase):
    def setUp(self):
        self.profile = {"platforms": {name: {"enabled": True} for name in platforms.PLATFORMS}}

    def test_explicit_override_on_plain_host(self):
        self.assertEqual(platforms.select_platform(self.profile, "amd_sev_snp", set()), "amd_sev_snp")

    def test_auto_detect(self):
        self.assertEqual(platforms.select_platform(self.profile, capabilities={"intel_tdx"}), "intel_tdx")

    def test_no_ambiguous_or_missing_fallback(self):
        for values in (set(), set(platforms.PLATFORMS), {"tpm"}):
            with self.assertRaises(BuildError):
                platforms.select_platform(self.profile, capabilities=values)

    def test_disabled_or_unknown_override(self):
        self.profile["platforms"]["intel_tdx"]["enabled"] = False
        for name in ("intel_tdx", "intel_tdx_vtpm", "none"):
            with self.assertRaises(BuildError):
                platforms.select_platform(self.profile, name, set())

    def test_vendor_alone_is_not_capability(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "cpuinfo").write_text("vendor_id : GenuineIntel\nflags : vmx\n")
            self.assertEqual(platforms.host_capabilities(root, root), set())
            (root / "module/kvm_intel/parameters").mkdir(parents=True)
            (root / "module/kvm_intel/parameters/tdx").write_text("Y\n")
            self.assertEqual(platforms.host_capabilities(root, root), {"intel_tdx"})
            (root / "cpuinfo").write_text("vendor_id : GenuineIntel\nflags : vmx tdx\n")
            (root / "module/kvm_intel/parameters/tdx").write_text("N\n")
            self.assertEqual(platforms.host_capabilities(root, root), set())


class ProfileTests(unittest.TestCase):
    def test_profile_rejects_typos_and_removed_options(self):
        path = Path(self.temp_directory()) / "profile.yml"
        for name in ("gpu_cont", "trustee_patch_digest", "gpu_attestation_binary"):
            with self.subTest(name=name):
                path.write_text(name + ": invalid\n")
                with self.assertRaisesRegex(BuildError, "Unknown profile fields: " + name):
                    config.profile(path)

    def test_root_overlay_defaults_to_half_of_guest_ram(self):
        profile = {"memory_gib": 8}
        self.assertEqual(config.resolve_root_overlay_max_mib(profile), 4096)
        self.assertEqual(profile["root_overlay_max_mib"], 4096)

    def test_root_overlay_accepts_an_explicit_capacity(self):
        profile = {"memory_gib": 8, "root_overlay_max_mib": 3072}
        self.assertEqual(config.resolve_root_overlay_max_mib(profile), 3072)

    def test_root_overlay_rejects_invalid_or_impossible_capacities(self):
        for value in (0, -1, True, "4096", 8193):
            with self.subTest(value=value), self.assertRaises(BuildError):
                config.resolve_root_overlay_max_mib({"memory_gib": 8, "root_overlay_max_mib": value})

    def test_root_overlay_limit_is_in_the_measured_command_line(self):
        command = kernel_command_line("ab" * 32, 1024, 3072)
        self.assertIn("cvm.root_overlay_max_mib=3072", command.split())
        script = config.SOURCE / "initramfs/scripts/local-bottom/overlay_root"
        source = script.read_text()
        self.assertIn("cvm.root_overlay_max_mib=*", source)
        self.assertIn("size=${root_overlay_max_mib}M", source)

    def test_verity_cmdline_rejects_duplicates_including_empty_first_values(self):
        source = (config.SOURCE / "initramfs/scripts/local-top/verity_root").read_text()
        # Execute the shipped parser and validation, before any device operations.
        source = source.split("modprobe dm_verity", 1)[0].replace(
            ". /scripts/functions", 'panic() { printf "%s\\n" "$1" >&2; exit 1; }'
        )
        source = 'cat() { printf "%s\\n" "$CVM_TEST_CMDLINE"; }\n' + source
        valid = "roothash=" + "a" * 64 + " verity_hash_offset=4096"
        for command, expected in (
            (valid, ""),
            ("roothash= " + valid, "Duplicate verity root hash"),
            (valid + " roothash=" + "b" * 64, "Duplicate verity root hash"),
            ("verity_hash_offset= " + valid, "Duplicate verity hash offset"),
            (valid + " verity_hash_offset=8192", "Duplicate verity hash offset"),
        ):
            with self.subTest(command=command):
                result = subprocess.run(
                    ["sh", "-c", source], env=dict(os.environ, CVM_TEST_CMDLINE=command), capture_output=True, text=True
                )
                self.assertEqual(result.returncode, 1 if expected else 0)
                self.assertIn(expected, result.stderr)

    def test_gpu_profile_reallocates_large_pci_bars(self):
        command = kernel_command_line("ab" * 32, 1024, 3072, "nvidia_cc")
        self.assertIn("pci=realloc,nocrs", command.split())

    def test_gpu_count_is_bounded_by_qemu_pcie_slots(self):
        path = Path(self.temp_directory()) / "profile.yml"
        path.write_text("gpu_count: 9\n")
        with self.assertRaises(BuildError):
            config.profile(path)

    def shipped_gpu_policy(self):
        return json.loads((config.SOURCE / "config/gpu_policy.json").read_text())

    def write_gpu_policy(self, value):
        path = Path(tempfile.mkdtemp(dir=self.temp_directory())) / "gpu_policy.json"
        path.write_text(json.dumps(value))
        return path

    def temp_directory(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        return temp.name

    def test_shipped_gpu_policy_is_accepted(self):
        path = self.write_gpu_policy(self.shipped_gpu_policy())
        self.assertEqual(config.validate_gpu_policy(path), path)

    def test_gpu_policy_must_constrain_confidential_computing_state(self):
        # An omitted rule accepts that state, so Stage 1 must reject omissions.
        for claim in ("secboot", "dbgstat", "measres", "x-nvidia-gpu-attestation-report-nonce-match"):
            value = self.shipped_gpu_policy()
            del value["required-claims"][claim]
            with self.subTest(claim=claim), self.assertRaises(BuildError):
                config.validate_gpu_policy(self.write_gpu_policy(value))

    def test_gpu_policy_rejects_unsafe_claim_values(self):
        for claim, unsafe in (("secboot", False), ("dbgstat", "enabled"), ("measres", "fail")):
            value = self.shipped_gpu_policy()
            value["required-claims"][claim] = unsafe
            with self.subTest(claim=claim), self.assertRaises(BuildError):
                config.validate_gpu_policy(self.write_gpu_policy(value))

    def test_gpu_policy_rejects_incomplete_nested_certificate_rules(self):
        value = self.shipped_gpu_policy()
        del value["required-claims"]["x-nvidia-gpu-driver-rim-cert-chain"]["x-nvidia-cert-ocsp-status"]
        with self.assertRaises(BuildError):
            config.validate_gpu_policy(self.write_gpu_policy(value))

    def test_gpu_policy_rejects_malformed_or_unreadable_files(self):
        directory = Path(self.temp_directory())
        broken = directory / "broken.json"
        broken.write_text("{not json")
        for path in (broken, directory / "absent.json"):
            with self.subTest(path=path.name), self.assertRaises(BuildError):
                config.validate_gpu_policy(path)
        for mutate in (
            lambda v: v.update(version="2.0"),
            lambda v: v.pop("required-claims"),
            lambda v: v.update({"required-claims": []}),
        ):
            value = self.shipped_gpu_policy()
            mutate(value)
            with self.subTest(mutation=repr(value)[:40]), self.assertRaises(BuildError):
                config.validate_gpu_policy(self.write_gpu_policy(value))


class GuestProvisioningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.payload = self.directory / "payload"
        self.root = self.directory / "root"
        for path in ("inputs", "source/cvm/runtime", "source/services"):
            (self.payload / path).mkdir(parents=True)
        for name in ("kbs-client", "kbs-ca.pem", "as-public.pem", "runtime.json"):
            (self.payload / "inputs" / name).write_text(name)
        (self.payload / "source/cvm/runtime/bootstrap.py").write_text("# measured runtime\n")
        services = self.payload / "source/services"
        for unit in (config.SOURCE / "services").iterdir():
            (services / unit.name).write_bytes(unit.read_bytes())
        (self.payload / "inputs/nftables.conf").write_text("flush ruleset\ntable inet cvm {}\n")
        vendor = self.root / "usr/lib/systemd/system"
        vendor.mkdir(parents=True)
        (vendor / "docker.service").write_text(
            "[Unit]\nRequires=docker.socket\nAfter=network-online.target docker.socket\n"
            "[Service]\nExecStart=/usr/bin/dockerd -H fd:// --containerd=/run/containerd/containerd.sock\n"
        )
        (vendor / "finalrd.service").write_text(
            "[Unit]\nDefaultDependencies=no\n[Service]\nType=oneshot\nRemainAfterExit=yes\n"
            "ExecStart=/bin/true\nExecStop=/usr/bin/finalrd\n"
        )
        for path in (
            "hooks/cvm_verity",
            "scripts/local-top/verity_root",
            "scripts/local-bottom/overlay_root",
            "finalrd/cvm_shutdown.finalrd",
        ):
            location = self.payload / "source/initramfs" / path
            location.parent.mkdir(parents=True, exist_ok=True)
            location.write_text("#!/bin/sh\n")
        self.config = {
            "build_id": "cvm-0123456789abcdef",
            "build_user": "ubuntu",
            "dev_mode": False,
            "gpu": "none",
            "guest_release": "26.04",
            "kernel_version": "7.0.0-31-generic",
            "platform": "intel_tdx",
            "profile_version": "cpu-2026.09",
            "required_system_packages": ["python3=3.14.3-0ubuntu2"],
        }

    def test_configuration_rejects_extra_fields_and_unpinned_packages(self):
        path = self.directory / "config.json"
        path.write_text(json.dumps(self.config))
        self.assertEqual(load_config(path), self.config)
        for key, value in (("unexpected", True), ("required_system_packages", ["python3"])):
            changed = dict(self.config, **{key: value})
            path.write_text(json.dumps(changed))
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_config(path)

    def test_install_files_preserves_the_measured_guest_contract(self):
        install_files(self.config, self.payload, self.root)
        self.assertEqual((self.root / "etc/cvm_build_id").read_text(), self.config["build_id"] + "\n")
        self.assertEqual((self.root / "etc/modules-load.d/cvm.conf").read_text().splitlines()[0], "tdx_guest")
        self.assertTrue((self.root / "usr/lib/cvm/bin/kbs-client").stat().st_mode & stat.S_IXUSR)
        fstab = (self.root / "etc/fstab").read_text()
        self.assertNotIn("/vault ", fstab)
        self.assertNotIn(" swap ", fstab)
        daemon = json.loads((self.root / "etc/docker/daemon.json").read_text())
        self.assertEqual(daemon["data-root"], "/vault/docker/data")
        self.assertFalse(daemon["features"]["containerd-snapshotter"])
        self.assertEqual(
            {path.name for path in (self.root / "usr/lib/systemd/system").glob("cvm_*")},
            {"cvm_bootstrap.service", "cvm_integrity.service", "cvm_app.service"},
        )
        self.assertEqual(
            (self.root / "etc/nftables.conf").read_bytes(), (self.payload / "inputs/nftables.conf").read_bytes()
        )
        self.assertEqual(os.readlink(self.root / "etc/systemd/system/docker.socket"), "/dev/null")
        docker = (self.root / "usr/lib/systemd/system/docker.service").read_text()
        self.assertIn("-H unix:///var/run/docker.sock", docker)
        self.assertNotIn("fd://", docker)
        self.assertNotIn("docker.socket", docker)
        self.assertFalse((self.root / "etc/systemd/system/docker.service.d").exists())

    def test_shutdown_environment_is_prepared_and_checked_before_attestation(self):
        install_files(self.config, self.payload, self.root)
        path = self.root / "usr/lib/systemd/system/finalrd.service"
        unit = path.read_text()
        self.assertIn("ExecStart=/usr/bin/finalrd\n", unit)
        self.assertNotIn("ExecStop=", unit)
        for name in ("libmount.so.1", "libblkid.so.1"):
            self.assertIn(f"ExecStartPost=/usr/bin/test -r /run/initramfs/usr/lib/x86_64-linux-gnu/{name}\n", unit)
        hook = self.root / "etc/finalrd/cvm_shutdown.finalrd"
        self.assertEqual(
            hook.read_bytes(), (self.payload / "source/initramfs/finalrd/cvm_shutdown.finalrd").read_bytes()
        )
        self.assertTrue(hook.stat().st_mode & stat.S_IXUSR)
        install_files(self.config, self.payload, self.root)
        self.assertEqual(path.read_text(), unit)
        path.write_text(unit.replace("ExecStart=/usr/bin/finalrd", "ExecStart=/unsupported"))
        with self.assertRaisesRegex(ValueError, "Unsupported finalrd"):
            install_files(self.config, self.payload, self.root)

    def test_development_layout_keeps_identical_three_unit_files(self):
        self.config["dev_mode"] = True
        install_files(self.config, self.payload, self.root)
        for path in (config.SOURCE / "services").iterdir():
            self.assertEqual((self.root / "usr/lib/systemd/system" / path.name).read_bytes(), path.read_bytes())
        self.assertTrue((self.root / "etc/cvm/dev_mode").is_file())

    def test_gpu_runtime_and_masks_are_explicit(self):
        self.assertNotIn("runtimes", json.loads(docker_configuration("none")))
        self.assertTrue(json.loads(docker_configuration("none"))["no-new-privileges"])
        self.assertEqual(
            json.loads(docker_configuration("nvidia_cc"))["runtimes"]["nvidia"]["path"],
            "nvidia-container-runtime",
        )
        mask(self.root, ("ssh.service", "ssh.socket"))
        for name in ("ssh.service", "ssh.socket"):
            self.assertEqual(os.readlink(self.root / "etc/systemd/system" / name), "/dev/null")

    def test_kernel_hardening_sysctls_and_nts_only_clock_sources_are_installed(self):
        install_files(dict(self.config, time_servers=["time.example.org", "nts.example.net"]), self.payload, self.root)
        sysctl = (self.root / "etc/sysctl.d/99-cvm-hardening.conf").read_text()
        self.assertEqual(sysctl, HARDENING_SYSCTL)
        for setting in (
            "kernel.kexec_load_disabled = 1",
            "kernel.sysrq = 0",
            "kernel.dmesg_restrict = 1",
            "kernel.core_pattern = /dev/null",
            "kernel.unprivileged_bpf_disabled = 1",
        ):
            self.assertIn(setting, sysctl)
        chrony = (self.root / "etc/chrony/chrony.conf").read_text()
        self.assertIn("server time.example.org iburst nts", chrony)
        self.assertIn("server nts.example.net iburst nts", chrony)
        self.assertIn("authselectmode require", chrony)
        self.assertNotIn("pool ", chrony)
        self.assertNotIn("chrony-dhcp", chrony)
        self.assertEqual(chrony, chrony_configuration(["time.example.org", "nts.example.net"]))
        # Without explicit servers the packaged chrony configuration is kept.
        other = self.directory / "other-root"
        vendor = other / "usr/lib/systemd/system"
        vendor.mkdir(parents=True)
        (vendor / "docker.service").write_text((self.root / "usr/lib/systemd/system/docker.service").read_text())
        (vendor / "finalrd.service").write_text((self.root / "usr/lib/systemd/system/finalrd.service").read_text())
        install_files(self.config, self.payload, other)
        self.assertFalse((other / "etc/chrony/chrony.conf").exists())
        for invalid in ([], "time.example.org", ["bad server"], ["a", "a"], [1]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_time_servers(invalid)
        path = self.directory / "config.json"
        path.write_text(json.dumps(dict(self.config, time_servers=["time.example.org"])))
        self.assertEqual(load_config(path)["time_servers"], ["time.example.org"])

    def test_finalize_removes_build_identity_and_construction_sudo(self):
        root = self.directory / "final-root"
        for path in ("etc/sudoers.d", "etc/ssh", "var/log/journal/abc", "var/lib/apt/lists/partial", "var/lib/dbus"):
            (root / path).mkdir(parents=True)
        (root / "etc/sudoers.d/90-cloud-init-users").write_text("# cloud-init\nubuntu ALL=(ALL) NOPASSWD:ALL\n")
        (root / "etc/sudoers.d/README").write_text("# documentation\n")
        (root / "etc/sudoers.d/operator").write_text("operator ALL=(ALL) ALL\n")
        (root / "etc/machine-id").write_text("0123456789abcdef0123456789abcdef\n")
        (root / "var/lib/dbus/machine-id").write_text("0123456789abcdef0123456789abcdef\n")
        (root / "etc/ssh/ssh_host_ed25519_key").write_text("private")
        (root / "etc/ssh/ssh_host_ed25519_key.pub").write_text("public")
        (root / "etc/ssh/sshd_config").write_text("Port 22\n")
        (root / "var/log/journal/abc/system.journal").write_text("construction log")
        (root / "var/lib/apt/lists/partial/index").write_text("lists")
        self.assertEqual(
            sudoers_files_for("ubuntu", root / "etc/sudoers.d"), [root / "etc/sudoers.d/90-cloud-init-users"]
        )
        self.assertEqual(sudoers_files_for("ubuntu", root / "absent"), [])
        scrub_identity(root)
        self.assertEqual((root / "etc/machine-id").read_bytes(), b"")
        self.assertFalse((root / "var/lib/dbus/machine-id").exists())
        self.assertFalse((root / "etc/ssh/ssh_host_ed25519_key").exists())
        self.assertFalse((root / "etc/ssh/ssh_host_ed25519_key.pub").exists())
        self.assertTrue((root / "etc/ssh/sshd_config").exists())
        self.assertEqual(list((root / "var/log/journal").iterdir()), [])
        self.assertEqual(list((root / "var/lib/apt/lists").iterdir()), [])
        self.assertEqual((root / "etc/hostname").read_text(), "cvm\n")

    def test_package_service_suppression_restores_existing_policy(self):
        policy = self.directory / "policy-rc.d"
        policy.write_text("#!/bin/sh\nexit 0\n")
        policy.chmod(0o700)
        with suppress_service_starts(policy):
            self.assertEqual(policy.read_text(), "#!/bin/sh\nexit 101\n")
        self.assertEqual(policy.read_text(), "#!/bin/sh\nexit 0\n")
        self.assertEqual(stat.S_IMODE(policy.stat().st_mode), 0o700)


class ApplicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "cvm").mkdir()
        for name in ("image.tar", "ca", "cert", "key"):
            (self.root / name).write_text("input")
        self.value = {
            "cvm_image": "cvm",
            "docker_archive": "image.tar",
            "image_id": "sha256:" + "a" * 64,
            "requires_gpu": False,
            "container": {},
            "allowed_ports": [8080],
            "allowed_out_ports": [443],
            "applog_drive_size": 1,
            "user_config_drive_size": 1,
            "user_data_drive_size": 1,
            "vault_drive_size": 1,
        }

    def load(self):
        path = self.root / "app.yml"
        path.write_text(yaml.safe_dump(self.value))
        return config.application(path)

    def test_non_nvflare_application(self):
        app = self.load()
        self.assertEqual(app["cvm_image"], str((self.root / "cvm").resolve()))
        self.assertNotIn("deployment_id", app)
        self.assertNotIn("platforms", app)
        self.assertEqual(app["container"]["env"], {})
        self.assertNotIn("entrypoint", app["container"])

    def test_cvm_image_accepts_pinned_registry_references(self):
        reference = "registry.example.org:5000/cvm/cpu@sha256:" + "a" * 64
        for prefix in ("", "oci://", "https://"):
            self.value["cvm_image"] = prefix + reference
            with self.subTest(prefix=prefix):
                self.assertEqual(self.load()["cvm_image"], reference)
        for value in (None, "https://registry.example.org/cvm/cpu:latest", "oci://" + reference[:-1], "image.tar"):
            self.value["cvm_image"] = value
            with self.subTest(value=value), self.assertRaises(BuildError):
                self.load()

    def test_platform_filter_is_optional_but_must_be_valid_when_present(self):
        self.assertNotIn("platforms", self.load())
        self.value["platforms"] = ["intel_tdx"]
        self.assertEqual(self.load()["platforms"], ["intel_tdx"])
        for value in (None, [], "intel_tdx", ["custom"], ["intel_tdx", "intel_tdx"], [{}]):
            self.value["platforms"] = value
            with self.subTest(value=value), self.assertRaises(BuildError):
                self.load()

    def test_removed_vault_input_fields_are_rejected(self):
        for key, value in (("deployment_id", "generic-app"), ("cvm_profile", "profile_set.json"), ("trustee", {})):
            self.value[key] = value
            with self.subTest(key=key), self.assertRaises(BuildError):
                self.load()
            del self.value[key]

    def test_clear_input_directories_are_accepted(self):
        for name in ("user_config", "user_data"):
            source = self.root / name
            source.mkdir()
            (source / "input.yml").write_text("value: public\n")
            self.value[name] = name
        app = self.load()
        self.assertEqual(app["user_config"], str(self.root / "user_config"))
        self.assertEqual(app["user_data"], str(self.root / "user_data"))

    def test_runtime_has_no_build_paths_or_admin_credentials(self):
        self.value["container"]["env"] = {"APP_SECRET": "authenticated-only"}
        app = self.load()
        app["trustee"] = {"url": "https://trustee.test", "admin_token_file": str(self.root / "token.jwt")}
        projected = runtime_config(app)
        self.assertNotIn("trustee", projected)
        self.assertNotIn("docker_archive", projected)
        self.assertNotIn(str(self.root), json.dumps(projected))
        self.assertEqual(projected["container"]["env"]["APP_SECRET"], "authenticated-only")

    def test_provisioning_schema_is_external(self):
        for name in ("cc_params", "startup_kit", "role", "project", "nvflare_version"):
            self.value[name] = "external"
            with self.subTest(name=name), self.assertRaises(BuildError):
                self.load()
            del self.value[name]

    def test_clear_sidecar_rejects_private_keys(self):
        clear = self.root / "public-sidecar"
        clear.mkdir()
        for field in ("user_config", "user_data"):
            self.value[field] = str(clear)
            for name, content in (
                ("server.key", "opaque"),
                ("renamed.txt", "-----BEGIN PRIVATE KEY-----\nopaque"),
                ("identity.p12", "opaque"),
            ):
                path = clear / name
                path.write_text(content)
                with self.subTest(field=field, name=name), self.assertRaises(BuildError):
                    self.load()
                path.unlink()
            del self.value[field]
        (clear / "ext_mount.conf").write_text("files.example:/public-data\n")
        self.value["user_data"] = str(clear)
        self.assertEqual(self.load()["user_data"], str(clear))

    def test_arbitrary_entrypoint_argument_array(self):
        self.value["container"].update(entrypoint=["python3", "-m", "http.server"], command=["8080"])
        app = self.load()
        command = runtime.docker_argv(app, defaults={"Cmd": ["ignored"]})
        self.assertEqual(command[-4:], [app["image_id"], "-m", "http.server", "8080"])

    def test_default_image_entrypoint_is_preserved(self):
        app = self.load()
        command = runtime.docker_argv(app)
        self.assertNotIn("--entrypoint", command)
        self.assertEqual(command[-1], app["image_id"])

    def test_container_confinement_defaults_and_address_allowlists(self):
        app = self.load()
        container = app["container"]
        self.assertEqual(container["capabilities"], list(runtime.DEFAULT_CAPABILITIES))
        self.assertEqual(container["pids_limit"], runtime.DEFAULT_PIDS_LIMIT)
        self.assertFalse(container["host_bin"])
        self.assertFalse(container["read_only_rootfs"])
        self.assertNotIn("allowed_out_cidrs", runtime_config(app))
        self.value["allowed_out_cidrs"] = ["10.0.0.0/8"]
        self.value["allowed_in_cidrs"] = ["192.0.2.0/24"]
        self.value["container"].update(capabilities=["NET_BIND_SERVICE"], pids_limit=128, host_bin=True)
        projected = runtime_config(self.load())
        self.assertEqual(projected["allowed_out_cidrs"], ["10.0.0.0/8"])
        self.assertEqual(projected["allowed_in_cidrs"], ["192.0.2.0/24"])
        self.assertEqual(projected["container"]["capabilities"], ["NET_BIND_SERVICE"])
        for key, invalid in (
            ("capabilities", ["SYS_ADMIN"]),
            ("capabilities", ["CHOWN", "CHOWN"]),
            ("capabilities", "CHOWN"),
            ("pids_limit", 0),
            ("pids_limit", "many"),
            ("host_bin", "yes"),
            ("read_only_rootfs", 1),
        ):
            original = self.value["container"].get(key)
            self.value["container"][key] = invalid
            with self.subTest(key=key, invalid=invalid), self.assertRaises(BuildError):
                self.load()
            self.value["container"][key] = original
        for invalid in (["10.0.0.1/8"], "10.0.0.0/8", ["10.0.0.0/8", "10.0.0.0/8"]):
            self.value["allowed_out_cidrs"] = invalid
            with self.subTest(cidrs=invalid), self.assertRaises(BuildError):
                self.load()

    def test_entrypoint_override_preserves_default_command(self):
        self.value["container"]["entrypoint"] = ["/app/run", "--safe"]
        app = self.load()
        self.assertEqual(
            runtime.docker_argv(app, defaults={"Cmd": ["serve"]})[-3:], [app["image_id"], "--safe", "serve"]
        )

    def test_tee_opt_in_is_resolved_by_runtime(self):
        self.value["container"]["tee_device"] = True
        app = self.load()
        with self.assertRaises(BuildError):
            runtime.docker_argv(app)
        self.assertIn("/dev/tdx_guest", runtime.docker_argv(app, device="/dev/tdx_guest"))

    def test_gpu_application_passes_all_gpus_to_container(self):
        self.value["requires_gpu"] = True
        app = self.load()
        command = runtime.docker_argv(app)
        self.assertIn("--gpus", command)
        self.assertEqual(command[command.index("--gpus") + 1], "all")

    def test_invalid_ports_and_writable_sidecar_denied(self):
        self.value["container"]["ports"] = [{"host": 9090, "container": 80}]
        with self.assertRaises(BuildError):
            self.load()
        self.value["container"]["ports"] = []
        self.value["container"]["volumes"] = [{"source": "/user_data/input", "target": "/input", "read_only": False}]
        with self.assertRaises(BuildError):
            self.load()

    def test_duplicate_yaml_keys_denied(self):
        path = self.root / "duplicate.yml"
        path.write_text("a: 1\na: 2\n")
        with self.assertRaises(BuildError):
            config.load_yaml(path)

    def test_service_cannot_override_bootstrap(self):
        good = "[Unit]\nDescription=Application\n[Service]\nExecStart=/vault/application/start\n"
        validate_service("app_helper.service", good)
        for name, text in (
            ("cvm_bootstrap.service", good),
            ("app_x.service", good + "ExecStartPre=/bin/true\n"),
            ("app_x.service", good.replace("Description=Application", "Requires=cvm_bootstrap.service")),
            ("app_x.service", good + "Environment=TEE_DEVICE=/dev/tdx_guest\n"),
        ):
            with self.assertRaises((BuildError, ValueError)):
                validate_service(name, text)


class TokenTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.key = ec.generate_private_key(ec.SECP256R1())
        path = Path(self.temp.name) / "public.pem"
        path.write_bytes(
            self.key.public_key().public_bytes(
                serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
            )
        )
        self.config = {
            "token_algorithm": "ES256",
            "token_issuer": "test-as",
            "as_public_key": str(path),
            "attestation_policy_id": "cvm-test",
            "platform": "intel_tdx",
        }
        self.claims = {
            "iat": 1000,
            "exp": 1300,
            "iss": "test-as",
            "submods": {
                "cpu0": {
                    "ear.appraisal-policy-id": "cvm-test",
                    "ear.status": "affirming",
                    "ear.trustworthiness-vector": {"executables": 3, "hardware": 2, "configuration": 2},
                    "ear.veraison.annotated-evidence": {
                        "init_data": "0" * 96,
                        "tdx": {"td_attributes": {"debug": False}},
                    },
                }
            },
        }

    def token(self):
        def encode(value):
            return base64.urlsafe_b64encode(value).rstrip(b"=")

        body = encode(canonical({"alg": "ES256"})) + b"." + encode(canonical(self.claims))
        r, s = utils.decode_dss_signature(self.key.sign(body, ec.ECDSA(hashes.SHA256())))
        return body + b"." + encode(r.to_bytes(32, "big") + s.to_bytes(32, "big"))

    def test_positive_signed_appraisal(self):
        validate_token(self.token(), self.config, bytes(32), now=1001)

    def test_genuine_trustee_snp_token_signature_and_binding(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/snp_trustee_v022.json").read_text())
        public = Path(self.temp.name) / "upstream-as.pem"
        public.write_text(fixture["as_public_key"])
        config = dict(
            self.config,
            as_public_key=str(public),
            token_issuer="CoCo-Attestation-Service",
            platform="amd_sev_snp",
            attestation_policy_id="default",
        )
        token = fixture["token"].encode()
        now = fixture["issued_at"] + 1
        claims = validate_token(token, config, bytes(32), now=now)
        self.assertEqual(claims["submods"]["cpu0"]["ear.status"], "affirming")
        with self.assertRaisesRegex(BuildError, "binding mismatch"):
            validate_token(token, config, bytes([1]) * 32, now=now)

    def test_upstream_snp_hex_binding_and_legacy_encoding_rejected(self):
        digest = bytes.fromhex("fbff" * 16)
        self.config["platform"] = "amd_sev_snp"
        evidence = self.claims["submods"]["cpu0"]["ear.veraison.annotated-evidence"]
        evidence.clear()
        evidence.update(init_data=digest.hex(), snp={"policy_debug_allowed": False, "policy_migrate_ma": False})
        validate_token(self.token(), self.config, digest, now=1001)
        for value in (digest.hex().upper(), digest.hex()[:-1], base64.b64encode(digest).decode()):
            with self.subTest(value=value), self.assertRaises(BuildError):
                evidence["init_data"] = value
                validate_token(self.token(), self.config, digest, now=1001)

    def test_negative_policy_and_trust_vector_denied(self):
        cpu = self.claims["submods"]["cpu0"]
        for field, value in (
            ("ear.status", "contraindicated"),
            ("ear.appraisal-policy-id", "default"),
            ("ear.trustworthiness-vector", {"executables": 3, "hardware": 2}),
        ):
            original = cpu[field]
            cpu[field] = value
            with self.assertRaises(BuildError):
                validate_token(self.token(), self.config, bytes(32), now=1001)
            cpu[field] = original

    def test_wrong_binding_even_with_valid_signature(self):
        with self.assertRaises(BuildError):
            validate_token(self.token(), self.config, bytes([1]) * 32, now=1001)

    def test_expired_and_stale_appraisal(self):
        for now in (1300, 2000, 900):
            with self.assertRaises(BuildError):
                validate_token(self.token(), self.config, bytes(32), now=now)

    def test_tampered_signature(self):
        token = self.token()
        parts = token.split(b".")
        parts[-1] = b"A" * len(parts[-1])
        with self.assertRaises(BuildError):
            validate_token(b".".join(parts), self.config, bytes(32), now=1001)


class RuntimeContractTests(unittest.TestCase):
    def test_clock_service_can_rekey_nts_before_and_after_unlock(self):
        for inbound, outbound in (([], [443, 8443]), ([8080], [443, 8443])):
            with patch.object(runtime, "run") as apply:
                runtime.firewall(inbound, outbound)
            rules = apply.call_args.kwargs["input"].decode()
            self.assertTrue(rules.startswith("table inet cvm {}\ndelete table inet cvm\n"))
            output = rules.split("chain output", 1)[1].split("chain forward", 1)[0]
            self.assertIn("tcp dport 4460 accept", output)
            self.assertIn("udp dport 123 accept", output)
            self.assertIn("udp dport { 67, 547 } accept", output)
            self.assertIn("udp dport 53 accept", output)
            self.assertNotIn("4460", rules.split("chain forward", 1)[1])

    def test_firewall_restricts_dns_and_allowlisted_ports_to_addresses(self):
        rules = firewall_rules(
            [8080],
            [443],
            [{"host": 8080, "container": 80}],
            inbound_sources=["10.0.0.0/8"],
            outbound_destinations=["192.0.2.0/24", "2001:db8::/32"],
            resolvers=["10.0.0.53", "2001:db8::53"],
        )
        chains = {
            name: rules.split("chain " + name, 1)[1].split("chain", 1)[0] for name in ("input", "output", "forward")
        }
        self.assertIn("ip saddr { 10.0.0.0/8 } tcp dport { 8080 } accept", chains["input"])
        self.assertNotIn("\ntcp dport { 8080 } accept", chains["input"])
        self.assertIn("ip daddr { 10.0.0.53/32 } udp dport 53 accept", chains["output"])
        self.assertIn("ip6 daddr { 2001:db8::53/128 } tcp dport 53 accept", chains["output"])
        self.assertIn("ip daddr { 192.0.2.0/24 } tcp dport { 443 } accept", chains["output"])
        self.assertIn("ip6 daddr { 2001:db8::/32 } tcp dport { 443 } accept", chains["output"])
        self.assertIn('iifname "docker0" ip daddr { 192.0.2.0/24 } tcp dport { 443 } accept', chains["forward"])
        self.assertIn('iifname "docker0" ip daddr { 10.0.0.53/32 } udp dport 53 accept', chains["forward"])
        self.assertIn(
            'oifname "docker0" ip saddr { 10.0.0.0/8 } tcp dport 80 ct original proto-dst 8080 accept',
            chains["forward"],
        )
        self.assertNotIn('oifname "docker0" tcp dport', chains["forward"])
        for chain in chains.values():
            self.assertIn("ct state invalid drop", chain)
            self.assertNotIn("ip protocol icmp accept", chain)
        self.assertIn("icmp type { echo-request, echo-reply", chains["output"])
        # Without allowlists the ports stay open to any address, as before.
        open_rules = firewall_rules([8080], [443])
        self.assertIn("\ntcp dport { 8080 } accept", open_rules)
        self.assertIn("\ntcp dport { 443 } accept", open_rules)
        self.assertIn("\nudp dport 53 accept", open_rules)
        for invalid in (["10.0.0.1/8"], ["10.0.0.0/8", "10.0.0.0/8"], ["not-a-cidr"], "10.0.0.0/8"):
            with self.subTest(invalid=invalid), self.assertRaises(BuildError):
                firewall_rules([], [443], outbound_destinations=invalid)
        for invalid in (["127.0.0.1"], ["0.0.0.0"], ["dns"]):
            with self.subTest(resolver=invalid), self.assertRaises(BuildError):
                firewall_rules([], [443], resolvers=invalid)

    def test_published_ports_apply_both_source_address_families(self):
        for sources in ([], ["10.0.0.0/8"], ["2001:db8::/32"], ["10.0.0.0/8", "2001:db8::/32"]):
            with self.subTest(sources=sources):
                rules = firewall_rules([8080], [], [{"host": 8080, "container": 80}], inbound_sources=sources)
                accepts = [line for line in rules.splitlines() if line.startswith('oifname "docker0"')]
                expected = []
                for cidr in sources:
                    family = "ip6" if ":" in cidr else "ip"
                    expected.append(
                        f'oifname "docker0" {family} saddr {{ {cidr} }} '
                        "tcp dport 80 ct original proto-dst 8080 accept"
                    )
                self.assertEqual(
                    accepts, expected or ['oifname "docker0" tcp dport 80 ct original proto-dst 8080 accept']
                )

    def test_measured_command_line_locks_down_the_guest_kernel(self):
        tokens = kernel_command_line("ab" * 32, 1024, 4096).split()
        for token in (
            "lockdown=integrity",
            "module.sig_enforce=1",
            "loglevel=3",
            "printk.console_no_auto_verbose=1",
            "panic=1",
            "oops=panic",
            "systemd.verity=no",
        ):
            self.assertIn(token, tokens)
        self.assertEqual(tokens.count("roothash=" + "ab" * 32), 1)

    def test_vault_metadata_requires_the_pinned_deterministic_kdf(self):
        metadata = {
            "segments": {
                "0": {
                    "type": "crypt",
                    "encryption": "aes-xts-random",
                    "offset": str(HEADER_BYTES),
                    "sector_size": 512,
                    "integrity": {"type": "hmac(sha256)"},
                }
            },
            "config": {"json_size": "12288", "keyslots_size": str(HEADER_BYTES - 32768)},
            "keyslots": {
                "0": {
                    "type": "luks2",
                    "key_size": 96,
                    "kdf": {"type": "pbkdf2", "hash": "sha256", "iterations": 1000, "salt": "x"},
                    "area": {"offset": "32768", "size": "258048"},
                }
            },
        }
        validate_luks_metadata(metadata)
        for kdf in (
            {"type": "argon2id", "time": 4, "memory": 1048576, "cpus": 4, "salt": "x"},
            {"type": "pbkdf2", "hash": "sha512", "iterations": 1000, "salt": "x"},
            {"type": "pbkdf2", "hash": "sha256", "iterations": 999, "salt": "x"},
            {},
        ):
            metadata["keyslots"]["0"]["kdf"] = kdf
            with self.subTest(kdf=kdf), self.assertRaisesRegex(BuildError, "KDF"):
                validate_luks_metadata(metadata)

    def test_activated_vault_key_must_live_in_the_kernel_keyring(self):
        crypt = "0 100 crypt capi:authenc(hmac(sha256),xts(aes))-random {key} 0 253:0 0 1 integrity:48:aead"
        integrity = "0 100 integrity 253:1 0 48 J 0"
        for key, accepted in (
            (":96:logon:cryptsetup:uuid-d0", True),
            (":96:user:cryptsetup:uuid-d0", False),
            ("0" * 192, False),
            ("deadbeef", False),
        ):
            outputs = [crypt.format(key=key).encode(), integrity.encode()]
            with self.subTest(key=key), patch("cvm.common.luks.run", side_effect=outputs):
                if accepted:
                    self.assertEqual(validate_mapping("vault"), "253:0")
                else:
                    with self.assertRaisesRegex(BuildError, "keyring"):
                        validate_mapping("vault")

    def test_guest_requires_the_measured_gpu_count(self):
        with patch("cvm.runtime.gpu.run", return_value=b"0000:41:00.0\n0000:43:00.0\n"):
            gpu.readiness({"gpu": "nvidia_cc", "gpu_count": 2}, True)
            with self.assertRaises(BuildError):
                gpu.readiness({"gpu": "nvidia_cc", "gpu_count": 1}, True)

    def test_attestation_commands_share_one_hard_deadline(self):
        config = {
            "kbs_client": "/test/kbs-client",
            "kbs_url": "https://kbs.test",
            "kbs_cert": "/test/ca.pem",
            "build_id": "bundle-1",
            "platform": "intel_tdx",
        }
        with (
            patch("cvm.runtime.attestation.run", side_effect=[b"token", base64.b64encode(bytes(64))]) as execute,
            patch("cvm.runtime.attestation.validate_token"),
            patch("cvm.runtime.attestation.time.monotonic", side_effect=[100, 110, 140]),
        ):
            with authorized_key(config, bytes(32), budget=ATTESTATION_BUDGET_SECONDS):
                pass
        self.assertEqual([round(call.kwargs["timeout"]) for call in execute.call_args_list], [50, 20])
        self.assertEqual(
            [call.kwargs["operation"] for call in execute.call_args_list],
            ["KBS quote/appraisal", "KBS resource retrieval/decryption"],
        )

    def test_clock_gate_uses_bounded_chrony_correction(self):
        with patch("cvm.runtime.bootstrap.run") as execute:
            runtime.time_sync(max_tries=5)
        self.assertEqual(execute.call_args.kwargs["timeout"], 7)
        self.assertEqual(execute.call_args.args[0], ["/usr/bin/chronyc", "waitsync", "5", "0.5", "1000", "1"])

    def test_cold_clock_collects_samples_before_enforcing_final_quality_gate(self):
        with patch("cvm.runtime.bootstrap.run") as execute:
            runtime.time_sync(max_tries=90, initialize=True)
        self.assertEqual(
            [call.args[0] for call in execute.call_args_list],
            [
                ["/usr/bin/chronyc", "maxupdateskew", "1000"],
                ["/usr/bin/chronyc", "waitsync", "30", "0.5", "0", "1"],
                ["/usr/bin/chronyc", "burst", "8/16"],
                ["/usr/bin/chronyc", "waitsync", "90", "0.5", "1000", "1"],
            ],
        )
        self.assertEqual([call.kwargs["timeout"] for call in execute.call_args_list], [2, 32, 2, 92])

    def test_cold_clock_rejects_unsynchronized_or_excessive_skew(self):
        for failure_at in (1, 3):
            with patch("cvm.runtime.bootstrap.run", side_effect=[None] * failure_at + [BuildError("clock not ready")]):
                with self.assertRaises(BuildError):
                    runtime.time_sync(max_tries=90, initialize=True)

    def test_monitor_status_is_fail_closed(self):
        healthy_status("0 123456 integrity 0 123456 0")
        for value in ("0 123456 integrity 1 123456 0", "0 123456 error", "0 123456 integrity unknown 123456 0"):
            with self.assertRaises(BuildError):
                healthy_status(value)

    def test_policy_readback_must_be_exact_and_canonical(self):
        expected = b"package policy\ndefault allow = false\n"
        verify_readback(expected, expected)
        for returned in (b'[{"id":"resource-policy"}]', b"AAAA", b"YWJj=", b"YWJj\n"):
            with self.assertRaises(BuildError):
                verify_readback(expected, returned)

    def test_qemu_keeps_boot_inputs_and_binding_separate(self):
        for platform in platforms.PLATFORMS:
            manifest = {
                "platform": platform,
                "launch_shape": {
                    "cpu_model": "host",
                    "vcpus": 4,
                    "memory_gib": 8,
                    "quote_generation": {"type": "vsock", "cid": 2, "port": 4050},
                    "snp_policy": 0x30000,
                },
                "cmdline": "immutable cmdline",
                "contract": {"gpu": "none"},
            }
            args = qemu_command(manifest, "/bundle", ["root", "logs", "config", "data", "vault"], bytes(32), cbit=51)
            self.assertEqual(args[args.index("-append") + 1], "immutable cmdline")
            objects = [json.loads(args[i + 1]) for i, value in enumerate(args) if value == "-object"]
            tee = next(item for item in objects if item["id"] == "tee0")
            self.assertEqual(
                len(base64.b64decode(tee["mrconfigid" if platform == "intel_tdx" else "host-data"])),
                48 if platform == "intel_tdx" else 32,
            )
            files = [json.loads(args[i + 1]) for i, value in enumerate(args) if value == "-blockdev"]
            self.assertTrue(all(item["locking"] == "on" for item in files if item["driver"] == "file"))
            for layer in ("file", "qcow2"):
                self.assertEqual(
                    [item["read-only"] for item in files if item["driver"] == layer],
                    [True, False, True, True, False],
                )
            disks = [dict(pair.split("=", 1) for pair in x.split(",")[1:]) for x in args if x.startswith("scsi-hd,")]
            self.assertEqual([d["serial"] for d in disks], ["cvm-" + role for role in DISK_ROLES])
            for role in DISK_ROLES:
                self.assertEqual(Path(disk_device(role)).name, "scsi-0QEMU_QEMU_HARDDISK_cvm-" + role)


if __name__ == "__main__":
    unittest.main(verbosity=2)
