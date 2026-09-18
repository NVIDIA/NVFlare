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

"""Opt-in acceptance on real TEE hardware with a test vault containing the lab agent.

Set CVM_HARDWARE_TESTS=1, CVM_BUNDLE and CVM_VAULT to detached test artifacts.
Every mutation is applied to an independent file copy. Requires root and free
18080/18081 ports. Evidence logs are retained in CVM_HARDWARE_OUTPUT.
"""

import contextlib
import json
import os
import shutil
import signal
import socket
import subprocess
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from cvm.artifacts.bundle import verify_bundle
from cvm.build.config import contains_private_key
from cvm.build.storage import mounted, nbd
from cvm.common.errors import require
from cvm.common.io import digest_file, read_json, write_json
from cvm.common.linux import run
from cvm.common.luks import snapshot_header


@unittest.skipUnless(os.environ.get("CVM_HARDWARE_TESTS") == "1", "Opt-in real TEE acceptance")
class HardwareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = Path(os.environ["CVM_BUNDLE"]).resolve()
        cls.original = Path(os.environ["CVM_VAULT"]).resolve()
        cls.output = Path(os.environ["CVM_HARDWARE_OUTPUT"]).resolve()
        cls.output.mkdir(parents=True, exist_ok=True)
        cls.manifest = verify_bundle(cls.bundle)
        require(
            os.geteuid() == 0 and cls.manifest["profile_version"].startswith("test-"),
            "Hardware fixtures require a test bundle and root",
        )
        cls.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        import socket

        for port in (18080, 18081):
            with socket.socket() as probe:
                # Match QEMU's listener behavior after a preceding test leaves
                # closed connections in TIME_WAIT; a live listener still fails.
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", port))

    def setUp(self):
        self.directory = self.output / self._testMethodName
        self.directory.mkdir()
        self.vault = self.directory / "delivery"
        # Separate writable files; no backing image or shared attachment.
        run(["cp", "-a", "--reflink=auto", self.original, self.vault])
        self.process = None
        self.logs = []
        self.disposables = []
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.stop()
        # Retain serial evidence and result metadata, not multi-GiB disposable images.
        shutil.rmtree(self.vault)
        for path in self.disposables:
            shutil.rmtree(path, ignore_errors=True)

    def stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
            self.process.wait(timeout=45)
            self.process = None

    def boot(self, *, reverse_scsi=False, bundle=None):
        bundle = Path(bundle or self.bundle)
        self.boot_started = time.monotonic()
        log = self.directory / f"boot-{len(self.logs)}.log"
        self.logs.append(log)
        entry = ["-m", "cvm.host.launcher"]
        if reverse_scsi:
            # Keep the normal launcher checks and locking, but emulate a host
            # presenting the five disks at different SCSI target addresses.
            entry = [
                "-c",
                "import re; from builder import launcher; original=launcher.qemu_command; "
                "launcher.qemu_command=lambda *a,**k: ["
                "re.sub(r'scsi-id=(\\d)', lambda m: 'scsi-id='+str(4-int(m[1])), x) "
                "if x.startswith('scsi-hd,') else x for x in original(*a,**k)]; launcher.main()",
            ]
        command = [str(self.vault / "launch_cvm.sh"), "--cvm-bundle", str(bundle)]
        if reverse_scsi:
            command = ["python3", *entry, "--cvm-bundle", str(bundle), "--vault-directory", str(self.vault)]
        with log.open("wb") as output:
            self.process = subprocess.Popen(
                command,
                stdout=output,
                stderr=subprocess.STDOUT,
            )

    def request(self, path="/state", method="GET", port=18081):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}" + path, method=method, data=b"" if method == "POST" else None
        )
        with self.http.open(req, timeout=180 if method == "POST" else 3) as response:
            return response.read()

    def ready(self):
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            self.assertIsNone(self.process.poll(), "Guest exited before workload readiness; inspect serial evidence")
            try:
                state = json.loads(self.request())
                if state["marker"] and self.request("/", port=18080) == b"CVM_GENERIC_APPLICATION_OK\n":
                    self.assertEqual(state["cvm_bootstrap.service"], "active")
                    self.assertEqual(state["cvm_integrity.service"], "active")
                    self.assertEqual(
                        set(state["cvm_units"]),
                        {"cvm_bootstrap.service", "cvm_integrity.service", "cvm_app.service"},
                    )
                    self.assertEqual(state["docker_socket"], "masked")
                    self.assertEqual(state["nftables_enabled"], "enabled")
                    self.assertTrue(state["firewall_present"])
                    self.assertEqual(state["platform"], self.manifest["platform"])
                    self.assertTrue(state["core_dumps_disabled"])
                    self.assertTrue(state["sidecar_roles_correct"])
                    self.assertTrue(state["input_mounts_read_only"])
                    self.assertTrue(state["applog_writable"])
                    self.assertTrue(state["clock_synchronized"])
                    self.assertFalse(state["ssh_port_listening"])
                    self.assertEqual(
                        state["ssh_units"],
                        {
                            "ssh.service": {"active": "inactive", "enabled": "masked"},
                            "ssh.socket": {"active": "inactive", "enabled": "masked"},
                        },
                    )
                    state["readiness_seconds"] = round(time.monotonic() - self.boot_started, 3)
                    return state
            except (OSError, ValueError, urllib.error.URLError):
                pass
            time.sleep(1)
        self.fail("Timed out waiting for generic workload")

    def qemu_pid(self):
        children = Path(f"/proc/{self.process.pid}/task/{self.process.pid}/children").read_text().split()
        found = [int(pid) for pid in children if b"qemu-system" in Path(f"/proc/{pid}/cmdline").read_bytes()]
        self.assertEqual(len(found), 1)
        return found[0]

    def result(self, **values):
        write_json(
            self.directory / "result.json",
            dict(
                values,
                platform=self.manifest["platform"],
                manifest_sha256=digest_file(self.bundle / "cvm_manifest.json"),
                logs={p.name: digest_file(p) for p in self.logs},
            ),
        )

    @contextlib.contextmanager
    def drop_egress(self, port, addresses=()):
        table = "cvm_acceptance_" + str(os.getpid())
        run(["nft", "add", "table", "inet", table])
        try:
            run(
                [
                    "nft",
                    "add",
                    "chain",
                    "inet",
                    table,
                    "output",
                    "{",
                    "type",
                    "filter",
                    "hook",
                    "output",
                    "priority",
                    "-200",
                    ";",
                    "policy",
                    "accept",
                    ";",
                    "}",
                ]
            )
            if addresses:
                for address in addresses:
                    family = "ip6" if ":" in address else "ip"
                    run(
                        [
                            "nft",
                            "add",
                            "rule",
                            "inet",
                            table,
                            "output",
                            family,
                            "daddr",
                            address,
                            "tcp",
                            "dport",
                            str(port),
                            "drop",
                        ]
                    )
            else:
                run(["nft", "add", "rule", "inet", table, "output", "tcp", "dport", str(port), "drop"])
            yield
        finally:
            subprocess.run(
                ["nft", "delete", "table", "inet", table],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

    def test_generic_app_binding_and_exclusive_attachment(self):
        self.boot()
        state = self.ready()
        self.assertEqual(state["measurements"], self.manifest["measurements"])
        actual = bytes.fromhex(state["binding"])
        expected = bytes.fromhex(read_json(self.vault / "vault_manifest.json")["vault_bind"])
        self.assertEqual(actual, expected + (bytes(16) if len(actual) == 48 else b""))
        self.assertEqual(state["root_overlay_bytes"], self.manifest["contract"]["root_overlay_max_mib"] * 1024 * 1024)
        other = subprocess.run(
            [
                "python3",
                "-m",
                "cvm.host.launcher",
                "--cvm-bundle",
                str(self.bundle),
                "--vault-directory",
                str(self.vault),
            ],
            capture_output=True,
            timeout=30,
        )
        self.assertNotEqual(other.returncode, 0)
        self.assertIn(b"Vault is already attached", other.stderr)
        self.stop()
        # Replaying the clear sidecar's journal is required after QEMU exits;
        # ro,noload inspection intentionally hides recently committed metadata.
        with nbd(self.vault / "applog.qcow2") as device, mounted(device) as root:
            self.assertEqual((root / "acceptance-output").read_bytes(), b"CVM_APPLOG_WRITE_OK\n")
            container_gpu_attestation = self.manifest["contract"]["gpu"] == "nvidia_cc"
            if container_gpu_attestation:
                result = read_json(root / "container-gpu-attestation.json")
                self.assertEqual(
                    {key: result[key] for key in ("passed", "gpu_count", "nonce_bound", "signed_evidence")},
                    {
                        "passed": True,
                        "gpu_count": self.manifest["contract"]["gpu_count"],
                        "nonce_bound": True,
                        "signed_evidence": True,
                    },
                )
                self.assertRegex(result["nvat_source_commit"], r"^[0-9a-f]{40}$")
                self.assertRegex(result["evidence_sha256"], r"^[0-9a-f]{64}$")
        self.result(
            generic_application=True,
            binding=True,
            exclusive_attachment=True,
            readiness_seconds=state["readiness_seconds"],
            vault_bytes=state["vault_bytes"],
            vault_available_bytes=state["vault_available_bytes"],
            clock_synchronized_before_attestation=state["clock_synchronized"],
            ssh_service_and_socket_disabled=not state["ssh_port_listening"],
            root_overlay_capacity=True,
            read_only_input_disks=state["input_mounts_read_only"],
            writable_applog=state["applog_writable"],
            container_gpu_attestation=container_gpu_attestation,
        )

    def test_clear_sidecars_contain_no_private_keys(self):
        inspected = []
        for name in ("applog", "user_config", "user_data"):
            with nbd(self.vault / f"{name}.qcow2", readonly=True) as device, mounted(device, readonly=True) as root:
                for item in root.rglob("*"):
                    if not item.is_file():
                        continue
                    lowered = item.name.lower()
                    self.assertFalse(lowered.endswith((".key", ".p12", ".pfx", ".jks")))
                    self.assertFalse(contains_private_key(item))
                    inspected.append(f"{name}/{item.relative_to(root)}")
        self.result(clear_sidecar_scan=True, inspected_paths=inspected)

    def test_root_disk_corruption_prevents_startup(self):
        damaged = self.directory / "root-disk-corruption-bundle"
        self.disposables.append(damaged)
        run(["cp", "-a", "--reflink=auto", self.bundle, damaged])
        root = damaged / "verity_root.qcow2"
        run(["qemu-io", "-f", "qcow2", "-c", "write -P 0x55 0 4096", root])
        manifest_path = damaged / "cvm_manifest.json"
        manifest = read_json(manifest_path)
        self.assertEqual(manifest["measurements"], self.manifest["measurements"])
        manifest["sha256"]["verity_root.qcow2"] = digest_file(root)
        write_json(manifest_path, manifest, mode=0o644)
        verify_bundle(damaged)
        self.boot(bundle=damaged)
        self.process.wait(timeout=120)
        log = self.logs[-1].read_text(errors="replace")
        self.assertNotIn("CVM_WORKLOAD_STARTED", log)
        self.result(root_disk_corruption_prevented_startup=True, launch_measurements_unchanged=True)

    @unittest.skipUnless(os.environ.get("CVM_NETWORK_FAULTS") == "1", "Opt in to isolated host firewall faults")
    def test_attestation_drop_powers_off_within_deadline(self):
        self.boot()
        self.ready()
        kbs = urlparse(self.manifest["contract"]["kbs_url"])
        started = time.monotonic()
        with self.drop_egress(kbs.port or 443):
            self.request("/periodic", "POST")
            self.process.wait(timeout=110)
        elapsed = time.monotonic() - started
        self.assertLessEqual(elapsed, 105)
        self.assertIn("Power down", self.logs[-1].read_text(errors="replace"))
        self.result(attestation_drop_deadline=True, fail_closed_seconds=round(elapsed, 3))

    @unittest.skipUnless(os.environ.get("CVM_NETWORK_FAULTS") == "1", "Opt in to isolated host firewall faults")
    def test_snp_offline_collateral_works_without_kds(self):
        if self.manifest["platform"] != "amd_sev_snp":
            self.skipTest("SNP-only offline collateral acceptance")
        self.boot()
        self.ready()  # The operator preinstalls the upstream SNP offline certificate store.
        before = json.loads(self.request())["periodic"]["sequence"]
        kds_addresses = sorted({item[4][0] for item in socket.getaddrinfo("kdsintf.amd.com", 443)})
        self.assertTrue(kds_addresses, "AMD KDS did not resolve")
        with self.drop_egress(443, kds_addresses):
            self.request("/periodic", "POST")
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                self.assertIsNone(
                    self.process.poll(), "Guest powered off when AMD KDS was unavailable with offline collateral"
                )
                status = json.loads(self.request())["periodic"]
                if status["sequence"] != before and status["result"] == "success":
                    break
                time.sleep(0.5)
            else:
                self.fail("Offline SNP appraisal did not finish within 45 seconds")
        self.stop()
        self.result(snp_collateral_availability=True, amd_kds_blocked=True)

    @unittest.skipUnless(os.environ.get("CVM_GPU_HARDWARE_TESTS") == "1", "Opt in on an NVIDIA CC GPU host")
    @unittest.skipUnless(
        os.environ.get("CVM_BACKEND_LOCAL") == "1", "NRAS fault injection requires an isolated local KBS"
    )
    def test_periodic_gpu_denial_powers_off(self):
        self.assertEqual(self.manifest["contract"]["gpu"], "nvidia_cc")
        self.boot()
        self.ready()
        endpoint = urlparse(self.manifest["contract"]["gpu_attestation_url"])
        addresses = sorted({item[4][0] for item in socket.getaddrinfo(endpoint.hostname, endpoint.port or 443)})
        started = time.monotonic()
        with self.drop_egress(endpoint.port or 443, addresses):
            self.request("/periodic", "POST")
            self.process.wait(timeout=350)
        elapsed = time.monotonic() - started
        self.assertLessEqual(elapsed, 345)
        self.assertIn("Power down", self.logs[-1].read_text(errors="replace"))
        self.result(periodic_gpu_denial=True, backend_nras_drop=True, fail_closed_seconds=round(elapsed, 3))

    @unittest.skipUnless(
        os.environ.get("CVM_GPU_HARDWARE_TESTS") == "1" and os.environ.get("CVM_BACKEND_LOCAL") == "1",
        "Opt in with GPU hardware and an isolated local KBS",
    )
    def test_gpu_backend_unavailable_never_opens_vault(self):
        self.assertEqual(self.manifest["contract"]["gpu"], "nvidia_cc")
        endpoint = urlparse(self.manifest["contract"]["gpu_attestation_url"])
        addresses = sorted({item[4][0] for item in socket.getaddrinfo(endpoint.hostname, endpoint.port or 443)})
        with self.drop_egress(endpoint.port or 443, addresses):
            self.boot()
            self.process.wait(timeout=480)
        text = self.logs[-1].read_text(errors="replace")
        self.assertIn("CVM bootstrap failed", text)
        self.assertIn("Power down", text)
        # The bootstrap audit is emitted only after authorization and mounting;
        # an NRAS-unavailable boot must never reach that successful boundary.
        self.assertNotIn('"decision":"allow"', text.replace(" ", ""))
        self.result(gpu_backend_outage_key_denial=True, hardware_fault="backend_nras_drop")

    def test_reboot_after_writes_and_journal_interruption(self):
        self.boot()
        self.ready()
        self.request("/write", "POST")
        self.request("/write-loop", "POST")
        time.sleep(0.2)
        os.kill(self.qemu_pid(), signal.SIGKILL)
        self.process.wait(timeout=30)
        self.boot()
        state = self.ready()
        self.assertTrue(state["persisted"])
        self.stop()
        self.result(persistent_write_survived=True, journal_interruption_recovered=True)

    def test_reordered_scsi_targets_preserve_disk_roles(self):
        self.boot(reverse_scsi=True)
        state = self.ready()
        self.assertEqual(len(set(state["disk_paths"].values())), 5)
        self.assertEqual(state["measurements"], self.manifest["measurements"])
        self.stop()
        self.result(reversed_scsi_targets=True, correct_sidecar_mounts=True, disk_paths=state["disk_paths"])

    def test_integrity_monitor_crash_powers_off(self):
        self.boot()
        self.ready()
        self.request("/kill-monitor", "POST")
        self.process.wait(timeout=60)
        log = self.logs[-1].read_text(errors="replace")
        self.assertIn("Power down", log)
        self.result(monitor_crash_powered_off=True)

    def test_exec_child_cannot_dump_core(self):
        self.boot()
        self.ready()
        result = json.loads(self.request("/crash-child", "POST"))
        self.assertEqual(result, {"aborted": True, "core_dumped": False})
        self.stop()
        self.result(exec_child_core_dump_prevented=True)

    def test_cross_vault_with_fresh_hardware_appraisal(self):
        other = os.environ.get("CVM_OTHER_RESOURCE")
        if not other:
            self.skipTest("Provision a second key under this bundle and set CVM_OTHER_RESOURCE")
        self.boot()
        self.ready()
        result = json.loads(self.request("/cross-vault?resource=" + other, "POST"))
        self.assertEqual(result, {"cross_vault_denied": True, "fresh_positive_appraisal": True})
        self.stop()
        self.result(**result)

    def test_interrupted_docker_load_retries(self):
        if os.environ.get("CVM_EXTENDED_AGENT") != "1":
            self.skipTest("Use the current extended acceptance payload")
        self.boot()
        self.ready()
        self.assertEqual(
            json.loads(self.request("/prepare-interrupted-load", "POST")), {"load_in_progress": True, "marker": False}
        )
        time.sleep(0.5)
        self.assertFalse(json.loads(self.request())["marker"])
        os.kill(self.qemu_pid(), signal.SIGKILL)
        self.process.wait(timeout=30)
        self.boot()
        self.ready()
        self.stop()
        self.result(interrupted_docker_load_retried=True)

    def test_corruption_after_scan_stops_workload(self):
        if os.environ.get("CVM_EXTENDED_AGENT") != "1":
            self.skipTest("Use the current extended acceptance payload")
        self.boot()
        self.ready()
        # Fault injection by the untrusted host: change an existing data cluster
        # without changing qcow2 metadata, LUKS headers, or another VM's files.
        image = self.vault / "vault.qcow2"
        mapping = json.loads(run(["qemu-img", "map", "--force-share", "--output=json", image]))
        address = 64 * 1024**2
        block = next(b for b in mapping if b["start"] <= address < b["start"] + b["length"])
        self.assertTrue(block["data"] and block.get("offset") is not None)
        with image.open("r+b", buffering=0) as stream:
            stream.seek(block["offset"] + address - block["start"])
            data = stream.read(4096)
            stream.seek(block["offset"] + address - block["start"])
            stream.write(bytes(x ^ 0x55 for x in data))
            os.fsync(stream.fileno())
        self.request("/scan", "POST")
        self.process.wait(timeout=60)
        self.assertIn("Power down", self.logs[-1].read_text(errors="replace"))
        self.result(post_scan_corruption_powered_off=True)

    def test_wrong_header_binding_prevents_startup(self):
        path = self.vault / "vault_manifest.json"
        manifest = read_json(path)
        manifest["vault_bind"] = "a5" * 32
        write_json(path, manifest)
        self.boot()
        self.process.wait(timeout=120)
        log = self.logs[-1].read_text(errors="replace")
        self.assertIn("Power down", log)
        self.assertNotIn("Started \x1b[0;1;39mcvm_integrity.service", log)
        self.assertNotIn("CVM_WORKLOAD_STARTED", log)
        self.result(wrong_binding_prevented_startup=True)

    def test_payload_corruption_prevents_startup(self):
        with nbd(self.vault / "vault.qcow2") as device:
            header = snapshot_header(device)
            with open(device, "r+b", buffering=0) as stream:
                stream.seek(64 * 1024**2)
                original = stream.read(4096)
                stream.seek(64 * 1024**2)
                stream.write(bytes(x ^ 0x55 for x in original))
                os.fsync(stream.fileno())
            self.assertEqual(snapshot_header(device), header)
        self.boot()
        self.process.wait(timeout=180)
        log = self.logs[-1].read_text(errors="replace")
        self.assertIn("Power down", log)
        self.assertNotIn("CVM_WORKLOAD_STARTED", log)
        self.result(payload_corruption_prevented_startup=True, unchanged_header=True)
