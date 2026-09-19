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

"""Runtime bundle discovery and GPU selection use deterministic local inputs."""

import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from cvm.common.errors import BuildError
from cvm.common.io import write_json
from cvm.host.launcher import (
    find_bundle,
    gpu_devices,
    qemu_command,
    run_vm,
    runtime_state_path,
    select_gpus,
    shutdown,
    vfio_gpus,
)


class LauncherTests(unittest.TestCase):
    def test_all_launch_modes_disable_vmport(self):
        for platform in ("intel_tdx", "amd_sev_snp"):
            for dev in (False, True):
                manifest = {
                    "platform": platform,
                    "dev_mode": dev,
                    "contract": {"gpu": "none"},
                    "launch_shape": {
                        "cpu_model": "host",
                        "vcpus": 4,
                        "memory_gib": 8,
                        "quote_generation": {"type": "vsock", "cid": 2, "port": 4050},
                    },
                    "cmdline": "root=/dev/mapper/verity_root",
                }
                with self.subTest(platform=platform, dev=dev):
                    command = qemu_command(manifest, "/bundle", [f"/disk-{i}" for i in range(5)], bytes(32), cbit=51)
                    self.assertIn("vmport=off", command[command.index("-machine") + 1].split(","))

    def test_startup_signals_stop_qemu_and_restore_handlers(self):
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT, signal.SIGINT):
            for window in ("spawn", "state"):
                with self.subTest(signal=sig, window=window), tempfile.TemporaryDirectory() as temporary:
                    process = Mock(pid=123, **{"poll.return_value": None})
                    handlers = {}
                    original = {s: object() for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT, signal.SIGINT)}
                    state = Path(temporary) / "runtime.json"

                    def register(number, handler):
                        old = handlers.get(number, original[number])
                        handlers[number] = handler
                        return old

                    def spawn(command):
                        if window == "spawn":
                            handlers[sig](sig, None)
                        return process

                    def publish(directory, child):
                        write_json(state, {"launcher_pid": os.getpid(), "launcher_start": 456})
                        handlers[sig](sig, None)

                    with (
                        patch("cvm.host.launcher.signal.signal", side_effect=register),
                        patch("cvm.host.launcher.subprocess.Popen", side_effect=spawn),
                        patch("cvm.host.launcher.runtime_state_path", return_value=state),
                        patch("cvm.host.launcher.write_runtime_state", side_effect=publish) as writer,
                        patch("cvm.host.launcher._process_start", return_value=456),
                    ):
                        self.assertEqual(run_vm(["qemu"], temporary), 128 + sig)
                    process.terminate.assert_called_once()
                    process.wait.assert_called_once_with(timeout=30)
                    if window == "spawn":
                        writer.assert_not_called()
                    self.assertFalse(state.exists())
                    self.assertEqual(handlers, original)

    def test_qemu_exit_status_preserves_exit_codes_and_converts_signals(self):
        for status, expected in ((0, 0), (7, 7), (-signal.SIGTERM, 143), (-signal.SIGKILL, 137)):
            process = Mock(**{"poll.return_value": status, "wait.return_value": status})
            with (
                self.subTest(status=status),
                tempfile.TemporaryDirectory() as directory,
                patch("cvm.host.launcher.subprocess.Popen", return_value=process),
                patch("cvm.host.launcher.runtime_state_path", return_value=Path(directory) / "state.json"),
                patch("cvm.host.launcher.write_runtime_state"),
            ):
                self.assertEqual(run_vm(["qemu"], directory), expected)
            process.terminate.assert_not_called()

    def test_shutdown_detects_orphaned_disk_locks_without_signalling_processes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            image = directory / "vault.qcow2"
            image.touch()
            state = directory / "runtime.json"
            for lock in ("flock(f, fcntl.LOCK_EX)", "lockf(f, fcntl.LOCK_EX, 1, 100)"):
                command = (
                    "import fcntl, sys; f = open(sys.argv[1], 'r+b'); "
                    f"fcntl.{lock}; print('ready', flush=True); sys.stdin.read(1)"
                )
                with subprocess.Popen(
                    [sys.executable, "-c", command, str(image)], stdin=subprocess.PIPE, stdout=subprocess.PIPE
                ) as holder:
                    try:
                        self.assertEqual(holder.stdout.readline(), b"ready\n")
                        for stale in (False, True):
                            if stale:
                                write_json(state, {"launcher_pid": 123, "launcher_start": 456})
                            with (
                                self.subTest(lock=lock, stale=stale),
                                patch("cvm.host.launcher.runtime_state_path", return_value=state),
                                patch("cvm.host.launcher._process_start", side_effect=FileNotFoundError),
                                patch("cvm.host.launcher.os.kill") as kill,
                                self.assertRaisesRegex(BuildError, "Vault is still locked"),
                            ):
                                shutdown(directory)
                            kill.assert_not_called()
                    finally:
                        holder.communicate(b"x", timeout=5)

    def test_state_publication_failure_kills_unresponsive_qemu_and_restores_signals(self):
        signals = (signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT, signal.SIGINT)
        original = {sig: signal.getsignal(sig) for sig in signals}
        process = Mock(pid=123, **{"poll.return_value": None})
        process.wait.side_effect = [subprocess.TimeoutExpired("qemu", 30), -signal.SIGKILL]
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("cvm.host.launcher.subprocess.Popen", return_value=process),
                patch("cvm.host.launcher.runtime_state_path", return_value=Path(directory) / "state.json"),
                patch("cvm.host.launcher.write_runtime_state", side_effect=OSError("state unavailable")),
                self.assertRaisesRegex(OSError, "state unavailable"),
            ):
                run_vm(["qemu"], directory)
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        self.assertEqual(process.wait.call_args_list[-1].kwargs, {"timeout": 10})
        self.assertEqual({sig: signal.getsignal(sig) for sig in signals}, original)

    def test_self_contained_delivery_finds_embedded_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            delivery = Path(temporary) / "intel_tdx"
            bundle = delivery / "cvm_bundle"
            bundle.mkdir(parents=True)
            metadata = {
                "profile_version": "cpu-2026.09",
                "platform": "intel_tdx",
                "cvm_build_id": "cvm-example",
            }
            write_json(bundle / "cvm_manifest.json", {"build_id": "cvm-example"})
            self.assertEqual(find_bundle(delivery, metadata), bundle.resolve())

    def test_build_cache_requires_explicit_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            delivery = root / "vault_app" / "intel_tdx"
            bundle = root / "cvm_cpu-2026.09" / "intel_tdx"
            delivery.mkdir(parents=True)
            bundle.mkdir(parents=True)
            metadata = {
                "profile_version": "cpu-2026.09",
                "platform": "intel_tdx",
                "cvm_build_id": "cvm-example",
            }
            write_json(bundle / "cvm_manifest.json", {"build_id": "cvm-example"})
            with self.assertRaisesRegex(BuildError, "use --cvm-bundle"):
                find_bundle(delivery, metadata)
            self.assertEqual(find_bundle(delivery, metadata, bundle), bundle.resolve())
            # The documented explicit override takes precedence over an embedded bundle.
            embedded = delivery / "cvm_bundle"
            embedded.mkdir()
            write_json(embedded / "cvm_manifest.json", {"build_id": "stale"})
            self.assertEqual(find_bundle(delivery, metadata, bundle), bundle.resolve())

    def test_wrong_bundle_is_not_selected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            delivery = root / "vault_app" / "intel_tdx"
            bundle = root / "cvm_cpu-2026.09" / "intel_tdx"
            delivery.mkdir(parents=True)
            bundle.mkdir(parents=True)
            write_json(bundle / "cvm_manifest.json", {"build_id": "other"})
            with self.assertRaises(BuildError):
                find_bundle(
                    delivery,
                    {
                        "profile_version": "cpu-2026.09",
                        "platform": "intel_tdx",
                        "cvm_build_id": "cvm-example",
                    },
                )

    def test_gpu_detection_requires_the_configured_nvidia_display_controllers(self):
        with tempfile.TemporaryDirectory() as temporary:
            devices = Path(temporary) / "devices"
            groups = Path(temporary) / "groups"
            devices.mkdir()
            groups.mkdir()

            def device(address, vendor, pci_class):
                path = devices / address
                path.mkdir()
                (path / "vendor").write_text(vendor)
                (path / "class").write_text(pci_class)
                group = groups / address
                (group / "devices").mkdir(parents=True)
                (path / "iommu_group").symlink_to(group)
                return path

            device("0000:41:00.0", "0x10de", "0x030200")
            device("0000:42:00.0", "0x8086", "0x030000")
            self.assertEqual(gpu_devices(devices), ["0000:41:00.0"])
            self.assertEqual(select_gpus(None, 1, devices), ["0000:41:00.0"])
            device("0000:43:00.0", "0x10de", "0x030000")
            with self.assertRaises(BuildError):
                select_gpus(None, 1, devices)
            self.assertEqual(
                select_gpus(["41:00.0", "43:00.0"], 2, devices),
                ["0000:41:00.0", "0000:43:00.0"],
            )

    def test_vfio_setup_and_host_driver_restore(self):
        with tempfile.TemporaryDirectory() as temporary:
            pci = Path(temporary) / "pci"
            devices = pci / "devices"
            drivers = pci / "drivers"
            groups = Path(temporary) / "groups"
            device = devices / "0000:41:00.0"
            for path in (device, drivers / "nvidia", drivers / "vfio-pci", groups / "1/devices"):
                path.mkdir(parents=True)
            (device / "vendor").write_text("0x10de")
            (device / "class").write_text("0x030200")
            (device / "driver_override").write_text("(null)\n")
            (device / "iommu_group").symlink_to(groups / "1")
            (groups / "1/devices/0000:41:00.0").symlink_to(device)
            (device / "driver").symlink_to(drivers / "nvidia")

            def write(path, value):
                path = Path(path)
                if path.name == "driver_override":
                    path.write_text(value + "\n")
                elif path.name == "unbind":
                    (device / "driver").unlink()
                elif path.name == "drivers_probe":
                    selected = (device / "driver_override").read_text().strip() or "nvidia"
                    (device / "driver").symlink_to(drivers / selected)

            with (
                patch("cvm.host.launcher.run"),
                patch("cvm.host.launcher.subprocess.run"),
                patch("cvm.host.launcher._sysfs_write", side_effect=write),
            ):
                with vfio_gpus(expected_count=1, sysfs=devices) as assignments:
                    self.assertEqual(assignments, [["0000:41:00.0"]])
                    self.assertEqual((device / "driver").resolve().name, "vfio-pci")
            self.assertEqual((device / "driver").resolve().name, "nvidia")
            self.assertEqual((device / "driver_override").read_text(), "\n")

    def test_vfio_setup_includes_slot_functions_split_across_iommu_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            pci = Path(temporary) / "pci"
            devices = pci / "devices"
            groups = Path(temporary) / "groups"
            display = devices / "0000:41:00.0"
            audio = devices / "0000:41:00.1"
            drivers = pci / "drivers"
            for path in (
                display,
                audio,
                drivers / "nvidia",
                drivers / "snd_hda_intel",
                drivers / "vfio-pci",
                groups / "1/devices",
                groups / "2/devices",
            ):
                path.mkdir(parents=True)
            for device, pci_class, group, driver in (
                (display, "0x030200", groups / "1", "nvidia"),
                (audio, "0x040300", groups / "2", "snd_hda_intel"),
            ):
                (device / "vendor").write_text("0x10de")
                (device / "class").write_text(pci_class)
                (device / "driver_override").write_text("(null)\n")
                (device / "iommu_group").symlink_to(group)
                (group / "devices" / device.name).symlink_to(device)
                (device / "driver").symlink_to(drivers / driver)

            def write(path, value):
                path = Path(path)
                device = devices / value
                if path.name == "driver_override":
                    path.write_text(value + "\n")
                elif path.name == "unbind":
                    device = devices / value
                    (device / "driver").unlink()
                elif path.name == "drivers_probe":
                    selected = (device / "driver_override").read_text().strip() or "nvidia"
                    (device / "driver").symlink_to(drivers / selected)

            with (
                patch("cvm.host.launcher.run"),
                patch("cvm.host.launcher.subprocess.run"),
                patch("cvm.host.launcher._sysfs_write", side_effect=write),
            ):
                with vfio_gpus(expected_count=1, sysfs=devices) as assignments:
                    self.assertEqual(assignments, [["0000:41:00.0", "0000:41:00.1"]])
                    self.assertEqual(display.joinpath("driver").resolve().name, "vfio-pci")
                    self.assertEqual(audio.joinpath("driver").resolve().name, "vfio-pci")

    def test_detected_gpu_is_passed_to_qemu(self):
        manifest = {
            "platform": "intel_tdx",
            "contract": {"gpu": "nvidia_cc", "gpu_count": 2},
            "launch_shape": {
                "cpu_model": "host",
                "vcpus": 4,
                "memory_gib": 8,
                "quote_generation": {"type": "vsock", "cid": 2, "port": 4050},
            },
            "cmdline": "root=/dev/mapper/verity_root",
        }
        command = qemu_command(
            manifest,
            "/bundle",
            [f"/disk-{index}.qcow2" for index in range(5)],
            bytes(32),
            gpus=[["0000:41:00.0", "0000:41:00.1"], ["0000:43:00.0"]],
        )
        self.assertIn("vfio-pci,host=0000:41:00.0,bus=gpu1,romfile=,iommufd=iommufd0", command)
        self.assertIn("vfio-pci,host=0000:41:00.1,bus=gpu1,romfile=,iommufd=iommufd0", command)
        self.assertIn("vfio-pci,host=0000:43:00.0,bus=gpu2,romfile=,iommufd=iommufd0", command)
        self.assertEqual(command.count("pcie-root-port,id=gpu1,bus=pcie.0,chassis=1,slot=1,pref64-reserve=256G"), 1)
        self.assertEqual(command.count("pcie-root-port,id=gpu2,bus=pcie.0,chassis=2,slot=2,pref64-reserve=256G"), 1)
        self.assertIn('{"qom-type": "iommufd", "id": "iommufd0"}', command)

    def test_shutdown_signals_only_validated_launcher_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            delivery = Path(temporary) / "delivery"
            runtime = Path(temporary) / "run"
            delivery.mkdir()
            with patch.dict("os.environ", {"CVM_RUNTIME_DIRECTORY": str(runtime)}):
                state = runtime_state_path(delivery)
                state.parent.mkdir()
                write_json(state, {"launcher_pid": 123, "launcher_start": 456})

                def remove_state(pid, sent_signal):
                    self.assertEqual((pid, sent_signal), (123, signal.SIGTERM))
                    state.unlink()

                with (
                    patch("cvm.host.launcher._process_start", return_value=456),
                    patch("cvm.host.launcher.os.kill", side_effect=remove_state),
                ):
                    shutdown(delivery)


if __name__ == "__main__":
    unittest.main()
