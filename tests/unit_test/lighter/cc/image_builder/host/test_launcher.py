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

import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from cvm.common.errors import BuildError
from cvm.common.io import write_json
from cvm.host.launcher import (
    GRACEFUL_SHUTDOWN_SECONDS,
    find_bundle,
    gpu_devices,
    graceful_stop,
    launch,
    qemu_command,
    qmp_powerdown,
    qmp_socket_path,
    run_vm,
    runtime_state_path,
    select_gpus,
    shutdown,
    vfio_gpus,
)


def launch_manifest(platform="intel_tdx", dev=False, **shape):
    launch_shape = {
        "cpu_model": "host",
        "vcpus": 4,
        "memory_gib": 8,
        "quote_generation": {"type": "vsock", "cid": 2, "port": 4050},
        "snp_policy": 0x30000,
    }
    launch_shape.update(shape)
    return {
        "platform": platform,
        "dev_mode": dev,
        "contract": {"gpu": "none"},
        "launch_shape": launch_shape,
        "cmdline": "root=/dev/mapper/verity_root",
    }


DISKS = [f"/disk-{i}" for i in range(5)]


class LauncherTests(unittest.TestCase):
    def test_duplicate_launch_preserves_the_live_qmp_socket(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            bundle = directory / "bundle"
            bundle.mkdir()
            for path in [bundle / "verity_root.qcow2"] + [
                directory / f"{name}.qcow2" for name in ("applog", "user_config", "user_data", "vault")
            ]:
                path.touch()
            manifest = dict(launch_manifest(), build_id="test")
            write_json(
                directory / "vault_manifest.json",
                {"cvm_build_id": "test", "platform": "intel_tdx", "vault_bind": "00" * 32, "allowed_ports": []},
            )
            qmp = directory / "live.qmp"
            with (
                socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server,
                (directory / "vault.qcow2").open("r+b") as owner,
                patch("cvm.host.launcher.find_bundle", return_value=bundle),
                patch("cvm.host.launcher.verify_bundle", return_value=manifest),
                patch("cvm.host.launcher.host_capabilities", return_value=["intel_tdx"]),
                patch("cvm.host.launcher.qmp_socket_path", return_value=qmp),
                patch("cvm.host.launcher.run_vm") as run,
            ):
                server.bind(str(qmp))
                server.listen(1)
                server.settimeout(2)
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(BuildError, "Vault is already attached"):
                    launch(directory)
                # A pathname check alone is insufficient: the active endpoint
                # must still be reachable after the rejected launch.
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                    client.settimeout(2)
                    client.connect(str(qmp))
                    connection, _ = server.accept()
                    connection.close()
                run.assert_not_called()

    def test_all_launch_modes_disable_vmport(self):
        for platform in ("intel_tdx", "amd_sev_snp"):
            for dev in (False, True):
                manifest = launch_manifest(platform, dev)
                with self.subTest(platform=platform, dev=dev):
                    command = qemu_command(manifest, "/bundle", DISKS, bytes(32), cbit=51)
                    self.assertIn("vmport=off", command[command.index("-machine") + 1].split(","))

    def test_launch_uses_an_explicit_device_set_and_no_terminal_monitor(self):
        command = qemu_command(launch_manifest(), "/bundle", DISKS, bytes(32), qmp="/run/cvm-builder/x.qmp")
        self.assertIn("-nodefaults", command)
        self.assertNotIn("-nographic", command)
        self.assertEqual(command[command.index("-display") + 1], "none")
        self.assertEqual(command[command.index("-serial") + 1], "stdio")
        self.assertEqual(command[command.index("-monitor") + 1], "none")
        self.assertEqual(command[command.index("-qmp") + 1], "unix:/run/cvm-builder/x.qmp,server=on,wait=off")
        self.assertNotIn("-qmp", qemu_command(launch_manifest(), "/bundle", DISKS, bytes(32)))

    def test_snp_policy_comes_from_the_measured_launch_shape(self):
        def policy(**shape):
            command = qemu_command(launch_manifest("amd_sev_snp", **shape), "/bundle", DISKS, bytes(32), cbit=51)
            objects = [json.loads(command[i + 1]) for i, value in enumerate(command) if value == "-object"]
            return next(item for item in objects if item["id"] == "tee0")["policy"]

        self.assertEqual(policy(snp_policy=0x30000), 0x30000)
        self.assertEqual(policy(snp_policy=0x130000 | 0x0100), 0x130100)
        for unsafe in (0x30000 | (1 << 19), 0x30000 | (1 << 18), 0x10000, None, "0x30000", 0x2030000):
            with self.subTest(policy=unsafe), self.assertRaises(BuildError):
                policy(snp_policy=unsafe)

    def test_forwarded_ports_bind_to_the_requested_host_address(self):
        def network(**options):
            command = qemu_command(launch_manifest(), "/bundle", DISKS, bytes(32), host_ports=[8080], **options)
            return command[command.index("-netdev") + 1]

        self.assertIn("hostfwd=tcp:0.0.0.0:8080-:8080", network())
        self.assertIn("hostfwd=tcp:127.0.0.1:8080-:8080", network(bind_address="127.0.0.1"))
        for invalid in ("::1", "localhost", ""):
            with self.subTest(address=invalid), self.assertRaises(BuildError):
                network(bind_address=invalid)

    def test_graceful_stop_requests_powerdown_and_waits_before_termination(self):
        process = Mock(**{"poll.return_value": None})
        with patch("cvm.host.launcher.qmp_powerdown") as powerdown:
            graceful_stop(process, "/run/cvm-builder/x.qmp")
        powerdown.assert_called_once_with("/run/cvm-builder/x.qmp")
        process.wait.assert_called_once_with(timeout=GRACEFUL_SHUTDOWN_SECONDS)
        process.terminate.assert_not_called()
        # A dead control socket falls back to termination without waiting.
        process = Mock(**{"poll.return_value": None})
        with patch("cvm.host.launcher.qmp_powerdown", side_effect=OSError("gone")):
            graceful_stop(process, "/run/cvm-builder/x.qmp")
        process.wait.assert_not_called()
        graceful_stop(process, None)
        process.wait.assert_not_called()

    def test_qmp_powerdown_negotiates_capabilities_then_requests_powerdown(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "qmp.sock"
            received = []
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(str(path))
            server.listen(1)
            server.settimeout(5)

            def serve():
                connection, _ = server.accept()
                with connection, connection.makefile("rwb", buffering=0) as stream:
                    stream.write(b'{"QMP": {"version": {}, "capabilities": []}}\n')
                    received.append(json.loads(stream.readline()))
                    stream.write(b'{"return": {}}\n')
                    received.append(json.loads(stream.readline()))
                    # Events may interleave with command replies.
                    stream.write(b'{"event": "POWERDOWN", "timestamp": {"seconds": 1}}\n{"return": {}}\n')

            thread = threading.Thread(target=serve)
            thread.start()
            try:
                qmp_powerdown(path)
            finally:
                thread.join(timeout=5)
                server.close()
            self.assertEqual([item["execute"] for item in received], ["qmp_capabilities", "system_powerdown"])

    def test_qemu_is_detached_from_the_launch_terminal(self):
        process = Mock(**{"poll.return_value": 0, "wait.return_value": 0})
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("cvm.host.launcher.subprocess.Popen", return_value=process) as spawn,
            patch("cvm.host.launcher.runtime_state_path", return_value=Path(directory) / "state.json"),
            patch("cvm.host.launcher.write_runtime_state"),
        ):
            self.assertEqual(run_vm(["qemu"], directory, qmp="/run/x.qmp"), 0)
        self.assertEqual(spawn.call_args.kwargs, {"stdin": subprocess.DEVNULL, "start_new_session": True})
        with patch.dict("os.environ", {"CVM_RUNTIME_DIRECTORY": "/run/test"}):
            self.assertEqual(qmp_socket_path("/srv/cvm/delivery").suffix, ".qmp")
            self.assertEqual(qmp_socket_path("/srv/cvm/delivery").parent, Path("/run/test"))

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

                    def spawn(command, **kwargs):
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
