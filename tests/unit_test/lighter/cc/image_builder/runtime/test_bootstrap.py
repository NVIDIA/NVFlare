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

"""Fail-closed ordering and supervision of the three-unit guest."""

import configparser
import contextlib
import io
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

from cvm.build.config import SOURCE
from cvm.common.contracts import HEADER_BYTES, STORAGE_PROFILE
from cvm.common.errors import BuildError
from cvm.common.evidence import serial_evidence, verify_reference
from cvm.common.io import read_json
from cvm.runtime import bootstrap, storage, supervisor
from cvm.runtime.systemd import notify


class BootstrapTests(unittest.TestCase):
    def test_attached_vault_skips_only_the_reference_report_probe(self):
        with (
            patch.object(bootstrap.Path, "exists", return_value=False),
            patch.object(bootstrap.Path, "is_block_device", return_value=True),
            patch.object(bootstrap, "disk_device", return_value="/dev/vault") as disk,
            patch.object(bootstrap, "local_report") as report,
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            bootstrap.reference()
        disk.assert_called_once_with("vault")
        report.assert_not_called()
        self.assertEqual(output.getvalue(), "")

    def test_absent_vault_emits_reference_only_for_zero_hardware_binding(self):
        for platform in ("intel_tdx", "amd_sev_snp"):
            for bound in (False, True):
                nonce = b"n" * 64
                report = bytearray(1024 if platform == "intel_tdx" else 1184)
                if platform == "intel_tdx":
                    report[0] = 0x81
                    report[128:192] = nonce
                    report[576:624] = bytes([int(bound)]) * 48
                else:
                    struct.pack_into("<I", report, 0, 3)
                    report[80:144] = nonce
                    report[192:224] = bytes([int(bound)]) * 32
                with (
                    self.subTest(platform=platform, bound=bound),
                    patch.object(bootstrap.Path, "exists", return_value=False),
                    patch.object(bootstrap.Path, "is_block_device", return_value=False),
                    patch.object(bootstrap.Path, "is_file", return_value=True),
                    patch.object(bootstrap.Path, "read_bytes", return_value=b"fixture-ccel"),
                    patch.object(bootstrap, "guest_platform", return_value=platform),
                    patch.object(bootstrap, "local_report", return_value=(bytes(report), nonce)) as read_report,
                    contextlib.redirect_stdout(io.StringIO()) as output,
                ):
                    bootstrap.reference()
                read_report.assert_called_once_with(platform)
                if bound:
                    self.assertEqual(output.getvalue(), "")
                else:
                    evidence = serial_evidence(output.getvalue())
                    self.assertIsNotNone(evidence)
                    verify_reference(platform, evidence)

    def test_bootstrap_orders_reference_firewall_clock_mount_and_workload(self):
        for dev in (False, True):
            events = []
            config = {"gpu": "none", "bootstrap_egress": [443]}
            with (
                self.subTest(dev=dev),
                tempfile.TemporaryDirectory() as directory,
                patch.object(bootstrap, "STATE", Path(directory)),
                patch.object(bootstrap, "read_json", return_value=config),
                patch.object(bootstrap.Path, "exists", return_value=dev),
                patch.object(bootstrap, "reference", side_effect=lambda: events.append("reference")),
                patch.object(bootstrap, "run", side_effect=lambda *a, **k: events.append("firewall")),
                patch.object(bootstrap, "discovered_resolvers", return_value=["10.0.0.53"]),
                patch.object(bootstrap, "firewall", side_effect=lambda *a, **k: events.append("narrow")) as narrow,
                patch.object(bootstrap, "time_sync", side_effect=lambda **k: events.append("clock")),
                patch.object(bootstrap, "mount_vault", side_effect=lambda *a, **k: events.append("vault")),
                patch.object(
                    bootstrap,
                    "finish_bootstrap",
                    side_effect=lambda *a, **k: events.append("configure") or ["cvm_app.service"],
                ),
                patch.object(bootstrap, "supervise", side_effect=lambda *a: events.append("supervise")),
            ):
                bootstrap.bootstrap()
            self.assertEqual(
                events,
                ["reference", *([] if dev else ["firewall", "narrow", "clock"]), "vault", "configure", "supervise"],
            )
            if not dev:
                # Pre-unlock rules keep only bootstrap egress and the learned resolvers.
                narrow.assert_called_once_with([], [443], resolvers=["10.0.0.53"])

    def test_missing_firewall_or_bad_clock_never_reaches_vault(self):
        for failed in ("run", "time_sync"):
            with (
                self.subTest(failed=failed),
                tempfile.TemporaryDirectory() as directory,
                patch.object(bootstrap, "STATE", Path(directory)),
                patch.object(bootstrap, "read_json", return_value={"bootstrap_egress": []}),
                patch.object(bootstrap.Path, "exists", return_value=False),
                patch.object(bootstrap, "reference"),
                patch.object(bootstrap, "run"),
                patch.object(bootstrap, "firewall"),
                patch.object(bootstrap, "time_sync"),
                patch.object(bootstrap, failed, side_effect=BuildError("gate denied")),
                patch.object(bootstrap, "mount_vault") as mount,
                patch.object(bootstrap, "supervise") as supervise,
            ):
                with self.assertRaises(BuildError):
                    bootstrap.bootstrap()
                mount.assert_not_called()
                supervise.assert_not_called()

    def test_resolvers_come_from_resolved_upstreams_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resolv.conf"
            path.write_text(
                "# generated\nnameserver 10.0.0.53\nnameserver 127.0.0.53\nnameserver fe80::1%eth0\n"
                "nameserver 10.0.0.53\nnameserver not-an-address\nsearch example.org\n"
            )
            self.assertEqual(bootstrap.discovered_resolvers(path), ["10.0.0.53", "fe80::1"])
            self.assertEqual(bootstrap.discovered_resolvers(Path(directory) / "absent"), [])

    def test_prescan_follows_the_measured_profile_and_the_monitor_still_gates_startup(self):
        for prescan in (True, False):
            events = []
            config = {
                "platform": "intel_tdx",
                "build_id": "test",
                "profile_version": "test-profile",
                "vault_prescan": prescan,
            }
            manifest = dict(
                config,
                cvm_build_id="test",
                storage_profile=STORAGE_PROFILE,
                vault_header_bytes=HEADER_BYTES,
                luks_uuid="uuid",
            )

            def execute(argv, **kwargs):
                if argv[:2] == ["cryptsetup", "luksUUID"]:
                    return b"uuid"
                if argv[:2] == ["systemctl", "is-active"]:
                    events.append("monitor-check")
                    return b"active"
                if argv[0] == "mount":
                    events.append("mount")
                return b""

            with (
                self.subTest(prescan=prescan),
                tempfile.TemporaryDirectory() as directory,
                patch.object(bootstrap, "STATE", Path(directory)),
                patch.object(bootstrap, "disk_device", side_effect=lambda name, **k: "/dev/" + name),
                patch.object(bootstrap, "protect_process"),
                patch.object(bootstrap, "guest_platform", return_value="intel_tdx"),
                patch.object(bootstrap, "snapshot_header", return_value=b"header"),
                patch.object(bootstrap, "binding", return_value=bytes(32)),
                patch.object(bootstrap, "verify_local_binding"),
                patch.object(bootstrap, "memory_file", side_effect=lambda *a, **k: contextlib.nullcontext(11)),
                patch.object(bootstrap, "inspect_header"),
                patch.object(bootstrap, "authorized_key", side_effect=lambda *a: contextlib.nullcontext(12)),
                patch.object(bootstrap, "validate_mapping"),
                patch.object(bootstrap, "scan", side_effect=lambda *a: events.append("scan")),
                patch.object(bootstrap, "read_json", return_value=manifest),
                patch.object(bootstrap, "local_report", return_value=(b"report", b"nonce")),
                patch.object(bootstrap, "measurements", return_value={}),
                patch.object(bootstrap, "run", side_effect=execute),
            ):
                bootstrap.mount_vault(config)
            expected = ["scan", "monitor-check", "mount"] if prescan else ["monitor-check", "mount"]
            self.assertEqual(events[: len(expected)], expected)
            self.assertEqual(events.count("scan"), int(prescan))

    def test_reopen_requires_the_same_vault_identity_before_remounting(self):
        config = {"platform": "intel_tdx", "build_id": "test", "profile_version": "test-profile", "gpu": "none"}
        identity = {"digest": "00" * 32, "luks_uuid": "uuid"}
        for changed in (False, True):
            events = []

            def read(path):
                if path == bootstrap.CONFIG:
                    return config
                if str(path).endswith("binding.json"):
                    return identity
                return {
                    "platform": "intel_tdx",
                    "cvm_build_id": "test",
                    "profile_version": "test-profile",
                    "storage_profile": STORAGE_PROFILE,
                    "vault_header_bytes": HEADER_BYTES,
                    "luks_uuid": "uuid",
                }

            with (
                self.subTest(changed=changed),
                tempfile.TemporaryDirectory() as directory,
                patch.object(bootstrap, "STATE", Path(directory)),
                patch.object(bootstrap.Path, "exists", return_value=False),
                patch.object(bootstrap, "protect_process"),
                patch.object(bootstrap, "time_sync"),
                patch.object(bootstrap, "guest_platform", return_value="intel_tdx"),
                patch.object(bootstrap, "disk_device", return_value="/dev/vault"),
                patch.object(bootstrap, "read_json", side_effect=read),
                patch.object(
                    bootstrap,
                    "open_vault",
                    return_value=(bytes([1]) * 32 if changed else bytes(32), "uuid"),
                ),
                patch.object(bootstrap, "verify_payload", side_effect=lambda *a: events.append("verify")),
                patch.object(bootstrap, "mount_roles", side_effect=lambda *a: events.append("mount")),
                patch.object(bootstrap, "readiness", side_effect=lambda *a: events.append("ready")),
                patch.object(bootstrap, "close_vault") as close,
            ):
                if changed:
                    with self.assertRaisesRegex(BuildError, "identity changed"):
                        bootstrap.reopen()
                    self.assertEqual(events, [])
                else:
                    bootstrap.reopen()
                    self.assertEqual(events, ["verify", "mount", "ready"])
                self.assertEqual(close.call_count, int(changed))

    def test_failed_reopen_cleans_up_each_partial_stage_before_retry(self):
        for failed in ("open_vault", "verify_payload", "mount_roles", "check_vault_manifest", "readiness"):
            active = [False]

            def opened(*args):
                self.assertFalse(active[0], "previous attempt left its mapping active")
                active[0] = True
                return bytes(32), "uuid"

            with self.subTest(stage=failed), contextlib.ExitStack() as stack:
                for name in (
                    "protect_process",
                    "time_sync",
                    "verify_payload",
                    "mount_roles",
                    "check_vault_manifest",
                    "readiness",
                ):
                    stack.enter_context(patch.object(bootstrap, name))
                stack.enter_context(patch.object(bootstrap.Path, "exists", return_value=False))
                stack.enter_context(patch.object(bootstrap, "guest_platform", return_value="intel_tdx"))
                stack.enter_context(patch.object(bootstrap, "disk_device", return_value="/dev/vault"))
                stack.enter_context(
                    patch.object(
                        bootstrap,
                        "read_json",
                        side_effect=lambda path: (
                            {"platform": "intel_tdx"}
                            if path == bootstrap.CONFIG
                            else {"digest": "00" * 32, "luks_uuid": "uuid"}
                        ),
                    )
                )
                stack.enter_context(patch.object(bootstrap, "open_vault", side_effect=opened))
                close = stack.enter_context(
                    patch.object(bootstrap, "close_vault", side_effect=lambda: active.__setitem__(0, False))
                )

                def fail(*args):
                    if failed == "open_vault":
                        opened()
                    raise BuildError("transient failure")

                with patch.object(bootstrap, failed, side_effect=fail):
                    with self.assertRaisesRegex(BuildError, "transient failure"):
                        bootstrap.reopen()
                close.assert_called_once_with()
                self.assertFalse(active[0])
                bootstrap.reopen()
                self.assertTrue(active[0])

    def test_vault_cleanup_handles_partial_state_and_rejects_incomplete_cleanup(self):
        for mounted, mapped in ((False, False), (False, True), (True, True)):
            for fails in (False, True):
                if fails and not mapped:
                    continue
                state = {"mount": mounted, "mapping": mapped}
                commands = []

                def execute(argv, **kwargs):
                    if argv[0] == "dmsetup":
                        return b"  vault\n" if state["mapping"] else b"  other-vault\n"
                    commands.append(argv)
                    if not fails:
                        state["mount" if argv[0] == "umount" else "mapping"] = False

                with (
                    self.subTest(mounted=mounted, mapped=mapped, fails=fails),
                    patch.object(storage.Path, "is_mount", side_effect=lambda: state["mount"]),
                    # A killed cryptsetup may not have created its device link.
                    patch.object(storage.Path, "exists", return_value=False),
                    patch.object(storage, "run", side_effect=execute),
                ):
                    if fails:
                        with self.assertRaisesRegex(BuildError, "cleanup could not be confirmed"):
                            storage.close_vault()
                    else:
                        storage.close_vault()
                        self.assertFalse(any(state.values()))
                expected = ([["umount", "/vault"]] if mounted else []) + (
                    [["cryptsetup", "close", "vault"]] if mapped else []
                )
                self.assertEqual(commands, expected)

    def test_reopen_timeout_kills_the_process_group_before_vault_cleanup(self):
        events = []
        child = MagicMock(pid=123)
        child.__enter__.return_value = child
        child.communicate.side_effect = subprocess.TimeoutExpired("reopen", 3)
        child.wait.side_effect = lambda: events.append("reaped")
        with (
            patch.object(supervisor.subprocess, "Popen", return_value=child) as spawn,
            patch.object(supervisor.os, "killpg", side_effect=lambda *args: events.append("killed")) as kill,
        ):
            with self.assertRaisesRegex(BuildError, "TimeoutExpired"):
                supervisor.run_reopen(3, {})
        self.assertTrue(spawn.call_args.kwargs["start_new_session"])
        kill.assert_called_once_with(123, signal.SIGKILL)
        self.assertEqual(events, ["killed", "reaped"])

    def test_container_is_confined_and_environment_values_stay_out_of_argv(self):
        app = {
            "image_id": "sha256:" + "a" * 64,
            "requires_gpu": False,
            "container": {
                "ports": [],
                "volumes": [],
                "env": {"APP_SECRET": "value-with-secret", "type": "bind,source=/vault/application"},
                "tee_device": False,
            },
        }
        command = bootstrap.docker_argv(app)
        self.assertEqual(command[command.index("--cap-drop") + 1], "ALL")
        self.assertEqual(command[command.index("--security-opt") + 1], "no-new-privileges")
        self.assertEqual(command[command.index("--pids-limit") + 1], "4096")
        added = [command[i + 1] for i, value in enumerate(command) if value == "--cap-add"]
        self.assertIn("CHOWN", added)
        self.assertNotIn("NET_RAW", added)
        self.assertNotIn("SYS_ADMIN", added)
        self.assertNotIn("value-with-secret", " ".join(command))
        self.assertEqual(
            [command[i + 1] for i, value in enumerate(command) if value == "--env"], ["APP_SECRET", "type"]
        )
        self.assertTrue(
            all(
                "source=/vault/application,target=/vault/application" in m or "," in m
                for m in command
                if m.startswith("type=bind")
            )
        )
        self.assertNotIn("/host/bin", " ".join(command))
        self.assertNotIn("--read-only", command)
        self.assertEqual(bootstrap.docker_environment(app)["APP_SECRET"], "value-with-secret")

        app["container"].update(host_bin=True, read_only_rootfs=True, capabilities=["NET_BIND_SERVICE"], pids_limit=64)
        command = bootstrap.docker_argv(app)
        self.assertIn("type=bind,source=/usr/bin,target=/host/bin,readonly", command)
        self.assertIn("--read-only", command)
        self.assertEqual(
            [command[i + 1] for i, value in enumerate(command) if value == "--cap-add"], ["NET_BIND_SERVICE"]
        )
        self.assertEqual(command[command.index("--pids-limit") + 1], "64")
        app["container"]["capabilities"] = ["SYS_ADMIN"]
        with self.assertRaises(BuildError):
            bootstrap.docker_argv(app)

    def test_requested_stop_exits_zero_and_unexpected_exit_keeps_its_status(self):
        app = {
            "image_id": "sha256:" + "a" * 64,
            "requires_gpu": False,
            "container": {"ports": [], "volumes": [], "env": {}, "tee_device": False},
        }
        image = json.dumps([{"Id": app["image_id"], "Config": {}}]).encode()
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            for requested, status, expected in ((True, 143, 0), (False, 7, 7), (False, -9, 137)):
                stops = []

                def wait():
                    if requested:
                        os.kill(os.getpid(), signal.SIGTERM)
                    return status

                process = Mock(**{"wait.side_effect": wait})
                with (
                    self.subTest(requested=requested, status=status),
                    patch.object(bootstrap.Path, "is_dir", return_value=True),
                    patch.object(bootstrap.Path, "is_symlink", return_value=False),
                    patch.object(bootstrap.Path, "is_file", return_value=True),
                    patch.object(
                        bootstrap,
                        "read_json",
                        side_effect=lambda path: (
                            {"platform": "intel_tdx"}
                            if path == bootstrap.CONFIG
                            else {"image_id": app["image_id"]} if "loaded" in str(path) else app
                        ),
                    ),
                    patch.object(bootstrap, "run", return_value=image),
                    patch.object(bootstrap, "readiness"),
                    patch.object(bootstrap.subprocess, "Popen", return_value=process),
                    patch.object(bootstrap.subprocess, "run", side_effect=lambda argv, **k: stops.append(argv)),
                ):
                    self.assertEqual(bootstrap.application(), expected)
                self.assertEqual([argv[:2] for argv in stops], [["docker", "stop"]] if requested else [])
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)

    def test_monitor_ready_precedes_scan_and_mount_and_wrong_binding_prevents_authorization(self):
        for wrong_binding in (False, True):
            events = []
            config = {"platform": "intel_tdx", "build_id": "test", "profile_version": "test-profile"}
            manifest = dict(
                config,
                cvm_build_id="test",
                storage_profile=STORAGE_PROFILE,
                vault_header_bytes=HEADER_BYTES,
                luks_uuid="uuid",
            )

            def execute(argv, **kwargs):
                if argv[:2] == ["cryptsetup", "luksUUID"]:
                    return b"uuid"
                if argv == ["systemctl", "start", "cvm_integrity.service"]:
                    events.append("monitor-ready")
                if argv[:2] == ["systemctl", "is-active"]:
                    return b"active"
                if argv[0] == "mount":
                    events.append("mount")
                return b""

            with (
                self.subTest(wrong_binding=wrong_binding),
                tempfile.TemporaryDirectory() as directory,
                patch.object(bootstrap, "STATE", Path(directory)),
                patch.object(bootstrap, "disk_device", side_effect=lambda name, **k: "/dev/" + name),
                patch.object(bootstrap, "protect_process"),
                patch.object(bootstrap, "guest_platform", return_value="intel_tdx"),
                patch.object(bootstrap, "snapshot_header", return_value=b"header"),
                patch.object(bootstrap, "binding", return_value=bytes(32)),
                patch.object(
                    bootstrap,
                    "verify_local_binding",
                    side_effect=BuildError("wrong binding") if wrong_binding else None,
                ),
                patch.object(bootstrap, "memory_file", side_effect=lambda *a, **k: contextlib.nullcontext(11)),
                patch.object(bootstrap, "inspect_header"),
                patch.object(
                    bootstrap, "authorized_key", side_effect=lambda *a: contextlib.nullcontext(12)
                ) as authorize,
                patch.object(bootstrap, "validate_mapping"),
                patch.object(bootstrap, "scan", side_effect=lambda *a: events.append("scan")),
                patch.object(bootstrap, "read_json", return_value=manifest),
                patch.object(bootstrap, "local_report", return_value=(b"report", b"nonce")),
                patch.object(bootstrap, "measurements", return_value={}),
                patch.object(bootstrap, "run", side_effect=execute),
            ):
                if wrong_binding:
                    with self.assertRaises(BuildError):
                        bootstrap.mount_vault(config)
                    authorize.assert_not_called()
                    self.assertEqual(events, [])
                else:
                    bootstrap.mount_vault(config)
                    self.assertEqual(events[:3], ["monitor-ready", "scan", "mount"])

    def test_dev_mount_never_attests_or_starts_integrity_monitor(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(bootstrap, "STATE", Path(directory)),
            patch.object(bootstrap.Path, "exists", return_value=False),
            patch.object(bootstrap, "disk_device", side_effect=lambda name, **k: "/dev/" + name),
            patch.object(bootstrap, "read_json", return_value={"dev_mode": True, "cvm_build_id": "test"}),
            patch.object(bootstrap, "authorized_key") as authorize,
            patch.object(bootstrap, "run") as execute,
        ):
            bootstrap.mount_vault({"build_id": "test"}, dev=True)
            authorize.assert_not_called()
            self.assertEqual(len(execute.call_args_list), 4)
            self.assertTrue(all(call.args[0][0] == "mount" for call in execute.call_args_list))

    def test_nfs_and_generated_units_are_ready_before_workload_start(self):
        events = []
        app = {
            "requires_gpu": False,
            "allowed_ports": [],
            "allowed_out_ports": [443],
            "container": {"ports": []},
            "hosts_entries": {},
        }
        with (
            patch.object(bootstrap, "read_json", return_value=app),
            patch("builtins.open", mock_open()),
            patch.object(bootstrap, "discovered_resolvers", return_value=[]),
            patch.object(bootstrap, "firewall", side_effect=lambda *a, **k: events.append("firewall")),
            patch.object(bootstrap, "mount_user_data", side_effect=lambda: events.append("nfs")),
            patch.object(
                bootstrap, "install_services", side_effect=lambda: events.append("units") or ["app_test.service"]
            ),
            patch.object(bootstrap, "run", side_effect=lambda *a: events.append("reload")),
        ):
            self.assertEqual(
                bootstrap.finish_bootstrap({"gpu": "none", "bootstrap_egress": [443]}),
                ["cvm_app.service", "app_test.service"],
            )
        self.assertEqual(events, ["firewall", "nfs", "units", "reload"])

    def test_generated_units_depend_on_bootstrap_and_delegate_failure_to_pid1(self):
        destination = MagicMock()
        source = Mock()
        item = Mock(name="item")
        item.name = "app_test.service"
        item.read_text.return_value = "[Service]\nType=simple\nExecStart=/vault/application/test\n"
        source.iterdir.return_value = [item]

        def path(value):
            return destination if value == "/run/systemd/system" else source

        with patch.object(bootstrap, "Path", side_effect=path):
            self.assertEqual(bootstrap.install_services(), ["app_test.service"])
        unit = destination.__truediv__.return_value.write_text.call_args.args[0]
        for directive in (
            "After=cvm_bootstrap.service",
            "Requires=cvm_bootstrap.service",
            "BindsTo=cvm_bootstrap.service",
            "FailureAction=poweroff-force",
            "EnvironmentFile=/run/cvm/platform.env",
            "NoNewPrivileges=yes",
            "ProtectSystem=strict",
            "ReadWritePaths=/vault/application/runtime /vault/application/data /applog",
            "ProtectKernelModules=yes",
            "CapabilityBoundingSet=~CAP_SYS_MODULE",
        ):
            self.assertIn(directive, unit)
        self.assertNotIn("PartOf=", unit)
        self.assertNotIn("cvm_integrity.service", unit)

    def test_exactly_three_units_with_pid1_failure_actions(self):
        names = {p.name for p in (SOURCE / "services").iterdir()}
        self.assertEqual(names, {"cvm_bootstrap.service", "cvm_integrity.service", "cvm_app.service"})
        for name in names:
            config = configparser.ConfigParser(interpolation=None)
            config.read(SOURCE / "services" / name)
            self.assertEqual(config["Unit"]["FailureAction"], "poweroff-force")
            self.assertNotIn("OnFailure", config["Unit"])
            self.assertNotIn("ExecStopPost", config["Service"])
            if name != "cvm_app.service":
                self.assertEqual(config["Unit"]["SuccessAction"], "poweroff-force")
                self.assertEqual(config["Service"]["Type"], "notify")
            if name == "cvm_integrity.service":
                self.assertNotIn("Requires", config["Unit"])
                self.assertNotIn("After", config["Unit"])
                self.assertEqual(config["Service"]["WatchdogSec"], "15")
            if name == "cvm_app.service":
                # The wrapper stops the container on SIGTERM; a stop job must not
                # look like a workload failure and trigger the forced power-off.
                self.assertEqual(config["Service"]["KillMode"], "mixed")
                self.assertNotIn("ExecStop", config["Service"])

    def test_integrity_monitor_tolerates_only_a_deliberate_quarantine(self):
        from cvm.runtime import integrity

        with tempfile.TemporaryDirectory() as directory:
            flag = Path(directory) / "quarantine.json"
            mapper = Path(directory) / "vault"
            mapper.touch()
            with (
                patch.object(integrity, "validate_mapping", return_value="253:1") as resolve,
                patch.object(integrity, "run", return_value=b"0 123 integrity 0 123 0") as status,
            ):
                self.assertEqual(integrity.check_vault(None, flag, mapper), ["253", "1"])
                resolve.assert_called_once()
                self.assertEqual(status.call_args.args[0][:3], ["dmsetup", "status", "-j"])
                # A vanished mapping outside quarantine is fatal.
                mapper.unlink()
                with self.assertRaisesRegex(BuildError, "disappeared"):
                    integrity.check_vault(["253", "1"], flag, mapper)
            flag.write_text("{}")
            with patch.object(integrity, "validate_mapping", side_effect=AssertionError("must not resolve")):
                self.assertIsNone(integrity.check_vault(["253", "1"], flag, mapper))
            flag.unlink()
            with (
                patch.object(integrity, "validate_mapping", side_effect=BuildError("no mapping")),
                self.assertRaises(BuildError),
            ):
                integrity.check_vault(None, flag, mapper)

    def test_notify_uses_actual_systemd_datagram_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            address = str(Path(directory) / "notify")
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as receiver:
                receiver.bind(address)
                receiver.settimeout(1)
                with patch.dict(os.environ, NOTIFY_SOCKET=address):
                    notify("READY=1\nSTATUS=Ready")
                self.assertEqual(receiver.recv(4096), b"READY=1\nSTATUS=Ready")


@unittest.skipUnless(hasattr(signal, "sigtimedwait"), "Linux synchronous signal support")
class SupervisorTests(unittest.TestCase):
    def test_timed_out_reopen_leaves_no_running_descendant(self):
        spawn = subprocess.Popen
        program = (
            "import subprocess, sys, time\n"
            "from pathlib import Path\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
            "Path(sys.argv[1]).write_text(str(child.pid))\n"
            "time.sleep(60)\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            pidfile = Path(directory) / "descendant.pid"

            def fixture(argv, **kwargs):
                return spawn([sys.executable, "-c", program, str(pidfile)], **kwargs)

            with patch.object(supervisor.subprocess, "Popen", side_effect=fixture):
                with self.assertRaisesRegex(BuildError, "TimeoutExpired"):
                    supervisor.run_reopen(2, dict(os.environ))
            pid = int(pidfile.read_text())
            try:
                status = Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[0]
            except FileNotFoundError:
                status = "gone"
            try:
                self.assertIn(status, ("gone", "Z", "X"))
            finally:
                if status not in ("gone", "Z", "X"):
                    os.kill(pid, signal.SIGKILL)

    def test_ready_precedes_synchronous_start_and_pending_signal_runs_a_tick(self):
        events = []
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(supervisor.signal, "pthread_sigmask", side_effect=lambda *a: events.append("mask") or set()),
            patch.object(supervisor.signal, "sigtimedwait", side_effect=[object(), KeyboardInterrupt]) as wait,
            patch.object(supervisor, "notify", side_effect=lambda *a: events.append("ready")),
            patch.object(supervisor, "emit", side_effect=lambda *a: events.append("audit")),
            patch.object(supervisor, "run", side_effect=lambda *a, **k: events.append("start")) as start,
            patch.object(supervisor, "periodic_tick", side_effect=lambda *a: events.append("tick")),
            patch.object(supervisor.time, "monotonic", return_value=100),
        ):
            with self.assertRaises(KeyboardInterrupt):
                supervisor.supervise({}, ["cvm_app.service", "app_test.service"], Path(directory))
            self.assertEqual(events, ["mask", "audit", "ready", "start", "tick", "mask"])
            self.assertEqual(start.call_args.args[0], ["systemctl", "start", "cvm_app.service", "app_test.service"])
            self.assertEqual(wait.call_args.args, ({signal.SIGUSR1}, 300))

    def test_cadence_accounts_for_startup_and_slow_appraisal(self):
        clock = [0]
        starts = []
        waits = []

        def wait(signals, timeout):
            waits.append(timeout)
            clock[0] += timeout

        def tick(*args):
            starts.append(clock[0])
            clock[0] += 240
            if len(starts) == 3:
                raise KeyboardInterrupt

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(supervisor.signal, "pthread_sigmask", return_value=set()),
            patch.object(supervisor.signal, "sigtimedwait", side_effect=wait),
            patch.object(supervisor.time, "monotonic", side_effect=lambda: clock[0]),
            patch.object(supervisor, "notify"),
            patch.object(supervisor, "emit"),
            patch.object(supervisor, "run", side_effect=lambda *a, **k: clock.__setitem__(0, 100)),
            patch.object(supervisor, "periodic_tick", side_effect=tick),
        ):
            with self.assertRaises(KeyboardInterrupt):
                supervisor.supervise({}, ["cvm_app.service"], Path(directory))
        self.assertEqual(starts, [300, 600, 900])
        self.assertEqual(waits, [200, 60, 60])

    def test_requested_tick_and_overrun_do_not_add_a_full_sleep(self):
        clock = [0]
        waits = []
        starts = []

        def wait(signals, timeout):
            waits.append(timeout)
            clock[0] += 20 if len(waits) == 1 else timeout

        def tick(*args):
            starts.append(clock[0])
            clock[0] += 301
            if len(starts) == 2:
                raise KeyboardInterrupt

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(supervisor.signal, "pthread_sigmask", return_value=set()),
            patch.object(supervisor.signal, "sigtimedwait", side_effect=wait),
            patch.object(supervisor.time, "monotonic", side_effect=lambda: clock[0]),
            patch.object(supervisor, "notify"),
            patch.object(supervisor, "emit"),
            patch.object(supervisor, "run"),
            patch.object(supervisor, "periodic_tick", side_effect=tick),
        ):
            with self.assertRaises(KeyboardInterrupt):
                supervisor.supervise({}, ["cvm_app.service"], Path(directory))
        self.assertEqual(starts, [20, 321])
        self.assertEqual(waits, [300, 0])

    def test_sigusr1_queued_during_workload_start_runs_immediately(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(supervisor, "notify"),
            patch.object(supervisor, "emit"),
            patch.object(supervisor, "run", side_effect=lambda *a, **k: os.kill(os.getpid(), signal.SIGUSR1)),
            patch.object(supervisor, "periodic_tick", side_effect=KeyboardInterrupt) as tick,
        ):
            started = supervisor.time.monotonic()
            with self.assertRaises(KeyboardInterrupt):
                supervisor.supervise({}, ["cvm_app.service"], Path(directory))
            self.assertLess(supervisor.time.monotonic() - started, 2)
            tick.assert_called_once_with({}, Path(directory), 1)

    def test_periodic_failure_or_timeout_revokes_gpu_and_records_failure(self):
        for failure in (BuildError("denied"), BuildError("timeout")):
            with (
                self.subTest(failure=failure),
                tempfile.TemporaryDirectory() as directory,
                patch.object(supervisor, "run", side_effect=failure) as execute,
                patch.object(supervisor, "readiness") as readiness,
            ):
                state = Path(directory)
                with self.assertRaises(BuildError):
                    supervisor.periodic_tick({"gpu": "nvidia_cc"}, state, 1)
                readiness.assert_called_once_with({"gpu": "nvidia_cc"}, False)
                self.assertEqual(execute.call_args.kwargs["timeout"], 300)
                self.assertEqual(read_json(state / "periodic.json")["result"], "failed")

    def test_periodic_success_records_completion_and_drops_notify_socket(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, NOTIFY_SOCKET="private"),
            patch.object(supervisor, "run", return_value=b"") as execute,
            patch.object(supervisor, "readiness") as readiness,
        ):
            state = Path(directory)
            supervisor.periodic_tick({}, state, 3)
            record = read_json(state / "periodic.json")
            self.assertEqual(record["sequence"], 3)
            self.assertEqual(record["result"], "success")
            self.assertNotIn("NOTIFY_SOCKET", execute.call_args.kwargs["env"])
            readiness.assert_not_called()
            self.assertEqual(execute.call_args.args[0][-3:], ["-m", "cvm.runtime.bootstrap", "periodic"])

    def test_gpu_reset_failure_cannot_turn_a_failed_tick_into_success(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(supervisor, "run", side_effect=BuildError("denied")),
            patch.object(supervisor, "readiness", side_effect=BuildError("GPU unavailable")),
        ):
            state = Path(directory)
            with self.assertRaises(BuildError):
                supervisor.periodic_tick({}, state, 1)
            self.assertEqual(read_json(state / "periodic.json")["result"], "failed")

    def test_failed_tick_enters_quarantine_instead_of_powering_off_at_once(self):
        events = []
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(supervisor.signal, "pthread_sigmask", return_value=set()),
            patch.object(supervisor.signal, "sigtimedwait", side_effect=[object(), object(), KeyboardInterrupt]),
            patch.object(supervisor, "notify"),
            patch.object(supervisor, "emit"),
            patch.object(supervisor, "run"),
            patch.object(supervisor, "periodic_tick", side_effect=[BuildError("denied"), None]),
            patch.object(supervisor, "quarantine", side_effect=lambda *a: events.append("quarantine")),
            patch.object(supervisor.time, "monotonic", return_value=100),
        ):
            with self.assertRaises(KeyboardInterrupt):
                supervisor.supervise({}, ["cvm_app.service"], Path(directory))
        self.assertEqual(events, ["quarantine"])

    def test_quarantine_drops_the_key_retries_within_the_window_and_restarts_units(self):
        commands = []
        clock = [1000.0]

        def reopen(timeout, environment):
            commands.append(["reopen"])
            self.assertNotIn("NOTIFY_SOCKET", environment)
            self.assertEqual(timeout, 1900 - clock[0])
            if commands.count(["reopen"]) == 1:
                raise BuildError("still denied")
            return b"CVM_ATTESTATION=allow\n"

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(supervisor, "run", side_effect=lambda argv, **kwargs: commands.append(argv)),
            patch.object(supervisor, "run_reopen", side_effect=reopen),
            patch.object(supervisor, "readiness", side_effect=lambda *args: commands.append(["revoke"])),
            patch.object(supervisor, "close_vault", side_effect=lambda: commands.append(["close"])),
            patch.dict(os.environ, NOTIFY_SOCKET="private"),
            patch.object(supervisor, "emit", side_effect=lambda decision: commands.append(["emit", decision])),
            patch.object(
                supervisor.time, "sleep", side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)
            ),
            patch.object(supervisor.time, "monotonic", side_effect=lambda: clock[0]),
        ):
            state = Path(directory)
            supervisor.quarantine({"gpu": "none"}, ["cvm_app.service", "app_x.service"], state)
            self.assertFalse((state / "quarantine.json").exists())
        self.assertEqual(
            [c[:2] if c[0] != "emit" else c for c in commands],
            [
                ["emit", "quarantine"],
                ["systemctl", "stop"],
                ["systemctl", "stop"],
                ["revoke"],
                ["close"],
                ["reopen"],
                ["revoke"],
                ["close"],
                ["reopen"],
                ["systemctl", "start"],
            ],
        )
        self.assertEqual(commands[1][2:], ["cvm_app.service", "app_x.service"])
        self.assertEqual(commands[2][2:], ["docker.service", "containerd.service"])
        self.assertEqual(commands[-1][2:], ["cvm_app.service", "app_x.service"])

    def test_quarantine_window_expiry_fails_closed(self):
        clock = [0.0]

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(supervisor, "run") as run,
            patch.object(supervisor, "run_reopen", side_effect=BuildError("denied")) as reopen,
            patch.object(supervisor, "revoke_vault"),
            patch.object(supervisor, "emit"),
            patch.object(
                supervisor.time, "sleep", side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)
            ),
            patch.object(supervisor.time, "monotonic", side_effect=lambda: clock[0]),
        ):
            state = Path(directory)
            with self.assertRaises(BuildError):
                supervisor.quarantine({}, ["cvm_app.service"], state)
            # The flag stays while the vault is closed; PID 1 powers off next.
            self.assertTrue((state / "quarantine.json").exists())
            self.assertEqual(
                reopen.call_count, supervisor.QUARANTINE_WINDOW_SECONDS // supervisor.QUARANTINE_RETRY_SECONDS
            )
            self.assertFalse(any(c.args[0][:2] == ["systemctl", "start"] for c in run.call_args_list))

    def test_quarantine_bounds_attempts_and_rejects_late_success(self):
        for final_duration, success in ((39, True), (40, False), (899, False)):
            clock = [0.0]
            attempts = []

            def reopen(timeout, environment):
                attempts.append((clock[0], timeout))
                clock[0] += 800 if len(attempts) == 1 else final_duration
                if len(attempts) == 1:
                    raise BuildError("temporary failure")
                return b""

            with (
                self.subTest(duration=final_duration),
                tempfile.TemporaryDirectory() as directory,
                patch.object(supervisor, "run") as run,
                patch.object(supervisor, "run_reopen", side_effect=reopen),
                patch.object(supervisor, "revoke_vault") as revoke,
                patch.object(supervisor, "emit"),
                patch.object(supervisor.time, "monotonic", side_effect=lambda: clock[0]),
                patch.object(
                    supervisor.time, "sleep", side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)
                ),
            ):
                state = Path(directory)
                if success:
                    supervisor.quarantine({}, ["cvm_app.service"], state)
                else:
                    with self.assertRaisesRegex(BuildError, "deadline expired"):
                        supervisor.quarantine({}, ["cvm_app.service"], state)
                self.assertEqual(attempts, [(0, 900), (860, 40)])
                self.assertEqual(revoke.call_count, 2 if success else 3)
                self.assertEqual((state / "quarantine.json").exists(), not success)
                self.assertEqual(any(c.args[0][:2] == ["systemctl", "start"] for c in run.call_args_list), success)

    def test_quarantine_bounds_sleep_and_does_not_retry_after_expiry(self):
        clock = [0.0]

        def reopen(*args):
            clock[0] += 890
            raise BuildError("denied")

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(supervisor, "run") as run,
            patch.object(supervisor, "run_reopen", side_effect=reopen) as attempt,
            patch.object(supervisor, "revoke_vault"),
            patch.object(supervisor, "emit"),
            patch.object(supervisor.time, "monotonic", side_effect=lambda: clock[0]),
            patch.object(
                supervisor.time, "sleep", side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)
            ) as sleep,
        ):
            with self.assertRaisesRegex(BuildError, "deadline expired"):
                supervisor.quarantine({}, ["cvm_app.service"], Path(directory))
        self.assertEqual(attempt.call_count, 1)
        sleep.assert_called_once_with(10)
        self.assertFalse(any(c.args[0][:2] == ["systemctl", "start"] for c in run.call_args_list))

    def test_quarantine_cleanup_failure_never_retries_or_restarts(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(supervisor, "run") as run,
            patch.object(supervisor, "run_reopen", side_effect=BuildError("timeout")) as reopen,
            patch.object(supervisor, "readiness"),
            patch.object(supervisor, "close_vault", side_effect=[None, BuildError("still mounted")]),
            patch.object(supervisor, "emit"),
            patch.object(supervisor.time, "sleep") as sleep,
        ):
            state = Path(directory)
            with self.assertRaisesRegex(BuildError, "still mounted"):
                supervisor.quarantine({}, ["cvm_app.service"], state)
            self.assertTrue((state / "quarantine.json").exists())
        self.assertEqual(reopen.call_count, 1)
        sleep.assert_not_called()
        self.assertFalse(any(c.args[0][:2] == ["systemctl", "start"] for c in run.call_args_list))

    def test_readiness_reset_failure_still_closes_vault(self):
        with (
            patch.object(supervisor, "readiness", side_effect=BuildError("GPU unavailable")),
            patch.object(supervisor, "close_vault") as close,
        ):
            with self.assertRaises(BuildError):
                supervisor.revoke_vault({"gpu": "nvidia_cc"})
        close.assert_called_once_with()
