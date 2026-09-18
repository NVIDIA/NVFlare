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
import os
import signal
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

from cvm.build.config import SOURCE
from cvm.common.contracts import HEADER_BYTES, STORAGE_PROFILE
from cvm.common.errors import BuildError
from cvm.common.io import read_json
from cvm.runtime import bootstrap, supervisor
from cvm.runtime.systemd import notify


class BootstrapTests(unittest.TestCase):
    def test_bootstrap_orders_reference_firewall_clock_mount_and_workload(self):
        for dev in (False, True):
            events = []
            config = {"gpu": "none"}
            with (
                self.subTest(dev=dev),
                tempfile.TemporaryDirectory() as directory,
                patch.object(bootstrap, "STATE", Path(directory)),
                patch.object(bootstrap, "read_json", return_value=config),
                patch.object(bootstrap.Path, "exists", return_value=dev),
                patch.object(bootstrap, "reference", side_effect=lambda: events.append("reference")),
                patch.object(bootstrap, "run", side_effect=lambda *a, **k: events.append("firewall")),
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
                events, ["reference", *([] if dev else ["firewall", "clock"]), "vault", "configure", "supervise"]
            )

    def test_missing_firewall_or_bad_clock_never_reaches_vault(self):
        for failed in ("run", "time_sync"):
            with (
                self.subTest(failed=failed),
                tempfile.TemporaryDirectory() as directory,
                patch.object(bootstrap, "STATE", Path(directory)),
                patch.object(bootstrap, "read_json", return_value={}),
                patch.object(bootstrap.Path, "exists", return_value=False),
                patch.object(bootstrap, "reference"),
                patch.object(bootstrap, "run"),
                patch.object(bootstrap, "time_sync"),
                patch.object(bootstrap, failed, side_effect=BuildError("gate denied")),
                patch.object(bootstrap, "mount_vault") as mount,
                patch.object(bootstrap, "supervise") as supervise,
            ):
                with self.assertRaises(BuildError):
                    bootstrap.bootstrap()
                mount.assert_not_called()
                supervise.assert_not_called()

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
            patch.object(bootstrap, "firewall", side_effect=lambda *a: events.append("firewall")),
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
