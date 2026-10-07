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

"""Security boundaries exercised by the September PR review."""

import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from cvm.common.errors import BuildError
from cvm.common.firewall import firewall_rules
from cvm.runtime import bootstrap


class ReviewBoundaryTests(unittest.TestCase):
    def test_stalled_audit_cannot_delay_quarantine_or_denial(self):
        # Isolate the intentionally stalled daemon writer from the test runner.
        program = textwrap.dedent(
            """
            import contextlib, os, subprocess, sys, tempfile, threading, time
            from pathlib import Path
            from unittest.mock import patch
            from cvm.common.errors import BuildError
            from cvm.runtime import audit, bootstrap, supervisor
            entered, release = threading.Event(), threading.Event()
            def stuck(path, line):
                entered.set()
                release.wait(60)
            with tempfile.TemporaryDirectory() as directory, \
                 patch.object(audit, 'read_json', return_value={'build_id': 'test', 'attestation_policy_id': 'default'}), \
                 patch.object(audit.Path, 'exists', return_value=False), \
                 patch.object(audit, 'append', side_effect=stuck):
                audit.emit('allow')
                assert entered.wait(2)
                writer = audit._writer
                workload = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
                def stop(argv, **kwargs):
                    if argv[:2] == ['systemctl', 'stop']:
                        workload.terminate()
                        workload.wait(timeout=1)
                started = time.monotonic()
                try:
                    with patch.object(supervisor, 'run', side_effect=stop), \
                         patch.object(supervisor, 'revoke_vault'), \
                         patch.object(supervisor, 'watchdog') as watchdog, \
                         patch.object(supervisor, 'QUARANTINE_WINDOW_SECONDS', 0):
                        try:
                            supervisor.quarantine({}, ['cvm_app.service'], Path(directory))
                        except BuildError:
                            pass
                        else:
                            raise AssertionError('Denial must fail closed')
                        watchdog.assert_called_once_with(supervisor.REVOCATION_TIMEOUT_SECONDS)
                    assert workload.poll() is not None
                    with patch.object(sys, 'argv', ['bootstrap', 'bootstrap']), \
                         patch.object(bootstrap, 'bootstrap', side_effect=BuildError('denied')), \
                         patch.object(bootstrap, 'notify') as notify:
                        try:
                            bootstrap.main()
                        except SystemExit:
                            pass
                        else:
                            raise AssertionError('Must exit after denial')
                        notify.assert_called_once_with('WATCHDOG=trigger')
                    for _ in range(100):
                        audit.emit('deny')
                    assert audit._writer is writer and audit._records.qsize() <= 8
                    assert time.monotonic() - started < 2
                finally:
                    workload.kill() if workload.poll() is None else None
                    workload.wait(timeout=1)
                    release.set()
        """
        )
        result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_docker_client_does_not_receive_container_control_variables(self):
        values = {"DOCKER_HOST": "tcp://untrusted:2375", "PATH": "/evil", "LD_PRELOAD": "/evil.so", "SECRET": "private"}
        app = {
            "image_id": "sha256:" + "a" * 64,
            "requires_gpu": False,
            "container": {"env": values, "volumes": [], "ports": []},
        }
        image = json.dumps([{"Id": app["image_id"], "Config": {}}]).encode()
        stored = []

        @contextlib.contextmanager
        def memory(data, **kwargs):
            stored.append(data)
            yield 17

        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            with (
                patch.object(bootstrap.Path, "is_dir", return_value=True),
                patch.object(bootstrap.Path, "is_symlink", return_value=False),
                patch.object(bootstrap.Path, "is_file", return_value=True),
                patch.object(
                    bootstrap,
                    "read_json",
                    side_effect=lambda path: {"platform": "intel_tdx"} if path == bootstrap.CONFIG else app,
                ),
                patch.object(bootstrap, "run", return_value=image) as run,
                patch.object(bootstrap, "memory_file", side_effect=memory),
                patch.object(bootstrap.subprocess, "Popen", return_value=Mock(**{"wait.return_value": 0})) as spawn,
                patch.dict(os.environ, DOCKER_HOST="tcp://also-untrusted", LD_PRELOAD="bad.so"),
            ):
                self.assertEqual(bootstrap.application(), 0)
                command = spawn.call_args.args[0]
                self.assertEqual(command[:3], ["/usr/bin/docker", "--host", "unix:///var/run/docker.sock"])
                self.assertEqual(spawn.call_args.kwargs["env"], bootstrap.DOCKER_ENVIRONMENT)
                self.assertEqual(spawn.call_args.kwargs["pass_fds"], (17,))
                self.assertEqual(command[command.index("--env-file") + 1], "/proc/self/fd/17")
                self.assertNotIn("private", str(spawn.call_args))
                self.assertEqual(stored, ["".join(f"{k}={v}\n" for k, v in values.items()).encode()])
                self.assertEqual(run.call_args.kwargs["env"], bootstrap.DOCKER_ENVIRONMENT)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        for value in ("a\nb", "a\rb", "a\x00b"):
            app["container"]["env"] = {"BAD": value}
            with self.assertRaises(BuildError):
                bootstrap.docker_environment(app)

    def test_nfs_mount_never_uses_a_sidecar_path(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "nfs_data"
            target.mkdir()
            hostile = Path(directory) / "user_data"
            hostile.mkdir()
            (hostile / "mnt").symlink_to("/run/systemd/system")
            app = {"nfs_mount": {"server": "nfs.example.org", "export": "/records", "security": "krb5p"}}
            with (
                patch.object(bootstrap, "NFS_MOUNT", target),
                patch.object(bootstrap, "read_json", return_value=app),
                patch.object(bootstrap, "run") as mount,
            ):
                bootstrap.mount_user_data()
                self.assertEqual(mount.call_args.args[0][-1], str(target))
                self.assertNotIn(str(hostile), str(mount.call_args))
                target.rmdir()
                target.symlink_to(hostile)
                mount.reset_mock()
                with self.assertRaisesRegex(BuildError, "guest-owned"):
                    bootstrap.mount_user_data()
                mount.assert_not_called()

    def test_sidecars_are_ext4_and_old_output_is_reformatted_without_a_journal(self):
        with patch.object(bootstrap, "run") as run:
            bootstrap.mount_roles({"applog": "/dev/output", "user-data": "/dev/input"})
            format_call, log_mount, input_mount = [call.args[0] for call in run.call_args_list]
            self.assertEqual(format_call[0], "mkfs.ext4")
            self.assertIn("^has_journal", format_call)
            self.assertEqual(format_call[-1], "/dev/output")
            self.assertEqual(log_mount[:3], ["mount", "-t", "ext4"])
            self.assertEqual(input_mount[:3], ["mount", "-t", "ext4"])
            self.assertIn("noload", input_mount[4])
        with patch.object(bootstrap, "run", side_effect=BuildError("format denied")) as run:
            with self.assertRaises(BuildError):
                bootstrap.mount_roles({"applog": "/dev/output"})
            self.assertEqual(run.call_count, 1)

    def test_missing_resolvers_deny_dns_before_conntrack_and_general_port_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resolv.conf"
            for contents in (None, "", "nameserver 127.0.0.53\nnameserver ::\nnameserver invalid\n"):
                if contents is not None:
                    path.write_text(contents)
                resolvers = bootstrap.discovered_resolvers(path)
                self.assertEqual(resolvers, [])
                rules = firewall_rules([], [53, 443], resolvers=resolvers)
                for name in ("output", "forward"):
                    chain = rules.split("chain " + name, 1)[1].split("chain", 1)[0]
                    for protocol in ("udp", "tcp"):
                        self.assertNotIn(protocol + " dport 53 accept", chain)
                        self.assertLess(chain.index(protocol + " dport 53 drop"), chain.index("ct state established"))
        self.assertIn("udp dport 53 accept", firewall_rules([], [], resolvers=None))
