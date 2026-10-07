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

"""Opt-in Linux mount, packet and PID 1 checks; never run guest power-off actions on the host."""

import configparser
import errno
import http.client
import json
import os
import select
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from cvm.common.firewall import firewall_rules
from cvm.common.io import read_json
from cvm.runtime import audit, bootstrap
from cvm.runtime.systemd import notify, watchdog


def command(argv, **kwargs):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=15, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def firewall_namespace(directory):
    command(["ip", "link", "set", "lo", "up"])
    bootstrap.STATE = Path(directory)
    bootstrap.firewall([], [443])
    print("ready", flush=True)
    time.sleep(60)


def confined_firewall_agent(directory):
    import lab_guest_agent as agent

    status = Path(directory) / "firewall.json"
    # This service is in the namespace where bootstrap installed the real table,
    # but has the application's capability and filesystem restrictions.
    denied = subprocess.run(["nft", "list", "table", "inet", "cvm"], capture_output=True, timeout=5)
    assert denied.returncode != 0 and b"Operation not permitted" in denied.stderr
    try:
        status.write_text("tampered")
    except OSError as error:
        assert error.errno in (errno.EROFS, errno.EACCES)
    else:
        raise AssertionError("Application service could overwrite bootstrap verification")
    with patch.object(agent, "read_json", side_effect=lambda path: read_json(status)):
        with patch.object(agent, "state", side_effect=agent.firewall_state):
            with agent.http.server.HTTPServer(("127.0.0.1", 0), agent.Handler) as server:
                thread = threading.Thread(target=server.handle_request, daemon=True)
                thread.start()
                connection = http.client.HTTPConnection(*server.server_address, timeout=5)
                try:
                    connection.request("GET", "/state")
                    response = connection.getresponse()
                    assert response.status == 200
                    report = json.loads(response.read())
                    assert report["firewall_present"] is True
                    assert report["firewall_verified_at"] == read_json(status)["verified_at"]
                finally:
                    connection.close()
                    thread.join(timeout=5)
    print("Confined agent read verified firewall status; nft access and status writes denied.")


def mount_probe():
    assert os.readlink("/proc/self/ns/mnt") != os.readlink(f"/proc/{os.getppid()}/ns/mnt")
    command(["mount", "--make-rprivate", "/"])
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        log, nfs, victim, data = (root / name for name in ("applog", "nfs_data", "control", "user_data"))
        for path in (log, nfs, victim, data):
            path.mkdir()
        marker = victim / "untouched"
        marker.write_text("measured control data")
        (data / "mnt").symlink_to(victim)
        image = root / "output.img"
        with image.open("wb") as stream:
            stream.truncate(64 * 1024 * 1024)
        command(["mkfs.ext4", "-q", "-F", str(image)])
        command(["mount", "-t", "ext4", str(image), str(log)])
        (log / "untrusted-old.log").write_text("host-supplied data")
        command(["umount", str(log)])
        try:
            with patch.dict(bootstrap.MOUNT_POINTS, applog=str(log)):
                bootstrap.mount_roles({"applog": str(image)})
            assert not (log / "untrusted-old.log").exists()
            (log / "new.log").write_text("public output")
            features = command(["dumpe2fs", "-h", str(image)]).split("Filesystem features:", 1)[1].splitlines()[0]
            assert "has_journal" not in features
            original = bootstrap.run

            def mount(argv, **kwargs):
                # Exercise the actual kernel target resolution without needing a
                # Kerberos server. The transport's krb5p contract is unit-tested.
                assert argv[:3] == ["mount", "-t", "nfs4"]
                assert argv[-1] == str(nfs)
                return original(["mount", "-t", "tmpfs", "-o", "ro,nosuid,nodev,noexec", "tmpfs", argv[-1]])

            with (
                patch.object(bootstrap, "NFS_MOUNT", nfs),
                patch.object(
                    bootstrap,
                    "read_json",
                    return_value={"nfs_mount": {"server": "nfs.example.org", "export": "/data", "security": "krb5p"}},
                ),
                patch.object(bootstrap, "run", side_effect=mount),
            ):
                bootstrap.mount_user_data()
            assert marker.read_text() == "measured control data"
            assert os.path.ismount(nfs) and not os.path.ismount(victim)
        finally:
            for path in (nfs, log):
                if os.path.ismount(path):
                    command(["umount", str(path)])


def dns_server(ipv6):
    family = socket.AF_INET6 if ipv6 else socket.AF_INET
    addresses = ("2001:db8:2::2", "2001:db8:2::3") if ipv6 else ("198.18.2.2", "198.18.2.3")

    def serve(address, kind):
        with socket.socket(family, kind) as sock:
            sock.bind((address, 53))
            if kind == socket.SOCK_STREAM:
                sock.listen()
                while True:
                    connection, _ = sock.accept()
                    with connection:
                        connection.sendall(connection.recv(16))
            else:
                while True:
                    data, peer = sock.recvfrom(16)
                    sock.sendto(data, peer)

    for address in addresses:
        for kind in (socket.SOCK_STREAM, socket.SOCK_DGRAM):
            threading.Thread(target=serve, args=(address, kind), daemon=True).start()
    time.sleep(0.1)
    print("ready", flush=True)
    time.sleep(120)


def dns_client(ipv6, addresses):
    results = []
    for address in addresses:
        for kind in (socket.SOCK_STREAM, socket.SOCK_DGRAM):
            with socket.socket(socket.AF_INET6 if ipv6 else socket.AF_INET, kind) as sock:
                sock.settimeout(0.3)
                if kind == socket.SOCK_DGRAM:
                    # Reuse the tuple across firewall replacements to cover
                    # conntrack-established traffic from the discovery phase.
                    sock.bind(("::" if ipv6 else "0.0.0.0", 33333))
                try:
                    sock.connect((address, 53))
                    sock.send(b"query")
                    results.append(sock.recv(16) == b"query")
                except (TimeoutError, OSError):
                    results.append(False)
    print(json.dumps(results), flush=True)


def dns_probe(ipv6):
    assert os.readlink("/proc/self/ns/net") != os.readlink(f"/proc/{os.getppid()}/ns/net")
    command(["mount", "--make-rprivate", "/"])
    command(["mount", "-t", "tmpfs", "tmpfs", "/run"])
    Path("/run/netns").mkdir()
    version = "-6" if ipv6 else "-4"
    gateway, client, router, allowed, other = (
        ("2001:db8:1::1", "2001:db8:1::2", "2001:db8:2::1", "2001:db8:2::2", "2001:db8:2::3")
        if ipv6
        else ("198.18.1.1", "198.18.1.2", "198.18.2.1", "198.18.2.2", "198.18.2.3")
    )
    bits = "64" if ipv6 else "24"
    for namespace, interface, local, addresses in (
        ("client", "docker0", gateway, [client]),
        ("dns", "wan", router, [allowed, other]),
    ):
        command(["ip", "netns", "add", namespace])
        command(["ip", "link", "add", interface, "type", "veth", "peer", "name", "peer"])
        command(["ip", "link", "set", "peer", "netns", namespace])
        command(["ip", version, "addr", "add", local + "/" + bits, "dev", interface, "nodad"])
        command(["ip", "link", "set", interface, "up"])
        net = ["ip", "netns", "exec", namespace]
        for address in addresses:
            command(net + ["ip", version, "addr", "add", address + "/" + bits, "dev", "peer", "nodad"])
        command(net + ["ip", "link", "set", "peer", "up"])
        command(net + ["ip", "link", "set", "lo", "up"])
        command(net + ["ip", version, "route", "add", "default", "via", local])
        # Exercise DNS filtering, not neighbor-discovery timing on fresh links.
        router_mac = json.loads(command(["ip", "-j", "link", "show", interface]))[0]["address"]
        peer_mac = json.loads(command(net + ["ip", "-j", "link", "show", "peer"]))[0]["address"]
        command(
            net + ["ip", version, "neigh", "replace", local, "lladdr", router_mac, "nud", "permanent", "dev", "peer"]
        )
        for address in addresses:
            command(
                ["ip", version, "neigh", "replace", address, "lladdr", peer_mac, "nud", "permanent", "dev", interface]
            )
    command(["ip", "link", "set", "lo", "up"])
    Path("/proc/sys/net/ipv6/conf/all/forwarding" if ipv6 else "/proc/sys/net/ipv4/ip_forward").write_text("1")
    process = subprocess.Popen(
        ["ip", "netns", "exec", "dns", sys.executable, __file__, "--dns-server", str(int(ipv6))],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout.readline().strip() == "ready"
        command(["ip", "netns", "exec", "client", "ping", version, "-c", "1", "-W", "2", allowed])
        for resolvers, expected in ((None, [True] * 4), ([allowed], [True, True, False, False]), ([], [False] * 4)):
            rules = "table inet cvm {}\ndelete table inet cvm\n" + firewall_rules([], [53], resolvers=resolvers)
            command(["nft", "-f", "-"], input=rules)
            for namespace in (None, "client"):
                before = ["ip", "netns", "exec", namespace] if namespace else []
                result = command(before + [sys.executable, __file__, "--dns-client", str(int(ipv6)), allowed, other])
                assert json.loads(result) == expected, (resolvers, namespace, result)
    finally:
        process.kill()
        process.wait(timeout=5)
        for namespace in ("client", "dns"):
            command(["ip", "netns", "delete", namespace])


def stalled_supervisor(pidfile):
    workload = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    Path(pidfile).write_text(str(workload.pid))
    audit._write = lambda decision: time.sleep(60)
    audit.emit("deny")
    notify("READY=1")
    watchdog(2)
    time.sleep(60)


@unittest.skipUnless(
    os.environ.get("CVM_REVIEW_TESTS") == "1" and os.geteuid() == 0, "Opt-in root review boundary tests"
)
class LinuxReviewBoundaryTests(unittest.TestCase):
    def test_confined_acceptance_agent_reads_verified_firewall_status(self):
        source = Path(bootstrap.__file__).resolve().parents[2]
        unit = "cvm-review-agent-" + uuid.uuid4().hex + ".service"
        # /run stays visible with PrivateTmp=yes, unlike /tmp. No guest state or
        # host firewall is changed: the table lives in a disposable namespace.
        with tempfile.TemporaryDirectory(prefix="cvm-review-", dir="/run") as directory:
            keeper = subprocess.Popen(
                ["unshare", "--net", sys.executable, __file__, "--firewall-namespace", directory],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                self.assertTrue(select.select([keeper.stdout], [], [], 10)[0], "Namespace setup timed out")
                self.assertEqual(keeper.stdout.readline().strip(), "ready")
                properties = [
                    line.replace("ReadOnlyPaths=/run/cvm", "ReadOnlyPaths=" + directory)
                    for line in bootstrap.SERVICE_HARDENING
                    if not line.startswith("ReadWritePaths=")
                ]
                properties += [
                    "NetworkNamespacePath=/proc/" + str(keeper.pid) + "/ns/net",
                    "RuntimeMaxSec=20s",
                    "FailureAction=none",
                    "SuccessAction=none",
                ]
                result = subprocess.run(
                    [
                        "systemd-run",
                        "--quiet",
                        "--wait",
                        "--pipe",
                        "--collect",
                        "--unit=" + unit,
                        *("--property=" + prop for prop in properties),
                        "--setenv=PYTHONPATH=" + str(source),
                        sys.executable,
                        __file__,
                        "--confined-firewall-agent",
                        directory,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("status writes denied", result.stdout)
            finally:
                subprocess.run(["systemctl", "stop", unit], capture_output=True, timeout=10)
                subprocess.run(["systemctl", "reset-failed", unit], capture_output=True, timeout=10)
                keeper.terminate()
                keeper.communicate(timeout=10)

    def test_real_ext4_mount_and_hostile_nfs_target(self):
        command(["unshare", "--mount", sys.executable, __file__, "--mount-probe"])

    def test_dns_denial_covers_output_forward_ipv4_ipv6_and_existing_conntrack(self):
        for ipv6 in (False, True):
            with self.subTest(ipv6=ipv6):
                command(["unshare", "--mount", "--net", sys.executable, __file__, "--dns-probe", str(int(ipv6))])

    def test_pid1_watchdog_kills_stalled_supervisor_and_its_workload(self):
        source = Path(bootstrap.__file__).resolve().parents[2]
        settings = configparser.ConfigParser()
        settings.read(source / "services/cvm_bootstrap.service")
        unit = "cvm-review-watchdog-" + uuid.uuid4().hex + ".service"
        with tempfile.TemporaryDirectory() as directory:
            pidfile = Path(directory) / "workload.pid"
            started = time.monotonic()
            try:
                command(
                    [
                        "systemd-run",
                        "--unit",
                        unit,
                        "--service-type=notify",
                        "--property=NotifyAccess=main",
                        "--property=WatchdogSec=2s",
                        "--property=WatchdogSignal=" + settings["Service"]["WatchdogSignal"],
                        "--property=TimeoutAbortSec=" + settings["Service"]["TimeoutAbortSec"],
                        "--property=RuntimeMaxSec=10s",
                        "--property=TimeoutStartSec=5s",
                        "--property=StandardOutput=null",
                        "--property=StandardError=null",
                        "--property=FailureAction=none",
                        "--property=SuccessAction=none",
                        "--setenv=PYTHONPATH=" + str(source),
                        sys.executable,
                        __file__,
                        "--stalled-supervisor",
                        str(pidfile),
                    ]
                )
                while time.monotonic() - started < 8:
                    result = command(["systemctl", "show", unit, "--property=Result", "--value"]).strip()
                    if result == "watchdog":
                        break
                    time.sleep(0.1)
                self.assertEqual(result, "watchdog")
                self.assertTrue(pidfile.is_file())
                pid = int(pidfile.read_text())
                for _ in range(20):
                    if not Path(f"/proc/{pid}").exists():
                        break
                    time.sleep(0.1)
                self.assertFalse(Path(f"/proc/{pid}").exists(), "PID 1 must kill the workload cgroup")
            finally:
                subprocess.run(["systemctl", "stop", unit], capture_output=True, timeout=10)
                subprocess.run(["systemctl", "reset-failed", unit], capture_output=True, timeout=10)


if __name__ == "__main__":
    action = sys.argv[1:2]
    if action == ["--firewall-namespace"]:
        firewall_namespace(sys.argv[2])
    elif action == ["--confined-firewall-agent"]:
        confined_firewall_agent(sys.argv[2])
    elif action == ["--mount-probe"]:
        mount_probe()
    elif action == ["--dns-server"]:
        dns_server(bool(int(sys.argv[2])))
    elif action == ["--dns-client"]:
        dns_client(bool(int(sys.argv[2])), sys.argv[3:])
    elif action == ["--dns-probe"]:
        dns_probe(bool(int(sys.argv[2])))
    elif action == ["--stalled-supervisor"]:
        stalled_supervisor(sys.argv[2])
    else:
        unittest.main(verbosity=2)
