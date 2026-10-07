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

"""Opt-in packet tests: all interfaces, routing and rules live in disposable namespaces."""

import json
import os
import socket
import subprocess
import sys
import unittest
from pathlib import Path

from cvm.common.firewall import firewall_rules


def command(argv, **kwargs):
    return subprocess.run(argv, check=True, capture_output=True, text=True, timeout=10, **kwargs).stdout


def peer(role, ipv6):
    """Hold a peer namespace until configured, then listen or probe both paths."""
    print("ready", flush=True)
    settings = json.loads(sys.stdin.readline())
    family = socket.AF_INET6 if ipv6 else socket.AF_INET
    if role == "server":
        with socket.socket(family, socket.SOCK_STREAM) as server:
            server.bind((settings["address"], 8081))
            server.listen(16)
            print("listening", flush=True)
            sys.stdin.readline()
    else:
        results = []
        for source in settings["sources"]:
            connected = []
            for port in (9090, 8080):
                with socket.socket(family, socket.SOCK_STREAM) as client:
                    client.settimeout(0.5)
                    client.bind((source, 0))
                    try:
                        client.connect((settings["gateway"], port))
                        connected.append(True)
                    except TimeoutError:
                        connected.append(False)
            results.append(connected)
        print(json.dumps(results), flush=True)


def packet_probe(ipv6, restricted):
    """Run only in the unshare-created router namespace, never on the host."""
    assert os.readlink("/proc/self/ns/net") != os.readlink(f"/proc/{os.getppid()}/ns/net")
    if ipv6:
        sources = ["2001:db8:1::2", "2001:db8:1::102"]
        gateway, backend, bridge, bits = "2001:db8:1::1", "2001:db8:2::2", "2001:db8:2::1", "64"
        allowed = ["2001:db8:1::/120"]
    else:
        sources = ["198.51.100.2", "198.51.100.130"]
        gateway, backend, bridge, bits = "198.51.100.1", "172.18.0.2", "172.18.0.1", "24"
        allowed = ["198.51.100.0/25"]
    version = "-6" if ipv6 else "-4"
    children = []
    try:
        for role, name, addresses, route in (
            ("client", "wan", sources, gateway),
            ("server", "docker0", [backend], bridge),
        ):
            child = subprocess.Popen(
                ["unshare", "--net", sys.executable, __file__, "--peer", role, str(int(ipv6))],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            children.append(child)
            assert child.stdout.readline().strip() == "ready"
            command(["ip", "link", "add", name, "type", "veth", "peer", "name", name + "peer"])
            command(["ip", "link", "set", name + "peer", "netns", str(child.pid)])
            command(["ip", version, "addr", "add", route + "/" + bits, "dev", name, "nodad"])
            command(["ip", "link", "set", name, "up"])
            prefix = ["nsenter", "-t", str(child.pid), "-n"]
            command(prefix + ["ip", "link", "set", name + "peer", "name", "eth0"])
            for address in addresses:
                command(prefix + ["ip", version, "addr", "add", address + "/" + bits, "dev", "eth0", "nodad"])
            command(prefix + ["ip", "link", "set", "eth0", "up"])
            command(prefix + ["ip", "link", "set", "lo", "up"])
            command(prefix + ["ip", version, "route", "add", "default", "via", route])
            # Keep neighbor discovery latency out of the short TCP timeout.
            # These tests exercise source filtering, not ARP/NDP behavior.
            peer_mac = json.loads(command(prefix + ["ip", "-j", "link", "show", "eth0"]))[0]["address"]
            router_mac = json.loads(command(["ip", "-j", "link", "show", name]))[0]["address"]
            command(
                prefix
                + ["ip", version, "neigh", "replace", route, "lladdr", router_mac, "nud", "permanent", "dev", "eth0"]
            )
            for address in addresses:
                command(
                    ["ip", version, "neigh", "replace", address, "lladdr", peer_mac, "nud", "permanent", "dev", name]
                )
        command(["ip", "link", "set", "lo", "up"])
        forwarding = "/proc/sys/net/ipv6/conf/all/forwarding" if ipv6 else "/proc/sys/net/ipv4/ip_forward"
        Path(forwarding).write_text("1")
        rules = firewall_rules(
            [8080, 9090], [], [{"host": 8080, "container": 8081}], inbound_sources=allowed if restricted else []
        )
        command(["nft", "--check", "-f", "-"], input=rules)
        command(["nft", "-f", "-"], input=rules)
        family = "ip6" if ipv6 else "ip"
        destination = f"[{backend}]:8081" if ipv6 else f"{backend}:8081"
        nat = (
            f"table {family} nat {{\nchain prerouting {{\n type nat hook prerouting priority dstnat;\n"
            f'iifname "wan" tcp dport 8080 dnat to {destination}\n}}\n}}\n'
        )
        command(["nft", "-f", "-"], input=nat)
        with socket.socket(socket.AF_INET6 if ipv6 else socket.AF_INET, socket.SOCK_STREAM) as host:
            host.bind((gateway, 9090))
            host.listen(16)
            client, server = children
            server.stdin.write(json.dumps({"address": backend}) + "\n")
            server.stdin.flush()
            assert server.stdout.readline().strip() == "listening"
            client.stdin.write(json.dumps({"gateway": gateway, "sources": sources}) + "\n")
            client.stdin.flush()
            output, errors = client.communicate(timeout=10)
            assert client.returncode == 0, errors
            print(output.strip(), flush=True)
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=5)


@unittest.skipUnless(os.environ.get("CVM_NETWORK_TESTS") == "1" and os.geteuid() == 0, "Opt-in root network tests")
class FirewallPacketTests(unittest.TestCase):
    def test_inbound_cidrs_protect_host_and_dnat_paths(self):
        for ipv6 in (False, True):
            for restricted in (False, True):
                with self.subTest(ipv6=ipv6, restricted=restricted):
                    result = subprocess.run(
                        ["unshare", "--net", sys.executable, __file__, "--probe", str(int(ipv6)), str(int(restricted))],
                        capture_output=True,
                        text=True,
                        timeout=45,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(json.loads(result.stdout), [[True, True], [not restricted, not restricted]])


if __name__ == "__main__":
    if sys.argv[1:2] == ["--peer"]:
        peer(sys.argv[2], bool(int(sys.argv[3])))
    elif sys.argv[1:2] == ["--probe"]:
        packet_probe(bool(int(sys.argv[2])), bool(int(sys.argv[3])))
    else:
        unittest.main(verbosity=2)
