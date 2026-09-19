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

"""Render measured bootstrap and authenticated application nftables rules."""

from .errors import require
from .validation import ports


def firewall_rules(inbound, outbound, mappings=()):

    ports(inbound)
    ports(outbound)
    # A dedicated inet table, IPv4 and IPv6. The forward chain applies the same
    # restrictions to Docker's bridge, before Docker's own permissive chains.
    rules = [
        "table inet cvm {",
        "chain input { type filter hook input priority -10; policy drop;",
        'iifname "lo" accept',
        "ct state established,related accept",
        "ip protocol icmp accept",
        "ip6 nexthdr ipv6-icmp accept",
        "udp sport 67 udp dport 68 accept",
    ]
    if inbound:
        rules.append("tcp dport { " + ",".join(map(str, inbound)) + " } accept")
    rules += [
        "}",
        "chain output { type filter hook output priority -10; policy drop;",
        'oifname "lo" accept',
        "ct state established,related accept",
        "udp dport { 53,67,123,547 } accept",
        # Ubuntu's chrony defaults use NTS key exchange before NTP traffic.
        # Keep this host time-service allowance through the application rules.
        "tcp dport { 53,4460 } accept",
        "ip protocol icmp accept",
        "ip6 nexthdr ipv6-icmp accept",
    ]
    if outbound:
        rules.append("tcp dport { " + ",".join(map(str, outbound)) + " } accept")
    rules += [
        "}",
        "chain forward { type filter hook forward priority -10; policy drop;",
        "ct state established,related accept",
        'iifname "docker0" udp dport { 53,123 } accept',
        'iifname "docker0" tcp dport 53 accept',
    ]
    if outbound:
        rules.append('iifname "docker0" tcp dport { ' + ",".join(map(str, outbound)) + " } accept")
    for mapping in mappings:
        require(mapping["host"] in inbound, "Container port is not allowed")
        ports([mapping["container"]])
        rules.append(
            f'oifname "docker0" tcp dport {mapping["container"]} ct original proto-dst {mapping["host"]} accept'
        )
    rules += ["}", "}"]
    return "\n".join(rules) + "\n"
