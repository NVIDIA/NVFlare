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

import ipaddress

from .errors import require
from .validation import cidrs, ports

# Only the ICMP types a client needs: reachability probes and path errors.
ICMP_TYPES = "echo-request, echo-reply, destination-unreachable, time-exceeded, parameter-problem"


def _split(values):
    """Return (ipv4, ipv6) address or prefix lists for nft set literals."""
    v4, v6 = [], []
    for value in values:
        network = ipaddress.ip_network(value, strict=False)
        (v4 if network.version == 4 else v6).append(str(network))
    return v4, v6


def _set(values):
    return "{ " + ", ".join(values) + " }"


def _restricted(rules, addresses, selector, match, *, prefix=""):
    """Emit one accept rule per address family; no addresses means any destination."""
    if not addresses:
        rules.append(f"{prefix}{match} accept")
        return
    v4, v6 = _split(addresses)
    if v4:
        rules.append(f"{prefix}ip {selector} {_set(v4)} {match} accept")
    if v6:
        rules.append(f"{prefix}ip6 {selector} {_set(v6)} {match} accept")


def _dns(rules, resolvers, *, prefix=""):
    # None is an explicit construction-time discovery allowance. An empty
    # runtime list denies DNS, including previously established connections and
    # configurations that also include port 53 in the general TCP allowlist.
    for protocol in ("udp", "tcp"):
        if resolvers is None or resolvers:
            _restricted(rules, resolvers, "daddr", f"{protocol} dport 53", prefix=prefix)
        rules.append(f"{prefix}{protocol} dport 53 drop")


def firewall_rules(inbound, outbound, mappings=(), *, inbound_sources=(), outbound_destinations=(), resolvers=()):
    """Render the guest table.

    inbound/outbound are TCP port allowlists. inbound_sources and
    outbound_destinations optionally restrict those ports to CIDR lists.
    resolvers restricts DNS to the given server addresses; an empty list denies
    DNS. Only the measured discovery rules use None before DHCP is available.
    """
    ports(inbound)
    ports(outbound)
    cidrs(list(inbound_sources))
    cidrs(list(outbound_destinations))
    for resolver in resolvers or ():
        require(isinstance(resolver, str), "Resolver addresses must be strings")
        try:
            address = ipaddress.ip_address(resolver)
        except ValueError:
            require(False, "Invalid resolver address")
        require(
            not (address.is_loopback or address.is_unspecified or address.is_multicast or address.is_reserved),
            "Resolver address is not routable",
        )
    # A dedicated inet table, IPv4 and IPv6. The forward chain applies the same
    # restrictions to Docker's bridge, before Docker's own permissive chains.
    rules = [
        "table inet cvm {",
        "chain input { type filter hook input priority -10; policy drop;",
        'iifname "lo" accept',
        "ct state invalid drop",
        "ct state established,related accept",
        f"icmp type {{ {ICMP_TYPES} }} accept",
        "ip6 nexthdr ipv6-icmp accept",
        "udp sport 67 udp dport 68 accept",
    ]
    if inbound:
        _restricted(rules, inbound_sources, "saddr", "tcp dport " + _set(map(str, inbound)))
    rules += [
        "}",
        "chain output { type filter hook output priority -10; policy drop;",
        'oifname "lo" accept',
        "ct state invalid drop",
    ]
    _dns(rules, resolvers)
    rules += ["ct state established,related accept", "udp dport { 67, 547 } accept"]
    # Ubuntu's chrony defaults use NTS key exchange before NTP traffic.
    # Keep this host time-service allowance through the application rules.
    rules += [
        "udp dport 123 accept",
        "tcp dport 4460 accept",
        f"icmp type {{ {ICMP_TYPES} }} accept",
        "ip6 nexthdr ipv6-icmp accept",
    ]
    if outbound:
        _restricted(rules, outbound_destinations, "daddr", "tcp dport " + _set(map(str, outbound)))
    rules += [
        "}",
        "chain forward { type filter hook forward priority -10; policy drop;",
        "ct state invalid drop",
    ]
    _dns(rules, resolvers, prefix='iifname "docker0" ')
    rules.append("ct state established,related accept")
    rules.append('iifname "docker0" udp dport 123 accept')
    if outbound:
        _restricted(
            rules,
            outbound_destinations,
            "daddr",
            "tcp dport " + _set(map(str, outbound)),
            prefix='iifname "docker0" ',
        )
    for mapping in mappings:
        require(mapping["host"] in inbound, "Container port is not allowed")
        ports([mapping["container"]])
        _restricted(
            rules,
            inbound_sources,
            "saddr",
            f'tcp dport {mapping["container"]} ct original proto-dst {mapping["host"]}',
            prefix='oifname "docker0" ',
        )
    rules += ["}", "}"]
    return "\n".join(rules) + "\n"
