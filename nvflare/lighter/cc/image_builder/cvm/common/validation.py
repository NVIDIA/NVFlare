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

"""Shared application configuration validation and runtime projection."""

import ipaddress
import re
from pathlib import PurePosixPath

from .errors import require

# Docker's default capability set minus the ones a confined application never
# needs: raw sockets, device nodes, chroot, audit writes and file capabilities.
DEFAULT_CAPABILITIES = (
    "CHOWN",
    "DAC_OVERRIDE",
    "FOWNER",
    "FSETID",
    "KILL",
    "NET_BIND_SERVICE",
    "SETGID",
    "SETPCAP",
    "SETUID",
)


# Capabilities an application may request explicitly. Anything that would let the
# container reach the kernel, devices or other namespaces stays unavailable.
ALLOWED_CAPABILITIES = frozenset(
    DEFAULT_CAPABILITIES
    + (
        "AUDIT_WRITE",
        "IPC_LOCK",
        "MKNOD",
        "NET_RAW",
        "SETFCAP",
        "SYS_CHROOT",
        "SYS_NICE",
        "SYS_RESOURCE",
    )
)


DEFAULT_PIDS_LIMIT = 4096


RUNTIME_KEYS = (
    "image_id",
    "container",
    "hosts_entries",
    "requires_gpu",
    "allowed_ports",
    "allowed_out_ports",
)


OPTIONAL_RUNTIME_KEYS = ("nfs_mount", "allowed_in_cidrs", "allowed_out_cidrs")


def ports(values):
    require(isinstance(values, list), "Ports must be a list")
    require(all(type(p) is int and 1 <= p <= 65535 for p in values), "Invalid port")
    require(len(values) == len(set(values)), "Duplicate port")
    return values


def cidrs(values):
    """Validate a list of canonical IPv4/IPv6 network prefixes."""
    require(isinstance(values, list), "Address allowlists must be a list of CIDR strings")
    for value in values:
        require(isinstance(value, str) and "/" in value, "Address allowlist entries must be CIDR strings")
        try:
            network = ipaddress.ip_network(value, strict=True)
        except ValueError:
            require(False, "Invalid CIDR in address allowlist")
        require(str(network) == value, "Address allowlist entries must be canonical CIDR strings")
    require(len(values) == len(set(values)), "Duplicate CIDR in address allowlist")
    return values


def capabilities(values):
    require(isinstance(values, list), "container.capabilities must be a list")
    for value in values:
        require(isinstance(value, str) and value in ALLOWED_CAPABILITIES, "Unsupported container capability")
    require(len(values) == len(set(values)), "Duplicate container capability")
    return values


def runtime_config(value):
    result = {key: value[key] for key in RUNTIME_KEYS}
    for key in OPTIONAL_RUNTIME_KEYS:
        if value.get(key) is not None:
            result[key] = value[key]
    return result


def validate_nfs_mount(value):
    require(
        isinstance(value, dict) and set(value) == {"server", "export", "security"},
        "nfs_mount requires server, export, security",
    )
    require(value["security"] == "krb5p", "NFS requires authenticated and encrypted Kerberos transport (krb5p)")
    require(
        isinstance(value["server"], str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", value["server"]),
        "Invalid NFS server",
    )
    path = PurePosixPath(value["export"])
    require(
        path.is_absolute() and ".." not in path.parts and re.fullmatch(r"/[A-Za-z0-9_./-]*", value["export"]),
        "Invalid NFS export",
    )
