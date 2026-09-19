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

import re
from pathlib import PurePosixPath

from .errors import require


def ports(values):
    require(isinstance(values, list), "Ports must be a list")
    require(all(type(p) is int and 1 <= p <= 65535 for p in values), "Invalid port")
    require(len(values) == len(set(values)), "Duplicate port")
    return values


def runtime_config(value):
    result = {
        key: value[key]
        for key in ("image_id", "container", "hosts_entries", "requires_gpu", "allowed_ports", "allowed_out_ports")
    }
    if value.get("nfs_mount") is not None:
        result["nfs_mount"] = value["nfs_mount"]
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
