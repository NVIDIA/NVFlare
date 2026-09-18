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

"""Shared storage layout, platform and resource-binding contracts."""

import base64
import hashlib
import re

from .errors import require

HEADER_BYTES = 16777216


STORAGE_PROFILE = "luks2-xts-random-hmac-sha256-v1"


PLATFORMS = ("amd_sev_snp", "intel_tdx")


DISK_ROLES = ("root", "applog", "user-config", "user-data", "vault")


ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")


def identifier(value):
    require(isinstance(value, str) and ID.fullmatch(value), "Invalid immutable identifier")
    return value


def binding(header):
    require(len(header) == HEADER_BYTES, "Truncated or unsupported vault header")
    return hashlib.sha256(b"nvflare-vault-v2\x00" + header).digest()


def binding_id(platform, value):
    require(platform in PLATFORMS and len(value) == 32, "Invalid platform/binding")
    if platform == "amd_sev_snp":
        return base64.urlsafe_b64encode(value).decode().rstrip("=")
    return (value + bytes(16)).hex()


def qemu_binding(platform, value):
    require(platform in PLATFORMS and len(value) == 32, "Invalid platform/binding")
    return base64.b64encode(value + (bytes(16) if platform == "intel_tdx" else b"")).decode()


def resource_path(build_id, platform, value):
    return f"keys/{identifier(build_id)}/{binding_id(platform, value)}"


def validate_resource(path):
    parts = path.split("/")
    require(len(parts) == 3 and parts[0] == "keys", "Invalid resource namespace")
    identifier(parts[1])
    tag = parts[2]
    snp = re.fullmatch(r"[A-Za-z0-9_-]{42}[AEIMQUYcgkosw048]", tag)
    tdx = re.fullmatch(r"[0-9a-f]{64}0{32}", tag)
    require(snp or tdx, "Noncanonical binding identifier")
    return parts
