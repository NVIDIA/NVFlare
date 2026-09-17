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

"""Shared cert/package constants for distributed provisioning CLI flows."""

from nvflare.lighter.constants import AdminRole

CA_INFO_FIELD = "ca_info"
DEFAULT_PROVISION_VERSION = "00"
PROVISION_VERSION_FIELD = "provision_version"
ROOTCA_FINGERPRINT_FIELD = "rootCA_fingerprint"

VALID_CERT_TYPES = ("client", "server", "org_admin", "lead", "member")
ADMIN_CERT_TYPES = (AdminRole.ORG_ADMIN, AdminRole.LEAD, AdminRole.MEMBER)
KIT_TYPE_TO_ROLE = {
    "org_admin": AdminRole.ORG_ADMIN,
    "lead": AdminRole.LEAD,
    "member": AdminRole.MEMBER,
}


def is_valid_provision_version(value: str) -> bool:
    return isinstance(value, str) and value.isascii() and len(value) == 2 and value.isdigit()
