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

"""Validate Dashboard identities using the same rules as provisioning."""

from nvflare.apis.utils.format_check import name_check


class NameValidationError(ValueError):
    def __init__(self, errors):
        self.errors = errors
        super().__init__(" ".join(errors.values()))


CLIENT_NAMES = {"name": "client", "organization": "org"}
USER_NAMES = {"email": "admin", "organization": "org", "role": "simple_name"}
SERVER_NAMES = {"server1": "server"}

_NAME_MESSAGES = {
    "client": "Client name must contain only letters, digits, hyphens, and underscores.",
    "org": "Organization must contain only letters, digits, and underscores.",
    "admin": "Email must be a valid email address for the admin startup kit.",
    "simple_name": "Role must contain only letters, digits, and underscores.",
    "server": "Server name must be a valid hostname or IPv4 address, without a scheme or port.",
}


def validate_names(values, fields, required=(), allow_empty=()):
    """Reject supplied invalid names before any database changes.

    Missing PATCH fields are left alone. Empty organization/role values are
    allowed only for the Dashboard's staged user registration flow; empty
    server names are allowed while configuring an unfrozen project.
    """
    errors = {}
    for field, entity_type in fields.items():
        if field not in values and field not in required:
            continue
        value = values.get(field)
        if field in allow_empty and value == "":
            continue
        if not isinstance(value, str) or name_check(value, entity_type)[0]:
            errors[field] = _NAME_MESSAGES[entity_type]
    if errors:
        raise NameValidationError(errors)
