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

"""Actionable, non-secret errors shared by CVM tools."""


class BuildError(Exception):
    """An actionable, non-secret diagnostic safe to print."""


class ConfigurationError(BuildError):
    """A configuration diagnostic containing schema names, never input values."""


class KBSRequestError(BuildError):
    """A classified transport failure with a separate secret-free CLI summary."""

    def __init__(self, message, category, status=None):
        if category not in ("connection", "tls", "http"):
            raise ValueError("Invalid KBS failure category")
        if status is not None and (type(status) is not int or not 100 <= status <= 599):
            raise ValueError("Invalid KBS HTTP status")
        self.category = category
        self.status = status
        super().__init__(message)

    @property
    def diagnostic(self):
        if self.category == "http":
            return f"KBS HTTP {self.status}" if self.status is not None else "KBS HTTP failure"
        return "KBS TLS verification failure" if self.category == "tls" else "KBS connection failure"


STARTUP_PHASES = frozenset(
    (
        "configuration",
        "network",
        "time_sync",
        "verity",
        "attestation",
        "key_release",
        "vault_identity",
        "payload_authentication",
        "supervision",
    )
)


class StartupPhaseError(BuildError):
    """A fixed startup phase identifier that never includes untrusted data."""

    def __init__(self, phase):
        if phase not in STARTUP_PHASES:
            raise ValueError("Invalid startup phase")
        self.phase = phase
        super().__init__(phase)


def require_config(condition, message):
    """Use only fixed messages or schema-defined field names, never user values."""
    if not condition:
        raise ConfigurationError(message)


def require(condition, message):
    if not condition:
        raise BuildError(message)
