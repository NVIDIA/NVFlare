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

"""Site-wide catalog connector settings, read limits and installed capabilities.

Dataset publication is controlled by the source administrator's credentials, schemas and file access.
This module has no FLARE imports.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from nvflare.app_common.fdn.catalog.common.schema import DEFAULT_LIMITS, ReadLimits

CATALOG_COMPONENT = "catalog"
_SETTINGS_KEYS = frozenset({"limits", "connectors", "installed_operations"})
_CONNECTOR_KEYS = frozenset({"class", "args"})
_CONNECTOR_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class SettingsError(Exception):
    """Site-wide catalog settings are invalid."""


@dataclass(frozen=True)
class ConnectorSpec:
    path: str
    args: Mapping = field(default_factory=dict)


@dataclass(frozen=True)
class CatalogSettings:
    limits: ReadLimits = field(default_factory=lambda: ReadLimits(**DEFAULT_LIMITS))
    connectors: Mapping[str, ConnectorSpec] = field(default_factory=dict)
    installed_operations: frozenset[str] = frozenset()


def _name_set(value, label: str) -> frozenset[str]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(x, str) and x for x in value):
        raise SettingsError(f"{label} must be a list of non-empty strings")
    return frozenset(value)


def parse_settings(value) -> CatalogSettings:
    """Validate the site-wide catalog settings; None means all defaults."""
    if value is None:
        return CatalogSettings()
    if not isinstance(value, Mapping):
        raise SettingsError("usage.catalog must be a mapping")
    unknown = set(value) - _SETTINGS_KEYS
    if unknown:
        raise SettingsError(f"usage.catalog has {len(unknown)} unsupported key(s)")
    limits = value.get("limits", {})
    if (
        not isinstance(limits, Mapping)
        or set(limits) - set(DEFAULT_LIMITS)
        or not all(type(v) is int and v > 0 for v in limits.values())
    ):
        raise SettingsError(f"usage.catalog.limits must map {sorted(DEFAULT_LIMITS)} to positive integers")
    connectors = value.get("connectors", {})
    if not isinstance(connectors, Mapping):
        raise SettingsError("usage.catalog.connectors must be a mapping")
    specs = {}
    for name, spec in connectors.items():
        if not isinstance(name, str) or not _CONNECTOR_NAME.match(name):
            raise SettingsError("usage.catalog.connectors has an invalid connector name")
        if not isinstance(spec, Mapping) or set(spec) - _CONNECTOR_KEYS:
            raise SettingsError(f"usage.catalog.connectors.{name} must have only {sorted(_CONNECTOR_KEYS)}")
        class_path, args = spec.get("class"), spec.get("args", {})
        if not isinstance(class_path, str) or not class_path:
            raise SettingsError(f"usage.catalog.connectors.{name} needs a class path")
        if not isinstance(args, Mapping):
            raise SettingsError(f"usage.catalog.connectors.{name}.args must be a mapping")
        specs[name] = ConnectorSpec(class_path, dict(args))
    return CatalogSettings(
        limits=ReadLimits(**{**DEFAULT_LIMITS, **limits}),
        connectors=specs,
        installed_operations=_name_set(value.get("installed_operations", []), "usage.catalog.installed_operations"),
    )
