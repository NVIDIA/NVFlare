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

"""Site-wide catalog read limits.

Dataset publication is controlled by the source administrator's credentials, schemas and file access.
This module has no FLARE imports.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from nvflare.app_common.fdn.catalog.common.limits import DEFAULT_LIMITS, ReadLimits

CATALOG_COMPONENT = "catalog"


class SettingsError(Exception):
    """Site-wide catalog settings are invalid."""


@dataclass(frozen=True)
class CatalogSettings:
    limits: ReadLimits = field(default_factory=lambda: ReadLimits(**DEFAULT_LIMITS))


def parse_settings(value) -> CatalogSettings:
    """Validate the site-wide catalog settings; None means all defaults."""
    if value is None:
        return CatalogSettings()
    if not isinstance(value, Mapping):
        raise SettingsError("usage.catalog must be a mapping")
    unknown = set(value) - {"limits"}
    if unknown:
        raise SettingsError(f"usage.catalog has {len(unknown)} unsupported key(s)")
    limits = value.get("limits", {})
    if (
        not isinstance(limits, Mapping)
        or set(limits) - set(DEFAULT_LIMITS)
        or not all(type(v) is int and v > 0 for v in limits.values())
    ):
        raise SettingsError(f"usage.catalog.limits must map {sorted(DEFAULT_LIMITS)} to positive integers")
    return CatalogSettings(limits=ReadLimits(**{**DEFAULT_LIMITS, **limits}))
