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

"""The catalog's rules in the site dataset configuration: per-dataset publication rules and site-wide settings.

The dataset configuration API returns them as plain mappings (``usage.catalog`` of a dataset, and the top-level
``usage.catalog``); this module validates them and applies a dataset's rules to its catalog.

This module has no FLARE imports.
"""

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from nvflare.app_common.fdn.catalog.common.schema import (
    DEFAULT_LIMITS,
    CatalogInfo,
    Provenance,
    ReadLimits,
    SemanticProfile,
    TableInfo,
)

_logger = logging.getLogger(__name__)
CATALOG_COMPONENT = "catalog"  # the component name under `usage` in the dataset configuration
_RULE_KEYS = frozenset(
    {"share", "exclude_tables", "exclude_columns", "exclude_resources", "exclude_metadata", "operations", "connector"}
)
_SETTINGS_KEYS = frozenset({"limits", "connectors", "installed_operations"})
_CONNECTOR_KEYS = frozenset({"class", "args"})
_CONNECTOR_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_EXCLUDABLE_METADATA = frozenset(
    {"description", "owner", "modalities", "semantic_profile", "source_version", "lineage", "provenance"}
)


class RulesError(Exception):
    """The catalog's rules or settings are invalid. The message names fields, never values or keys a site typed:
    those can be private table, column or dataset names."""


@dataclass(frozen=True)
class DatasetRule:
    share: bool
    exclude_tables: frozenset[str] = frozenset()
    exclude_columns: Mapping[str, frozenset[str]] = field(default_factory=dict)
    exclude_resources: frozenset[str] = frozenset()
    exclude_metadata: frozenset[str] = frozenset()
    operations: frozenset[str] = frozenset()
    connector: str | None = None  # overrides the connector the dataset's type selects


@dataclass(frozen=True)
class ConnectorSpec:
    path: str  # a CatalogConnector class path
    args: Mapping = field(default_factory=dict)  # non-secret constructor arguments


@dataclass(frozen=True)
class CatalogSettings:
    limits: ReadLimits = field(default_factory=lambda: ReadLimits(**DEFAULT_LIMITS))
    connectors: Mapping[str, ConnectorSpec] = field(default_factory=dict)  # site-declared, in addition to built-ins
    installed_operations: frozenset[str] = frozenset()


def _name_set(value, label: str) -> frozenset[str]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(x, str) and x for x in value):
        raise RulesError(f"{label} must be a list of non-empty strings")
    return frozenset(value)


def parse_rule(value, label: str = "usage.catalog") -> DatasetRule:
    """Validate one dataset's catalog rules."""
    if not isinstance(value, Mapping):
        raise RulesError(f"{label} must be a mapping")
    unknown = set(value) - _RULE_KEYS
    if unknown:
        raise RulesError(f"{label} has {len(unknown)} unsupported key(s)")
    share = value.get("share")
    if not isinstance(share, bool):
        raise RulesError(f"{label}.share is required and must be a boolean")
    columns = value.get("exclude_columns", {})
    if not isinstance(columns, Mapping):
        raise RulesError(f"{label}.exclude_columns must be a mapping")
    metadata = _name_set(value.get("exclude_metadata", []), f"{label}.exclude_metadata")
    if not metadata <= _EXCLUDABLE_METADATA:
        raise RulesError(f"{label}.exclude_metadata accepts only {sorted(_EXCLUDABLE_METADATA)}")
    connector = value.get("connector")
    if connector is not None and (not isinstance(connector, str) or not _CONNECTOR_NAME.match(connector)):
        raise RulesError(f"{label}.connector must be a connector name")
    return DatasetRule(
        share=share,
        exclude_tables=_name_set(value.get("exclude_tables", []), f"{label}.exclude_tables"),
        exclude_columns={t: _name_set(c, f"{label}.exclude_columns entries") for t, c in columns.items()},
        exclude_resources=_name_set(value.get("exclude_resources", []), f"{label}.exclude_resources"),
        exclude_metadata=metadata,
        operations=_name_set(value.get("operations", []), f"{label}.operations"),
        connector=connector,
    )


def parse_settings(value) -> CatalogSettings:
    """Validate the site-wide catalog settings; None means all defaults."""
    if value is None:
        return CatalogSettings()
    if not isinstance(value, Mapping):
        raise RulesError("usage.catalog must be a mapping")
    unknown = set(value) - _SETTINGS_KEYS
    if unknown:
        raise RulesError(f"usage.catalog has {len(unknown)} unsupported key(s)")
    limits = value.get("limits", {})
    if (
        not isinstance(limits, Mapping)
        or set(limits) - set(DEFAULT_LIMITS)
        or not all(type(v) is int and v > 0 for v in limits.values())
    ):
        raise RulesError(f"usage.catalog.limits must map {sorted(DEFAULT_LIMITS)} to positive integers")
    connectors = value.get("connectors", {})
    if not isinstance(connectors, Mapping):
        raise RulesError("usage.catalog.connectors must be a mapping")
    specs = {}
    for name, spec in connectors.items():
        if not isinstance(name, str) or not _CONNECTOR_NAME.match(name):
            raise RulesError("usage.catalog.connectors has an invalid connector name")
        if not isinstance(spec, Mapping) or set(spec) - _CONNECTOR_KEYS:
            raise RulesError(f"usage.catalog.connectors.{name} must have only {sorted(_CONNECTOR_KEYS)}")
        class_path, args = spec.get("class"), spec.get("args", {})
        if not isinstance(class_path, str) or not class_path:
            raise RulesError(f"usage.catalog.connectors.{name} needs a class path")
        if not isinstance(args, Mapping):
            raise RulesError(f"usage.catalog.connectors.{name}.args must be a mapping")
        specs[name] = ConnectorSpec(class_path, dict(args))
    return CatalogSettings(
        limits=ReadLimits(**{**DEFAULT_LIMITS, **limits}),
        connectors=specs,
        installed_operations=_name_set(value.get("installed_operations", []), "usage.catalog.installed_operations"),
    )


def _warn_unmatched(rule: DatasetRule, catalog: CatalogInfo, removed_tables: set[str]) -> None:
    """An exclusion that matches nothing excludes nothing, which is how a typo shares what the operator meant to
    hide. Say so, with a count only: job logs can be streamed to the server, and the names in a policy are the
    site's own, so the warning never repeats them."""
    tables = {t.name.casefold(): t for t in catalog.tables}
    unmatched = sum(1 for name in rule.exclude_tables if name.casefold() not in tables)
    for table, columns in rule.exclude_columns.items():
        found = tables.get(table.casefold())
        if found is None:
            unmatched += 1
        elif found.name not in removed_tables:
            present = {c.name.casefold() for c in found.columns}
            unmatched += sum(1 for column in columns if column.casefold() not in present)
    if unmatched:
        _logger.warning(
            "catalog rules for dataset %r: %d exclusion(s) match nothing in the catalog and exclude nothing; "
            "check usage.catalog in the site dataset configuration",
            catalog.dataset,
            unmatched,
        )


def apply_policy(rule: DatasetRule, catalog: CatalogInfo, installed_ops: set[str]) -> CatalogInfo:
    """The shareable view of a catalog. Returns a new object; the caller validates it.

    Names are matched without regard to case, so a differently cased entry excludes at least what it names."""
    excluded_tables = {n.casefold() for n in rule.exclude_tables}
    excluded_columns: dict[str, set[str]] = {}
    for table, columns in rule.exclude_columns.items():
        excluded_columns.setdefault(table.casefold(), set()).update(c.casefold() for c in columns)
    excluded_resources = {r.casefold() for r in rule.exclude_resources}
    removed_tables = {t.name for t in catalog.tables if t.name.casefold() in excluded_tables}
    removed_columns = {
        t.name: {c.name for c in t.columns if c.name.casefold() in excluded_columns.get(t.name.casefold(), ())}
        for t in catalog.tables
        if t.name not in removed_tables
    }
    _warn_unmatched(rule, catalog, removed_tables)
    removed_folded = {n.casefold() for n in removed_tables}
    removed_columns_folded = {t.casefold(): c for t, c in removed_columns.items()}

    tables = []
    for t in catalog.tables:
        if t.name in removed_tables:
            continue
        gone = removed_columns[t.name]
        relationships = tuple(
            r
            for r in t.relationships
            if not (set(r.columns) & gone)
            and r.target_table.casefold() not in removed_folded
            and not (set(r.target_columns) & removed_columns_folded.get(r.target_table.casefold(), set()))
        )
        tables.append(
            TableInfo(
                name=t.name,
                columns=tuple(c for c in t.columns if c.name not in gone),
                primary_key=() if set(t.primary_key) & gone else t.primary_key,
                relationships=relationships,
            )
        )

    excluded = rule.exclude_metadata
    return replace(
        catalog,
        tables=tuple(tables),
        resources=tuple(r for r in catalog.resources if r.type.casefold() not in excluded_resources),
        description=None if "description" in excluded else catalog.description,
        owner=None if "owner" in excluded else catalog.owner,
        modalities=() if "modalities" in excluded else catalog.modalities,
        semantic_profile=SemanticProfile() if "semantic_profile" in excluded else catalog.semantic_profile,
        source_version=None if "source_version" in excluded else catalog.source_version,
        lineage=() if "lineage" in excluded else catalog.lineage,
        provenance=(Provenance(source=catalog.provenance.source) if "provenance" in excluded else catalog.provenance),
        operations=tuple(sorted(set(catalog.operations) & rule.operations & set(installed_ops))),
    )
