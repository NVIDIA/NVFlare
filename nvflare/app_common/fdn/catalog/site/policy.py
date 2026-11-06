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

"""The site-owned Study publication policy: which parts of a dataset's catalog may be shared.

This module has no FLARE imports.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from nvflare.app_common.fdn.catalog.common.schema import CatalogInfo, Provenance, SemanticProfile, TableInfo

POLICY_FORMAT_VERSION = 1
_TOP_LEVEL_KEYS = frozenset({"format_version", "datasets"})
_RULE_KEYS = frozenset(
    {"share", "exclude_tables", "exclude_columns", "exclude_resources", "exclude_metadata", "operations"}
)
_EXCLUDABLE_METADATA = frozenset(
    {"description", "owner", "modalities", "semantic_profile", "source_version", "lineage", "provenance"}
)


class PolicyError(Exception):
    """The policy file is missing, unreadable or invalid. There is no fallback."""


@dataclass(frozen=True)
class DatasetRule:
    share: bool
    exclude_tables: frozenset[str] = frozenset()
    exclude_columns: Mapping[str, frozenset[str]] = field(default_factory=dict)
    exclude_resources: frozenset[str] = frozenset()
    exclude_metadata: frozenset[str] = frozenset()
    operations: frozenset[str] = frozenset()


@dataclass(frozen=True)
class StudyPolicy:
    datasets: Mapping[str, DatasetRule]


def _error(path: Path, message: str) -> PolicyError:
    return PolicyError(f"invalid study policy '{path}': {message}")


def _name_set(value, label: str, path: Path) -> frozenset[str]:
    if not isinstance(value, list) or not all(isinstance(x, str) and x for x in value):
        raise _error(path, f"{label} must be a list of non-empty strings")
    return frozenset(value)


def _parse_rule(value, label: str, path: Path) -> DatasetRule:
    if not isinstance(value, dict):
        raise _error(path, f"{label} must be an object")
    unknown = set(value) - _RULE_KEYS
    if unknown:
        raise _error(path, f"{label} has unsupported keys {sorted(unknown)}")
    share = value.get("share")
    if not isinstance(share, bool):
        raise _error(path, f"{label}.share is required and must be a boolean")
    columns = value.get("exclude_columns", {})
    if not isinstance(columns, dict):
        raise _error(path, f"{label}.exclude_columns must be an object")
    metadata = _name_set(value.get("exclude_metadata", []), f"{label}.exclude_metadata", path)
    if not metadata <= _EXCLUDABLE_METADATA:
        raise _error(path, f"{label}.exclude_metadata has unsupported names {sorted(metadata - _EXCLUDABLE_METADATA)}")
    return DatasetRule(
        share=share,
        exclude_tables=_name_set(value.get("exclude_tables", []), f"{label}.exclude_tables", path),
        exclude_columns={t: _name_set(c, f"{label}.exclude_columns.{t}", path) for t, c in columns.items()},
        exclude_resources=_name_set(value.get("exclude_resources", []), f"{label}.exclude_resources", path),
        exclude_metadata=metadata,
        operations=_name_set(value.get("operations", []), f"{label}.operations", path),
    )


def load_policy(path: Path) -> StudyPolicy:
    """Read and validate the policy file. Not cached: a replaced file is seen on the next call."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise PolicyError(f"cannot read study policy '{path}': {type(e).__name__}") from None
    if not isinstance(data, dict):
        raise _error(path, "must be an object")
    unknown = set(data) - _TOP_LEVEL_KEYS
    if unknown:
        raise _error(path, f"unsupported keys {sorted(unknown)}")
    version = data.get("format_version")
    if type(version) is not int or version != POLICY_FORMAT_VERSION:
        raise _error(path, f"format_version must be {POLICY_FORMAT_VERSION}")
    datasets = data.get("datasets", {})
    if not isinstance(datasets, dict) or not all(isinstance(k, str) and k for k in datasets):
        raise _error(path, "datasets must be an object keyed by dataset name")
    return StudyPolicy(datasets={name: _parse_rule(rule, f"datasets.{name}", path) for name, rule in datasets.items()})


def apply_policy(rule: DatasetRule, catalog: CatalogInfo, installed_ops: set[str]) -> CatalogInfo:
    """The shareable view of a catalog. Returns a new object; the caller validates it."""
    removed_tables = {t.name for t in catalog.tables if t.name in rule.exclude_tables}
    removed_columns = {
        t.name: {c.name for c in t.columns if c.name in rule.exclude_columns.get(t.name, ())}
        for t in catalog.tables
        if t.name not in removed_tables
    }

    tables = []
    for t in catalog.tables:
        if t.name in removed_tables:
            continue
        gone = removed_columns[t.name]
        relationships = tuple(
            r
            for r in t.relationships
            if not (set(r.columns) & gone)
            and r.target_table not in removed_tables
            and not (set(r.target_columns) & removed_columns.get(r.target_table, set()))
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
        resources=tuple(r for r in catalog.resources if r.type not in rule.exclude_resources),
        description=None if "description" in excluded else catalog.description,
        owner=None if "owner" in excluded else catalog.owner,
        modalities=() if "modalities" in excluded else catalog.modalities,
        semantic_profile=SemanticProfile() if "semantic_profile" in excluded else catalog.semantic_profile,
        source_version=None if "source_version" in excluded else catalog.source_version,
        lineage=() if "lineage" in excluded else catalog.lineage,
        provenance=(Provenance(source=catalog.provenance.source) if "provenance" in excluded else catalog.provenance),
        operations=tuple(sorted(set(catalog.operations) & rule.operations & set(installed_ops))),
    )
