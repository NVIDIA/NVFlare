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

"""Catalog metadata model.

Typed, frozen dataclasses for code; plain dicts on the wire. This module has no FLARE or I/O imports.
Error messages carry field paths only, never values from the source.
"""

from dataclasses import dataclass, field

MAX_TEXT_LEN = 256
MAX_DESCRIPTION_LEN = 4096
RELATION_DERIVED_FROM = "derived_from"


class ValidationError(ValueError):
    """A catalog violates the metadata schema. The message names fields, never source values."""


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    type: str
    nullable: bool | None = None


@dataclass(frozen=True)
class Relationship:
    columns: tuple[str, ...]
    target_table: str
    target_columns: tuple[str, ...]


@dataclass(frozen=True)
class TableInfo:
    name: str
    columns: tuple[ColumnInfo, ...]
    primary_key: tuple[str, ...] = ()
    relationships: tuple[Relationship, ...] = ()


@dataclass(frozen=True)
class Resource:
    type: str
    supported_search: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticProfile:
    standard: str | None = None
    version: str | None = None
    vocabulary_version: str | None = None
    mapping_version: str | None = None
    declaration_ref: str | None = None
    validation_ref: str | None = None


@dataclass(frozen=True)
class Lineage:
    dataset: str
    source_version: str | None
    relation: str = RELATION_DERIVED_FROM


@dataclass(frozen=True)
class Provenance:
    source: str
    declaration_ref: str | None = None
    validation_ref: str | None = None


@dataclass(frozen=True)
class ReadLimits:
    max_duration_s: float
    max_calls: int
    max_records: int


DEFAULT_LIMITS = {"max_duration_s": 60, "max_calls": 1000, "max_records": 10000}


@dataclass(frozen=True)
class CatalogInfo:
    """The catalog of one Study dataset."""

    dataset: str
    display_name: str
    provenance: Provenance
    tables: tuple[TableInfo, ...] = ()
    resources: tuple[Resource, ...] = ()
    semantic_profile: SemanticProfile = field(default_factory=SemanticProfile)
    modalities: tuple[str, ...] = ()
    operations: tuple[str, ...] = ()
    description: str | None = None
    owner: str | None = None
    source_version: str | None = None
    lineage: tuple[Lineage, ...] = ()
    observed_at: str | None = None

    def metadata_wire(self) -> dict:
        return {
            "display_name": self.display_name,
            "description": self.description,
            "owner": self.owner,
            "tables": [_table_wire(t) for t in self.tables],
            "resources": [{"type": r.type, "supported_search": list(r.supported_search)} for r in self.resources],
            "semantic_profile": _profile_wire(self.semantic_profile),
            "modalities": list(self.modalities),
            "operations": list(self.operations),
            "source_version": self.source_version,
            "lineage": [
                {"dataset": x.dataset, "source_version": x.source_version, "relation": x.relation} for x in self.lineage
            ],
            "provenance": {
                "source": self.provenance.source,
                "declaration_ref": self.provenance.declaration_ref,
                "validation_ref": self.provenance.validation_ref,
            },
        }

    def to_wire(self) -> dict:
        return {"dataset": self.dataset, "metadata": self.metadata_wire(), "observed_at": self.observed_at}

    @classmethod
    def from_wire(cls, d: dict) -> "CatalogInfo":
        """Build from the wire dict. Rejects unknown or missing keys and wrong types."""
        _check_keys(d, "catalog", _CATALOG_KEYS)
        md = d["metadata"]
        _check_keys(md, "metadata", _METADATA_KEYS)
        profile = md["semantic_profile"]
        _check_keys(profile, "metadata.semantic_profile", PROFILE_KEYS)
        prov = md["provenance"]
        _check_keys(prov, "metadata.provenance", _PROVENANCE_KEYS)
        return cls(
            dataset=_str(d["dataset"], "dataset"),
            display_name=_str(md["display_name"], "metadata.display_name"),
            provenance=Provenance(
                source=_str(prov["source"], "metadata.provenance.source"),
                declaration_ref=_opt_str(prov["declaration_ref"], "metadata.provenance.declaration_ref"),
                validation_ref=_opt_str(prov["validation_ref"], "metadata.provenance.validation_ref"),
            ),
            tables=tuple(
                _parse_table(t, f"metadata.tables[{i}]") for i, t in enumerate(_list(md["tables"], "metadata.tables"))
            ),
            resources=tuple(
                _parse_resource(r, f"metadata.resources[{i}]")
                for i, r in enumerate(_list(md["resources"], "metadata.resources"))
            ),
            semantic_profile=SemanticProfile(
                **{k: _opt_str(profile[k], f"metadata.semantic_profile.{k}") for k in PROFILE_KEYS}
            ),
            modalities=_str_tuple(md["modalities"], "metadata.modalities"),
            operations=_str_tuple(md["operations"], "metadata.operations"),
            description=_opt_str(md["description"], "metadata.description"),
            owner=_opt_str(md["owner"], "metadata.owner"),
            source_version=_opt_str(md["source_version"], "metadata.source_version"),
            lineage=tuple(
                _parse_lineage(x, f"metadata.lineage[{i}]")
                for i, x in enumerate(_list(md["lineage"], "metadata.lineage"))
            ),
            observed_at=_opt_str(d["observed_at"], "observed_at"),
        )

    def record_count(self) -> int:
        """Dataset, table, column, relationship, resource and lineage objects."""
        return (
            1
            + len(self.tables)
            + sum(len(t.columns) + len(t.relationships) for t in self.tables)
            + len(self.resources)
            + len(self.lineage)
        )


_CATALOG_KEYS = frozenset({"dataset", "metadata", "observed_at"})
_METADATA_KEYS = frozenset(
    {
        "display_name",
        "description",
        "owner",
        "tables",
        "resources",
        "semantic_profile",
        "modalities",
        "operations",
        "source_version",
        "lineage",
        "provenance",
    }
)
PROFILE_KEYS = ("standard", "version", "vocabulary_version", "mapping_version", "declaration_ref", "validation_ref")
_PROVENANCE_KEYS = frozenset({"source", "declaration_ref", "validation_ref"})
_TABLE_KEYS = frozenset({"name", "columns", "primary_key", "relationships"})
_COLUMN_KEYS = frozenset({"name", "type", "nullable"})
_RELATIONSHIP_KEYS = frozenset({"columns", "target_table", "target_columns"})
_RESOURCE_KEYS = frozenset({"type", "supported_search"})
_LINEAGE_KEYS = frozenset({"dataset", "source_version", "relation"})


def _table_wire(t: TableInfo) -> dict:
    return {
        "name": t.name,
        "columns": [{"name": c.name, "type": c.type, "nullable": c.nullable} for c in t.columns],
        "primary_key": list(t.primary_key),
        "relationships": [
            {"columns": list(r.columns), "target_table": r.target_table, "target_columns": list(r.target_columns)}
            for r in t.relationships
        ],
    }


def _profile_wire(p: SemanticProfile) -> dict:
    return {k: getattr(p, k) for k in PROFILE_KEYS}


def _check_keys(value, path: str, keys) -> None:
    if not isinstance(value, dict):
        raise ValidationError(f"{path} must be an object")
    if set(value) - set(keys):
        raise ValidationError(f"{path} has unsupported fields")
    missing = sorted(set(keys) - set(value))
    if missing:
        raise ValidationError(f"{path} is missing {missing}")


def _str(value, path: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{path} must be a string")
    return value


def _opt_str(value, path: str) -> str | None:
    return None if value is None else _str(value, path)


def _list(value, path: str) -> list:
    if not isinstance(value, list):
        raise ValidationError(f"{path} must be a list")
    return value


def _str_tuple(value, path: str) -> tuple[str, ...]:
    return tuple(_str(x, f"{path}[{i}]") for i, x in enumerate(_list(value, path)))


def _parse_table(t, path: str) -> TableInfo:
    _check_keys(t, path, _TABLE_KEYS)
    columns = []
    for i, c in enumerate(_list(t["columns"], f"{path}.columns")):
        cpath = f"{path}.columns[{i}]"
        _check_keys(c, cpath, _COLUMN_KEYS)
        nullable = c["nullable"]
        if nullable is not None and not isinstance(nullable, bool):
            raise ValidationError(f"{cpath}.nullable must be a boolean or null")
        columns.append(
            ColumnInfo(name=_str(c["name"], f"{cpath}.name"), type=_str(c["type"], f"{cpath}.type"), nullable=nullable)
        )
    relationships = []
    for i, r in enumerate(_list(t["relationships"], f"{path}.relationships")):
        rpath = f"{path}.relationships[{i}]"
        _check_keys(r, rpath, _RELATIONSHIP_KEYS)
        relationships.append(
            Relationship(
                columns=_str_tuple(r["columns"], f"{rpath}.columns"),
                target_table=_str(r["target_table"], f"{rpath}.target_table"),
                target_columns=_str_tuple(r["target_columns"], f"{rpath}.target_columns"),
            )
        )
    return TableInfo(
        name=_str(t["name"], f"{path}.name"),
        columns=tuple(columns),
        primary_key=_str_tuple(t["primary_key"], f"{path}.primary_key"),
        relationships=tuple(relationships),
    )


def _parse_resource(r, path: str) -> Resource:
    _check_keys(r, path, _RESOURCE_KEYS)
    return Resource(
        type=_str(r["type"], f"{path}.type"),
        supported_search=_str_tuple(r["supported_search"], f"{path}.supported_search"),
    )


def _parse_lineage(x, path: str) -> Lineage:
    _check_keys(x, path, _LINEAGE_KEYS)
    return Lineage(
        dataset=_str(x["dataset"], f"{path}.dataset"),
        source_version=_opt_str(x["source_version"], f"{path}.source_version"),
        relation=_str(x["relation"], f"{path}.relation"),
    )


def _text(
    value: str | None, path: str, *, required: bool = True, max_len: int = MAX_TEXT_LEN, allow_empty: bool = False
) -> None:
    if value is None:
        if required:
            raise ValidationError(f"{path} is required")
        return
    if not value and not allow_empty:
        raise ValidationError(f"{path} must not be empty")
    if len(value) > max_len:
        raise ValidationError(f"{path} exceeds {max_len} characters")


def _unique(values, path: str) -> None:
    if len(set(values)) != len(values):
        raise ValidationError(f"{path} has duplicate entries")


def validate(catalog: CatalogInfo, limits: ReadLimits) -> None:
    """Enforce metadata format bounds and references. Raises ValidationError."""
    if catalog.record_count() > limits.max_records:
        raise ValidationError("catalog exceeds the record budget")
    _text(catalog.dataset, "dataset")
    _text(catalog.display_name, "display_name")
    _text(catalog.description, "description", required=False, max_len=MAX_DESCRIPTION_LEN, allow_empty=True)
    _text(catalog.owner, "owner", required=False)

    _unique([t.name for t in catalog.tables], "tables")
    tables = {t.name: {c.name for c in t.columns} for t in catalog.tables}
    for i, t in enumerate(catalog.tables):
        path = f"tables[{i}]"
        _text(t.name, f"{path}.name")
        _unique([c.name for c in t.columns], f"{path}.columns")
        for j, c in enumerate(t.columns):
            _text(c.name, f"{path}.columns[{j}].name")
            _text(c.type, f"{path}.columns[{j}].type")
        _unique(t.primary_key, f"{path}.primary_key")
        if not set(t.primary_key) <= tables[t.name]:
            raise ValidationError(f"{path}.primary_key references an unknown column")
        for j, r in enumerate(t.relationships):
            rpath = f"{path}.relationships[{j}]"
            if not r.columns or len(r.columns) != len(r.target_columns):
                raise ValidationError(f"{rpath} column lists must be nonempty and of equal length")
            if not set(r.columns) <= tables[t.name]:
                raise ValidationError(f"{rpath} references an unknown column")
            if r.target_table not in tables or not set(r.target_columns) <= tables[r.target_table]:
                raise ValidationError(f"{rpath} references an unknown target")

    _unique([r.type for r in catalog.resources], "resources")
    for i, r in enumerate(catalog.resources):
        _text(r.type, f"resources[{i}].type")
        _unique(r.supported_search, f"resources[{i}].supported_search")
        for s in r.supported_search:
            _text(s, f"resources[{i}].supported_search")

    for name, values in (("modalities", catalog.modalities), ("operations", catalog.operations)):
        _unique(values, name)
        for v in values:
            _text(v, name)

    profile = catalog.semantic_profile
    for k in PROFILE_KEYS:
        _text(getattr(profile, k), f"semantic_profile.{k}", required=False)
    _text(catalog.source_version, "source_version", required=False)

    _unique([x.dataset for x in catalog.lineage], "lineage")
    for i, x in enumerate(catalog.lineage):
        _text(x.dataset, f"lineage[{i}].dataset")
        _text(x.source_version, f"lineage[{i}].source_version", required=False)
        if x.relation != RELATION_DERIVED_FROM:
            raise ValidationError(f"lineage[{i}].relation is not supported")

    _text(catalog.provenance.source, "provenance.source")
    _text(catalog.provenance.declaration_ref, "provenance.declaration_ref", required=False)
    _text(catalog.provenance.validation_ref, "provenance.validation_ref", required=False)
