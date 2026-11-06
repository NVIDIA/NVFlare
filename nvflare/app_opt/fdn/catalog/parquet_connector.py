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

"""Reads a dataset's catalog from Parquet footers, plus an optional site-authored catalog.json sidecar."""

import time
from pathlib import Path

from nvflare.app_common.fdn.catalog.common.schema import PROFILE_KEYS, CatalogInfo, ReadLimits, ValidationError
from nvflare.app_common.fdn.catalog.site.connector import CatalogConnector, ConnectorError
from nvflare.app_opt.fdn.catalog.declarations import SIDECAR_NAME, read_declarations


class ParquetConnector(CatalogConnector):
    """One table per Parquet file, or per top-level directory of nested parts. Read only footers, never rows."""

    source = "parquet"

    def read_catalog(self, dataset: str, limits: ReadLimits, options: dict | None = None) -> CatalogInfo:
        if options:
            raise ConnectorError("built-in parquet reader accepts no options")
        import pyarrow.parquet as pq

        root = self._datasets.get(dataset)
        if root is None or not root.exists():
            raise ConnectorError("dataset mount is missing")
        deadline = time.monotonic() + limits.max_duration_s
        calls = 0

        def read_schema(path: Path):
            nonlocal calls
            calls += 1
            if calls > limits.max_calls or time.monotonic() >= deadline:
                raise ConnectorError("read budget exceeded", budget=True)
            try:
                return pq.read_schema(path)
            except Exception:
                raise ConnectorError("parquet metadata is unreadable") from None

        tables = []
        for name, parts in self._table_files(root, deadline, limits.max_calls):
            schema = read_schema(parts[0])
            for part in parts[1:]:
                if read_schema(part) != schema:
                    raise ConnectorError("part files have inconsistent schemas")
            tables.append(
                {
                    "name": name,
                    "columns": [{"name": f.name, "type": str(f.type), "nullable": f.nullable} for f in schema],
                    "primary_key": [],
                    "relationships": [],
                }
            )
            if len(tables) + sum(len(t["columns"]) for t in tables) > limits.max_records:
                raise ConnectorError("record budget exceeded", budget=True)

        sidecar = self._read_sidecar(root)
        extra = sidecar.get("tables", {})
        if not isinstance(extra, dict):
            raise ConnectorError("sidecar tables must be an object")
        for table in tables:
            declared = extra.get(table["name"], {})
            if not isinstance(declared, dict) or set(declared) - {"primary_key", "relationships"}:
                raise ConnectorError("sidecar table entry is invalid")
            table["primary_key"] = declared.get("primary_key", [])
            table["relationships"] = declared.get("relationships", [])

        provenance = sidecar.get("provenance", {})
        if not isinstance(provenance, dict) or set(provenance) - {"declaration_ref", "validation_ref"}:
            raise ConnectorError("sidecar provenance may only carry references")
        profile = sidecar.get("semantic_profile", {})
        if not isinstance(profile, dict) or set(profile) - set(PROFILE_KEYS):
            raise ConnectorError("sidecar semantic_profile is invalid")

        wire = {
            "dataset": dataset,
            "metadata": {
                "display_name": sidecar.get("display_name", dataset),
                "description": sidecar.get("description"),
                "owner": sidecar.get("owner"),
                "tables": tables,
                "resources": [],
                "semantic_profile": {k: profile.get(k) for k in PROFILE_KEYS},
                "modalities": sidecar.get("modalities", []),
                "operations": sidecar.get("operations", []),
                "source_version": sidecar.get("source_version"),
                "lineage": sidecar.get("lineage", []),
                "provenance": {
                    "source": self.source,
                    "declaration_ref": provenance.get("declaration_ref"),
                    "validation_ref": provenance.get("validation_ref"),
                },
            },
            "observed_at": sidecar.get("observed_at"),
        }
        try:
            catalog = CatalogInfo.from_wire(wire)
        except ValidationError:
            raise ConnectorError("sidecar metadata is invalid") from None
        if catalog.record_count() > limits.max_records:
            raise ConnectorError("record budget exceeded", budget=True)
        return catalog

    @staticmethod
    def _table_files(root: Path, deadline: float, max_calls: int) -> list[tuple[str, list[Path]]]:
        """(table name, part files) in name order. The mount's listing order is not stable, so sort. The walk of a
        partitioned table stops at the read budget: each part costs a read, and a deep tree can take long to list."""
        if root.is_file():
            # Container mount aliases need not retain the source filename's extension.
            return [(root.stem, [root])]
        tables = []
        for child in sorted(root.iterdir()):
            if child.is_file() and child.suffix == ".parquet":
                tables.append((child.stem, [child]))
            elif child.is_dir():
                parts = []
                for part in child.rglob("*.parquet"):
                    if len(parts) >= max_calls or time.monotonic() >= deadline:
                        raise ConnectorError("read budget exceeded", budget=True)
                    if part.is_file():
                        parts.append(part)
                parts.sort()
                if parts:
                    tables.append((child.name, parts))
        return tables

    @staticmethod
    def _read_sidecar(root: Path) -> dict:
        return read_declarations(root / SIDECAR_NAME) if root.is_dir() else {}
