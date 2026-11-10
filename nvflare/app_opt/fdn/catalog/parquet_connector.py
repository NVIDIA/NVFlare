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

"""Reads a dataset's catalog from Parquet footers.

The dataset's ``uri`` is a ``file://`` URI of a Parquet file or directory; the connector takes no options.
"""

import time
from pathlib import Path

from nvflare.apis.dataset_config_spec import DatasetConfig
from nvflare.app_common.fdn.catalog.common.schema import CatalogInfo, ReadLimits
from nvflare.app_common.fdn.catalog.site.connector import CatalogConnector, ConnectorError
from nvflare.app_opt.fdn.catalog._metadata import build_catalog, local_path, selection


class ParquetConnector(CatalogConnector):
    """One table per Parquet file, or per top-level directory of nested parts. Read only footers, never rows."""

    source = "parquet"

    def read_catalog(self, dataset: DatasetConfig, secret, limits: ReadLimits) -> CatalogInfo:
        selection(dataset.options, ())
        import pyarrow.parquet as pq

        root = local_path(dataset)
        if not root.exists():
            raise ConnectorError("dataset files are missing")
        deadline = time.monotonic() + limits.max_duration_s
        calls = 0

        def spend():
            """One call of the budget, shared by listing a part file and reading a footer."""
            nonlocal calls
            calls += 1
            if calls > limits.max_calls or time.monotonic() >= deadline:
                raise ConnectorError("read budget exceeded", budget=True)

        def read_schema(path: Path):
            spend()
            try:
                return pq.read_schema(path)
            except Exception:
                raise ConnectorError("parquet metadata is unreadable") from None

        tables = []
        for name, parts in self._table_files(root, deadline, spend):
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

        catalog = build_catalog(dataset.name, self.source, tables, limits)
        if time.monotonic() >= deadline:
            raise ConnectorError("read budget exceeded", budget=True)
        return catalog

    @staticmethod
    def _table_files(root: Path, deadline: float, spend) -> list[tuple[str, list[Path]]]:
        """(table name, part files) in name order. The mount's listing order is not stable, so sort. Each part file
        listed costs one call of the read budget (``spend``), which its footer read then costs again, so the walk of
        a partitioned table stops at the budget and a deep tree cannot take long to list."""

        def check():
            if time.monotonic() >= deadline:
                raise ConnectorError("read budget exceeded", budget=True)

        def children(directory):
            check()
            try:
                for path in directory.iterdir():
                    check()
                    # Hidden and underscore-prefixed entries are not data: Delta Lake's _delta_log, Spark's
                    # _temporary and _SUCCESS, Hive's .hive-staging, and editor or OS files.
                    if path.name.startswith((".", "_")):
                        continue
                    yield path
            except OSError:
                # unreadable, or removed during the walk; never the raw error, which names a local path
                raise ConnectorError("dataset directory is unreadable") from None
            check()

        check()
        if root.is_file():
            return [(root.stem, [root])]
        tables = []

        def add(parts, path):
            spend()
            parts.append(path)

        for child in children(root):
            if child.is_file() and child.suffix == ".parquet":
                parts = []
                add(parts, child)
                tables.append((child.stem, parts))
            elif child.is_dir() and not child.is_symlink():
                parts, pending = [], [child]
                while pending:
                    for part in children(pending.pop()):
                        if part.is_dir() and not part.is_symlink():
                            pending.append(part)
                        elif part.suffix == ".parquet" and part.is_file():
                            add(parts, part)
                parts.sort()
                if parts:
                    tables.append((child.name, parts))
        check()
        names = [name for name, _ in tables]
        if len(set(names)) != len(names):
            # e.g. a.parquet beside a partitioned a/; publishing either would hide the other
            raise ConnectorError("a file and a directory in the dataset map to the same table name")
        return sorted(tables)
