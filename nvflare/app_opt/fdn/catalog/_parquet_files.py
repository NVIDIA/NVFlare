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

"""Group local Parquet files into named tables for a FLARE catalog request.

A dataset URI can point to a single file or to a directory containing several tables.
For example, given a directory with ``person.parquet``, ``visits/2024/part.parquet``
and ``visits/2025/part.parquet``, FLARE reports two tables: ``person`` and ``visits``.
Both files under ``visits/`` are parts of that table.

The site reader calls ``table_files(local_path(dataset), budget.deadline, budget.request)``.
For each returned table, it checks that the parts have matching footer schemas and
passes one representative file to the upstream Parquet importer.
"""

import time
from pathlib import Path

from nvflare.app_common.fdn.catalog.site.source_error import CatalogSourceError


def table_files(root: Path, deadline: float, spend) -> list[tuple[str, list[Path]]]:
    """Return sorted (table name, sorted part paths) pairs for the configured root.

    A single file takes its table name from its stem. Under a directory root, each
    top-level Parquet file or directory containing Parquet parts names one table.
    During traversal, check the absolute monotonic ``deadline`` and call ``spend()``
    for each discovered part. Ambiguous table names raise ``CatalogSourceError``.
    """

    def check():
        if time.monotonic() >= deadline:
            raise CatalogSourceError("read budget exceeded", budget=True)

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
            raise CatalogSourceError("dataset directory is unreadable") from None
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
        raise CatalogSourceError("a file and a directory in the dataset map to the same table name")
    return sorted(tables)
