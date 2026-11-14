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

"""Offline catalog views, following the federated statistics visualization pattern.

Pass the ``catalog_results.json`` of a catalog read, as a file path or the loaded dict, for example the result of
``get_catalog(...)``. No server connection is opened here. ``get_*`` methods return DataFrames; ``show_*`` methods
return pandas Stylers that Jupyter renders with a title and explanation. pandas and Jinja2 are visualization
dependencies only.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from html import escape
from pathlib import Path

import pandas as pd

from nvflare.app_common.fdn.catalog.common.payload import PayloadError, check_catalog_results
from nvflare.app_common.fdn.catalog.common.schema import CatalogInfo, ValidationError

PUBLISHED, UNAVAILABLE, SKIPPED = "published", "unavailable", "skipped"


@dataclass(frozen=True)
class _Entry:
    site: str
    dataset: str
    status: str  # published, unavailable or skipped
    catalog: CatalogInfo | None = None


def _load(data) -> Mapping:
    if isinstance(data, (str, Path)):
        with Path(data).open() as stream:
            data = json.load(stream)
    if not isinstance(data, Mapping):
        raise ValueError("Expected a catalog_results.json file path or its loaded dict.")
    try:
        results = check_catalog_results(data)
    except PayloadError as error:
        raise ValueError(f"Not a catalog read result: {error}") from error
    if not isinstance(results.get("study"), str) or not results["study"]:
        raise ValueError("Not a catalog read result: it names no Study")
    return results


def _read_entries(results) -> list[_Entry]:
    entries = []
    for item in results["datasets"]:
        if not isinstance(item, Mapping) or not isinstance(item.get("site"), str) or not item["site"]:
            raise ValueError("Each published dataset needs a nonempty site.")
        try:
            catalog = CatalogInfo.from_wire({k: v for k, v in item.items() if k != "site"})
        except ValidationError as error:
            raise ValueError(f"Invalid catalog metadata for {item['site']}/{item.get('dataset')}: {error}") from error
        entries.append(_Entry(item["site"], catalog.dataset, PUBLISHED, catalog))
    for site, info in results["sites"].items():
        if not isinstance(info, Mapping):
            raise ValueError(f"Invalid status for site {site}.")
        for status, key in ((UNAVAILABLE, "unavailable"), (SKIPPED, "skipped")):
            entries.extend(_Entry(site, name, status) for name in info.get(key) or [])
    return entries


class CatalogVisualization:
    """Reusable views over one catalog read's results.

    Sources and type names are shown as published; no type equivalence or scientific compatibility is inferred.
    Datasets a site reported as unavailable, or skipped for its read budget, appear by name without metadata.
    """

    def __init__(self, data):
        self._results = _load(data)
        entries = _read_entries(self._results)
        keys = [(entry.site, entry.dataset) for entry in entries]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate site/dataset entries in the catalog results.")
        self._entries = sorted(entries, key=lambda entry: (entry.site, entry.dataset))

    @property
    def study(self) -> str:
        """The Study this catalog read covered."""
        return self._results["study"]

    def _titled(self, frame, title, description):
        """A styled view whose title names the Study, so every rendered table says which Study it describes."""
        return self._style(frame, f"{title} · Study {self.study}", description)

    def get_sites(self):
        """One row per site the read targeted: its status, reason code and dataset counts."""
        published = {}
        for entry in self._entries:
            if entry.status == PUBLISHED:
                published[entry.site] = published.get(entry.site, 0) + 1
        return pd.DataFrame(
            [
                {
                    "site": site,
                    "status": info.get("status"),
                    "reason_code": info.get("reason_code"),
                    "published": published.get(site, 0),
                    "unavailable": len(info.get("unavailable") or []),
                    "skipped": len(info.get("skipped") or []),
                }
                for site, info in sorted(self._results["sites"].items())
            ],
            columns=["site", "status", "reason_code", "published", "unavailable", "skipped"],
        )

    def show_sites(self):
        """Display each site's status in this read."""
        return self._titled(
            self.get_sites(),
            "Site status",
            "OK: the site completed its read. FAILED: its configuration could not be used. "
            "NO_RESULT: no result arrived, for example after a timeout or disconnect.",
        )

    def _select(self, site=None, dataset=None):
        return [
            entry
            for entry in self._entries
            if (site is None or entry.site == site) and (dataset is None or entry.dataset == dataset)
        ]

    @staticmethod
    def _style(frame, title=None, description=None):
        # Catalog text is site supplied: escape both values and labels in notebook HTML.
        styled = (
            frame.style.format(escape="html", na_rep="not published", precision=0)
            .format_index(escape="html", axis=0)
            .format_index(escape="html", axis=1)
        )
        if title:
            # Captions travel with the rendered table, including under %run, which
            # omits the notebook's Markdown cells. Table names are site supplied.
            styled.set_caption(f"<strong>{escape(title)}</strong><br>{escape(description or '')}")
            styled.set_table_styles(
                [
                    {
                        "selector": "caption",
                        "props": [
                            ("caption-side", "top"),
                            ("text-align", "left"),
                            ("font-size", "1rem"),
                            ("line-height", "1.5"),
                            ("padding", "1rem 0 0.5rem"),
                        ],
                    }
                ],
                overwrite=False,
            )
        return styled

    def get_datasets(self, site=None, dataset=None):
        """One row per site/dataset: its source, status and schema size."""
        rows = []
        for entry in self._select(site, dataset):
            catalog = entry.catalog
            rows.append(
                {
                    "site": entry.site,
                    "dataset": entry.dataset,
                    "source": catalog.provenance.source if catalog else None,
                    "status": entry.status,
                    "tables": len(catalog.tables) if catalog else None,
                    "columns": sum(len(table.columns) for table in catalog.tables) if catalog else None,
                    "observed_at": catalog.observed_at if catalog else None,
                }
            )
        return pd.DataFrame(rows, columns=["site", "dataset", "source", "status", "tables", "columns", "observed_at"])

    def show_datasets(self, site=None, dataset=None):
        """Display the site/dataset overview."""
        return self._titled(
            self.get_datasets(site, dataset),
            "Site datasets and data sources",
            "One row per site and dataset, with its source, status and published table/column counts.",
        ).format(escape="html", na_rep="unknown", subset=["observed_at"])

    def get_tables(self, site=None, dataset=None):
        """Published tables, column counts and declared primary keys."""
        return pd.DataFrame(
            [
                {
                    "site": entry.site,
                    "dataset": entry.dataset,
                    "table": table.name,
                    "columns": len(table.columns),
                    "primary_key": ", ".join(table.primary_key),
                    "relationships": len(table.relationships),
                }
                for entry in self._select(site, dataset)
                if entry.catalog is not None
                for table in entry.catalog.tables
            ],
            columns=["site", "dataset", "table", "columns", "primary_key", "relationships"],
        )

    def show_tables(self, site=None, dataset=None):
        return self._titled(
            self.get_tables(site, dataset),
            "Published tables",
            "Tables, column counts, primary keys and relationship counts for each site and dataset.",
        )

    def get_schema(self, table, site=None, dataset=None):
        """One row per column, preserving each source's column order and type names."""
        rows = []
        for entry in self._select(site, dataset):
            if entry.catalog is None:
                continue
            for info in entry.catalog.tables:
                if info.name == table:
                    for position, column in enumerate(info.columns, start=1):
                        rows.append(
                            {
                                "site": entry.site,
                                "dataset": entry.dataset,
                                "position": position,
                                "column": column.name,
                                "type": column.type,
                                "nullable": column.nullable,
                                "primary_key": column.name in info.primary_key,
                            }
                        )
        return pd.DataFrame(rows, columns=["site", "dataset", "position", "column", "type", "nullable", "primary_key"])

    def show_schema(self, table, site=None, dataset=None):
        return self._titled(
            self.get_schema(table, site, dataset),
            f"Table schema: {table}",
            "Columns in source order, with reported types, nullability and primary-key membership.",
        ).format(escape="html", na_rep="unknown", subset=["nullable"])

    def _comparison(self, values, index, site, dataset):
        entries = self._select(site, dataset)
        columns = pd.MultiIndex.from_tuples([(e.site, e.dataset) for e in entries], names=["site", "dataset"])
        frame = pd.DataFrame(index=pd.Index(index, name="table"), columns=columns)
        for entry in entries:
            key = (entry.site, entry.dataset)
            for name, value in values.get(key, {}).items():
                frame.at[name, key] = value
        frame.attrs["unavailable"] = [(e.site, e.dataset) for e in entries if e.catalog is None]
        return frame

    def get_table_comparison(self, site=None, dataset=None):
        """Compare published table presence and column counts across site/dataset pairs."""
        values = {
            (entry.site, entry.dataset): {table.name: len(table.columns) for table in entry.catalog.tables}
            for entry in self._select(site, dataset)
            if entry.catalog is not None
        }
        tables = sorted({table for counts in values.values() for table in counts})
        return self._comparison(values, tables, site, dataset)

    def get_column_comparison(self, table, site=None, dataset=None, field="type"):
        """Compare type, nullable or primary_key, including across differently named datasets."""
        if field not in ("type", "nullable", "primary_key"):
            raise ValueError("field must be 'type', 'nullable', or 'primary_key'")
        schema = self.get_schema(table, site, dataset)
        values = {}
        for row in schema.to_dict("records"):
            value = row[field]
            if field in ("nullable", "primary_key"):
                value = "unknown" if value is None else ("yes" if value else "no")
            values.setdefault((row["site"], row["dataset"]), {})[row["column"]] = value
        # Preserve first observed order, appending columns found only at later sources.
        columns = list(dict.fromkeys(schema["column"]))
        frame = self._comparison(values, columns, site, dataset)
        frame.index.name = "column"
        return frame

    @staticmethod
    def _comparison_styles(frame, differences):
        styles = pd.DataFrame("", index=frame.index, columns=frame.columns)
        styles[frame.isna()] = "background-color: #fce4e4"
        if differences:
            for index, row in frame.iterrows():
                if row.dropna().nunique() > 1:
                    styles.loc[index, row.notna()] = "background-color: #fff0c2"
        for column in frame.attrs.get("unavailable", []):
            styles[column] = "background-color: #e5e7eb"
        return styles

    def _show_comparison(self, frame, title, description, differences=False):
        styles = self._comparison_styles(frame, differences)
        styled = self._titled(frame, title, description).apply(lambda _: styles, axis=None)
        for column in frame.attrs.get("unavailable", []):
            styled = styled.format(escape="html", na_rep="unavailable", subset=[column])
        return styled

    def show_table_comparison(self, site=None, dataset=None):
        return self._show_comparison(
            self.get_table_comparison(site, dataset),
            "Table comparison across sites and datasets",
            "Numbers are published column counts. Red: table not published; gray: dataset not read (unavailable or skipped). "
            "Site policy may omit tables.",
        )

    def show_column_comparison(self, table, site=None, dataset=None, field="type"):
        frame = self.get_column_comparison(table, site, dataset, field)
        title, description = {
            "type": ("Column type comparison", "Types are shown as reported by each source."),
            "nullable": (
                "Column nullability comparison",
                "Yes means nullable; no means not nullable; unknown means nullability was not reported.",
            ),
            "primary_key": ("Primary key comparison", "Yes marks a primary-key column; no marks a non-key column."),
        }[field]
        return self._show_comparison(
            frame,
            f"{title}: {table}",
            description
            + " Yellow: different reported values; red: column not published; gray: dataset not read (unavailable or skipped).",
            differences=True,
        )

    def get_relationships(self, site=None, dataset=None):
        return pd.DataFrame(
            [
                {
                    "site": entry.site,
                    "dataset": entry.dataset,
                    "table": table.name,
                    "columns": ", ".join(relation.columns),
                    "references": f"{relation.target_table}({', '.join(relation.target_columns)})",
                }
                for entry in self._select(site, dataset)
                if entry.catalog is not None
                for table in entry.catalog.tables
                for relation in table.relationships
            ],
            columns=["site", "dataset", "table", "columns", "references"],
        )

    def show_relationships(self, site=None, dataset=None):
        return self._titled(
            self.get_relationships(site, dataset),
            "Declared table relationships",
            "Foreign-key columns and the tables/columns they reference, as published by each site.",
        )
