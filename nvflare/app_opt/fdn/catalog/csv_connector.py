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

"""Infer a CSV schema locally from its header and a bounded sample; never publish rows."""

import csv
import math
import re
import time
from datetime import date
from itertools import islice

from nvflare.app_common.fdn.catalog.common.schema import CatalogInfo, ColumnInfo, Provenance, TableInfo
from nvflare.app_common.fdn.catalog.site.connector import CatalogConnector, ConnectorError

SAMPLE_ROWS = 100
MAX_SAMPLE_BYTES = 1024 * 1024
_NUMBER = re.compile(r"[+-]?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")


def _value_type(value):
    if value == "":
        return None
    if value.lower() in {"true", "false"}:
        return "bool"
    if _NUMBER.fullmatch(value):
        if "." not in value and "e" not in value.lower():
            # Keep oversized integers and identifier-like leading zeros as strings.
            if len(value.lstrip("+-")) <= 19 and -(2**63) <= int(value) < 2**63:
                return "int64"
            return "string"
        if math.isfinite(float(value)):
            return "float64"
    try:
        if date.fromisoformat(value).isoformat() == value:
            return "date"
    except ValueError:
        pass
    return "string"


class CSVConnector(CatalogConnector):
    """One UTF-8, comma-delimited file per dataset, with a mandatory unique header.

    Types describe at most 100 sampled rows, not a validated full-file schema.
    Empty cells are null; nullability remains unknown. No keys are inferred.
    """

    source = "csv"

    def read_catalog(self, dataset, limits, options=None):
        if options:
            raise ConnectorError("built-in csv reader accepts no options")
        path = self._datasets.get(dataset)
        if path is None or not path.is_file():
            raise ConnectorError("CSV dataset is unavailable")
        deadline = time.monotonic() + limits.max_duration_s
        calls, consumed = 0, 0

        def lines(stream):
            nonlocal calls, consumed
            while True:
                calls += 1
                if calls > limits.max_calls or time.monotonic() >= deadline:
                    raise ConnectorError("CSV sample read budget exceeded", budget=True)
                line = stream.readline(MAX_SAMPLE_BYTES - consumed + 1)
                consumed += len(line)
                if consumed > MAX_SAMPLE_BYTES:
                    raise ConnectorError("CSV sample byte budget exceeded", budget=True)
                if not line:
                    return
                yield line.decode("utf-8-sig" if calls == 1 else "utf-8")

        try:
            with path.open("rb") as stream:
                reader = csv.reader(lines(stream), strict=True)
                header = next(reader, None)
                if not header or any(not name.strip() for name in header) or len(set(header)) != len(header):
                    raise ConnectorError("CSV requires nonempty, unique column headers")
                if 2 + len(header) > limits.max_records:
                    raise ConnectorError("CSV schema record budget exceeded", budget=True)
                types = [set() for _ in header]
                for row in islice(reader, SAMPLE_ROWS):
                    if len(row) != len(header):
                        raise ConnectorError("CSV row does not match header width")
                    for observed, value in zip(types, row):
                        kind = _value_type(value)
                        if kind is not None:
                            observed.add(kind)
        except (OSError, UnicodeError, csv.Error):
            raise ConnectorError("CSV dataset is unreadable") from None
        if time.monotonic() >= deadline:
            raise ConnectorError("CSV sample read budget exceeded", budget=True)
        columns = []
        for name, observed in zip(header, types):
            if not observed:
                kind = "unknown"
            elif len(observed) == 1:
                kind = next(iter(observed))
            elif observed <= {"int64", "float64"}:
                kind = "float64"
            else:
                kind = "string"
            columns.append(ColumnInfo(name, kind))
        return CatalogInfo(
            dataset=dataset,
            display_name=dataset,
            provenance=Provenance(self.source),
            tables=(TableInfo(path.stem, tuple(columns)),),
            modalities=("tabular",),
        )
