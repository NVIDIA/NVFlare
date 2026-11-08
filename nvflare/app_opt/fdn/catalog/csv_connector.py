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

"""Discover CSV schemas with PyArrow; publish metadata, never source rows."""

import time

from nvflare.app_common.fdn.catalog.common.schema import CatalogInfo, ColumnInfo, Provenance, TableInfo
from nvflare.app_common.fdn.catalog.site.connector import CatalogConnector, ConnectorError

MAX_SAMPLE_BYTES = 1024 * 1024
INFERENCE_BLOCK_BYTES = 64 * 1024


class CSVConnector(CatalogConnector):
    """One UTF-8 CSV file per dataset, with a mandatory unique, nonempty header.

    Arrow infers types from its initial 64 KiB block, with at most 1 MiB of
    local input for record boundaries/read-ahead. These are sampled types,
    not a full-file validation. Nullability stays unknown; no keys are inferred.
    """

    source = "csv"

    def read_catalog(self, dataset, limits, options=None):
        if options:
            raise ConnectorError("built-in csv reader accepts no options")
        import pyarrow as pa
        import pyarrow.csv as arrow_csv

        path = self._datasets.get(dataset)
        if path is None or not path.is_file():
            raise ConnectorError("CSV dataset is unavailable")
        deadline = time.monotonic() + limits.max_duration_s
        if limits.max_calls < 1 or time.monotonic() >= deadline:
            raise ConnectorError("CSV sample read budget exceeded", budget=True)
        try:
            # Stage bounded local bytes: Arrow's internal read-ahead cannot read
            # more source data, and no file handle outlives this context.
            with path.open("rb") as stream:
                sample = stream.read(MAX_SAMPLE_BYTES)
            if time.monotonic() >= deadline:
                raise ConnectorError("CSV sample read budget exceeded", budget=True)
            with arrow_csv.open_csv(
                pa.BufferReader(sample),
                read_options=arrow_csv.ReadOptions(block_size=INFERENCE_BLOCK_BYTES, use_threads=False),
                parse_options=arrow_csv.ParseOptions(newlines_in_values=True),
            ) as reader:
                schema = reader.schema
        except (OSError, pa.ArrowException):
            raise ConnectorError("CSV schema is unreadable") from None
        if time.monotonic() >= deadline:
            raise ConnectorError("CSV sample read budget exceeded", budget=True)
        names = schema.names
        if any(not name.strip() for name in names) or len(set(names)) != len(names):
            raise ConnectorError("CSV requires nonempty, unique column headers")
        if 2 + len(schema) > limits.max_records:
            raise ConnectorError("CSV schema record budget exceeded", budget=True)
        return CatalogInfo(
            dataset=dataset,
            display_name=dataset,
            provenance=Provenance(self.source),
            tables=(TableInfo(path.stem, tuple(ColumnInfo(f.name, str(f.type)) for f in schema)),),
            modalities=("tabular",),
        )
