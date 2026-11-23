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


"""Generate and validate an ODCS contract from the metadata of one site dataset."""

import csv
import json
import tempfile
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from nvflare.app_common.fdn.catalog.common.odcs import ODCS_VERSION, check_contract
from nvflare.app_common.fdn.catalog.site.source_error import CatalogSourceError
from nvflare.app_opt.fdn.catalog._metadata import local_path, selection
from nvflare.app_opt.fdn.catalog.source_import import import_schema


@lru_cache(maxsize=1)
def _validator():
    from jsonschema import Draft201909Validator

    schema = json.loads(files(__package__).joinpath("odcs-json-schema-v3.2.0.json").read_text(encoding="utf-8"))
    return Draft201909Validator(schema)


def _csv_schema(dataset, budget):
    from datacontract.imports.csv_importer import import_csv

    selection(dataset.options, ())
    path = local_path(dataset)
    if not path.is_file():
        raise CatalogSourceError("CSV file unavailable")
    # newline="" lets csv.reader handle CR, LF and CRLF, including quoted multiline names.
    # Decode strictly only for header lines consumed by the parser, regardless of buffered row bytes.
    with path.open("r", encoding="utf-8-sig", errors="surrogateescape", newline="") as stream:

        def header_lines():
            remaining = 1024 * 1024
            while line := stream.readline(remaining + 1):
                remaining -= len(line.encode("utf-8"))
                if remaining < 0:
                    raise CatalogSourceError("CSV header exceeds the read budget", budget=True)
                yield line

        names = next(csv.reader(header_lines(), strict=True), [])
    if not names or any(not name.strip() for name in names) or len(set(names)) != len(names):
        raise CatalogSourceError("CSV requires nonempty, unique column headers")
    budget.records(2 + len(names))
    # The importer profiles rows and interpolates column names into SQL. Give it only a
    # header projection with safe identifiers. Restore the original names at publication.
    # Batching bounds the upstream per-column profiling cost for wide headers.
    ids = [f"column_{i}" for i in range(len(names))]
    with tempfile.TemporaryDirectory() as temp:
        staged = Path(temp) / "headers.csv"
        if "'" in str(staged):
            raise CatalogSourceError("CSV importer cannot quote the temporary directory")
        for start in range(0, len(ids), 64):
            batch = ids[start : start + 64]
            budget.request()
            staged.write_text(",".join(batch) + "\n", encoding="ascii")
            model = import_csv(str(staged))
            if [p.name for p in model.schema_[0].properties] != batch:
                raise CatalogSourceError("CSV importer changed the header")
    return [{"name": dataset.name, "properties": [{"name": name} for name in names]}]


def build_odcs_contract(dataset, secret, budget, *, site):
    """Generate an ODCS contract from the configured source metadata."""
    if dataset.type == "csv":
        objects = _csv_schema(dataset, budget)
    else:
        objects = import_schema(dataset, secret, budget)
    identity = json.dumps([dataset.study, site, dataset.name], ensure_ascii=False)
    contract = {
        "apiVersion": ODCS_VERSION,
        "kind": "DataContract",
        "id": uuid5(NAMESPACE_URL, identity).urn,
        "version": "1.0.0",
        "status": "draft",
        "name": dataset.name,
        "schema": objects,
    }
    budget.records(check_contract(contract))
    _validator().validate(contract)
    budget.remaining()
    return contract
