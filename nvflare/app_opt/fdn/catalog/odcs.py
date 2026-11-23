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


"""Site-side reader for native Open Data Contract Standard documents."""

import csv
import json
import tempfile
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from urllib.parse import unquote, urlsplit
from uuid import NAMESPACE_URL, uuid5

from nvflare.app_common.fdn.catalog.common.odcs import MAX_CONTRACT_BYTES, ODCS_VERSION, check_contract
from nvflare.app_common.fdn.catalog.site.connector import ConnectorError
from nvflare.app_opt.fdn.catalog import duckdb_connector, parquet_connector, sql_connector, unity_catalog_connector
from nvflare.app_opt.fdn.catalog._metadata import ReadBudget, local_path, selection


@lru_cache(maxsize=1)
def _validator():
    from jsonschema import Draft201909Validator

    schema = json.loads(files(__package__).joinpath("odcs-json-schema-v3.2.0.json").read_text(encoding="utf-8"))
    return Draft201909Validator(schema)


def _provided(uri):
    parsed = urlsplit(uri)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost") or parsed.query or parsed.fragment:
        raise ValueError("a local contract file URI is required")
    path = Path(unquote(parsed.path))
    if not path.is_absolute() or not path.is_file():
        raise ValueError("contract file unavailable")
    with path.open("rb") as stream:
        contents = stream.read(MAX_CONTRACT_BYTES + 1)
    if len(contents) > MAX_CONTRACT_BYTES:
        raise ValueError("contract file is too large")
    from ruamel.yaml import YAML
    from ruamel.yaml.constructor import SafeConstructor
    from ruamel.yaml.events import AliasEvent

    class ContractConstructor(SafeConstructor):
        # Dates in ODCS are strings. YAML's implicit timestamp conversion would
        # otherwise turn a valid document into a non-JSON Python object.
        yaml_constructors = {
            **SafeConstructor.yaml_constructors,
            "tag:yaml.org,2002:timestamp": SafeConstructor.construct_scalar,
        }

    loader = YAML(typ="safe", pure=True)
    loader.Constructor = ContractConstructor
    if any(isinstance(event, AliasEvent) for event in loader.parse(contents)):
        raise ValueError("contract YAML aliases are unsupported")
    return loader.load(contents)


def _csv_schema(dataset, limits):
    from datacontract.imports.csv_importer import import_csv

    selection(dataset.options, ())
    path = local_path(dataset)
    if not path.is_file():
        raise ConnectorError("CSV file unavailable")
    with path.open("rb") as stream:

        def header_lines():
            remaining = 1024 * 1024
            encoding = "utf-8-sig"
            while line := stream.readline(remaining + 1):
                remaining -= len(line)
                if remaining < 0:
                    raise ConnectorError("CSV header exceeds the read budget", budget=True)
                yield line.decode(encoding)
                encoding = "utf-8"

        names = next(csv.reader(header_lines(), strict=True), [])
    if not names or any(not name.strip() for name in names) or len(set(names)) != len(names):
        raise ConnectorError("CSV requires nonempty, unique column headers")
    if 2 + len(names) > limits.max_records:
        raise ConnectorError("CSV record budget exceeded", budget=True)
    # The importer profiles rows and interpolates column names into SQL. Give it only a
    # header projection with safe identifiers. Restore the original names at publication.
    aliases = {f"column_{i}": name for i, name in enumerate(names)}
    with tempfile.TemporaryDirectory() as temp:
        staged = Path(temp) / "headers.csv"
        if "'" in str(staged):
            raise ConnectorError("CSV importer cannot quote the temporary directory")
        staged.write_text(",".join(aliases) + "\n", encoding="ascii")
        model = import_csv(str(staged))
    columns = [prop.name for prop in model.schema_[0].properties]
    if columns != list(aliases):
        raise ConnectorError("CSV importer changed the header")
    return [{"name": dataset.name, "properties": [{"name": aliases[col]} for col in columns]}]


def _objects(tables):
    objects = []
    # ODCS references use IDs so quoted, dotted or non-ASCII source names are unambiguous.
    refs = {}
    for i, table in enumerate(tables):
        obj = {"id": f"t{i}", "name": table["name"], "properties": []}
        keys = table["primary_key"]
        for j, column in enumerate(table["columns"]):
            name = column["name"]
            prop = {"id": f"c{j}", "name": name, "physicalType": column["type"]}
            if column.get("nullable") is not None:
                prop["required"] = not column["nullable"]
            if name in keys:
                prop.update(primaryKey=True, primaryKeyPosition=keys.index(name) + 1)
            obj["properties"].append(prop)
            refs[table["name"], name] = f"schema/t{i}/properties/c{j}"
        objects.append(obj)
    for table, obj in zip(tables, objects):
        for relation in table["relationships"]:
            obj.setdefault("relationships", []).append(
                {
                    "type": "foreignKey",
                    "from": [refs[table["name"], col] for col in relation["columns"]],
                    "to": [refs[relation["target_table"], col] for col in relation["target_columns"]],
                }
            )
    return objects


def read_odcs(dataset, secret, limits, *, site):
    """Read a steward-approved file or generate a structural contract from the configured source."""
    budget = ReadBudget(limits)
    rules = dataset.usage_rules("catalog") or {}
    if "contract_uri" in rules:
        budget.request()
        contract = _provided(rules["contract_uri"])
    else:
        if dataset.type == "csv":
            budget.request()
            objects = _csv_schema(dataset, limits)
        else:
            readers = {
                "parquet": parquet_connector.read_tables,
                "duckdb": duckdb_connector.read_tables,
                "sql": sql_connector.read_tables,
                "unity-catalog": unity_catalog_connector.read_tables,
            }
            reader = readers.get(dataset.type)
            if reader is None:
                raise ConnectorError("unsupported catalog source")
            objects = _objects(reader(dataset, secret, limits))
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
