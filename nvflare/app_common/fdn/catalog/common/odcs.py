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


"""Checks shared by the site and the server for a JSON-compatible ODCS payload."""

import json

MAX_CONTRACT_BYTES = 4 * 1024 * 1024
ODCS_VERSION = "v3.2.0"


# Fields consumed by the offline viewer. The site performs full ODCS schema validation.
_ELEMENT_TYPES = {
    "name": str,
    "id": str,
    "physicalType": str,
    "logicalType": str,
    "required": bool,
    "primaryKey": bool,
}


def check_contract(contract):
    """Check the transport envelope and count schema records, including nested ODCS properties."""
    if (
        not isinstance(contract, dict)
        or contract.get("apiVersion") != ODCS_VERSION
        or contract.get("kind") != "DataContract"
    ):
        raise ValueError("unsupported ODCS document")
    if not all(isinstance(contract.get(key), str) and contract[key] for key in ("id", "version")):
        raise ValueError("missing contract identity or version")
    encoded = json.dumps(contract, ensure_ascii=False, allow_nan=False).encode()
    if len(encoded) > MAX_CONTRACT_BYTES:
        raise ValueError("ODCS document is too large")

    def count_element(item, *, named=True, table=False):
        if not isinstance(item, dict) or (named and not (isinstance(item.get("name"), str) and item["name"])):
            raise ValueError("invalid ODCS schema element")
        for key, expected in _ELEMENT_TYPES.items():
            if key in item and type(item[key]) is not expected:
                raise ValueError(f"invalid ODCS {key}")
        if "primaryKeyPosition" in item:
            position = item["primaryKeyPosition"]
            if type(position) is not int and not (type(position) is float and position.is_integer()):
                raise ValueError("invalid ODCS primaryKeyPosition")
        relations = item.get("relationships", [])
        if not isinstance(relations, list):
            raise ValueError("invalid ODCS relationships")
        for relation in relations:
            for key in (("from", "to") if table else ("to",)):
                ref = relation.get(key) if isinstance(relation, dict) else None
                refs = ref if isinstance(ref, list) else [ref]
                if not refs or not all(isinstance(value, str) and value for value in refs):
                    raise ValueError("invalid ODCS relationship reference")
        children = item.get("properties", [])
        if not isinstance(children, list):
            raise ValueError("invalid ODCS properties")
        records = 1 + len(relations) + sum(count_element(prop) for prop in children)
        if "items" in item:
            records += count_element(item["items"], named=False)
        if "map" in item:
            mapping = item["map"]
            if not isinstance(mapping, dict) or set(mapping) != {"key", "value"}:
                raise ValueError("invalid ODCS map")
            records += sum(count_element(child, named=False) for child in mapping.values())
        return records

    tables = contract.get("schema", [])
    if not isinstance(tables, list):
        raise ValueError("invalid ODCS schema")
    return 1 + sum(count_element(table, table=True) for table in tables)
