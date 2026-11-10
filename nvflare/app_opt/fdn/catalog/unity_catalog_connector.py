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

"""Read catalog schemas through the Unity Catalog REST API. No Spark or storage access."""

import math
from ipaddress import ip_address
from urllib.parse import quote, urlsplit

import requests

from nvflare.app_common.fdn.catalog.site.connector import CatalogConnector, ConnectorError
from nvflare.app_opt.fdn.catalog._metadata import ReadBudget, build_catalog, selection


def _keep_request_headers(request):
    """Session auth that changes nothing. A session with no auth of its own lets requests apply a matching .netrc
    login, which would replace the configured Authorization header; any auth of its own turns that off."""
    return request


class UnityCatalogConnector(CatalogConnector):
    """One dataset maps to one UC catalog/schema, optionally restricted to selected tables.

    - ``uri``: the API root, including ``/api/2.1/unity-catalog``.
    - secret ``token``, when configured, is sent as a bearer token. Leave it out for a local server with
      authorization disabled. A bearer token requires HTTPS, except to a loopback address or localhost.
    - options: ``catalog`` and ``schema`` are required; ``tables`` optionally restricts the read.

    ``request_timeout_s`` is a connector argument a site can set where it declares the connector.
    """

    source = "unity-catalog"

    def __init__(self, request_timeout_s=10.0):
        if not math.isfinite(request_timeout_s) or request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive and finite")
        self.request_timeout_s = request_timeout_s

    def read_catalog(self, dataset, secret, limits):
        options = selection(dataset.options, {"catalog", "schema", "tables"}, required={"catalog", "schema"})
        budget = ReadBudget(limits)
        url = dataset.uri.rstrip("/")
        try:
            parsed = urlsplit(url)
        except ValueError:
            raise ConnectorError("Unity Catalog endpoint is invalid") from None
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ConnectorError("Unity Catalog endpoint is not configured")
        headers = {"Accept": "application/json"}
        if "token" in dataset.secret_keys:
            if parsed.scheme != "https":
                try:
                    loopback = ip_address(parsed.hostname).is_loopback
                except ValueError:
                    loopback = parsed.hostname.lower().rstrip(".") == "localhost"
                if not loopback:
                    raise ConnectorError("Unity Catalog bearer-token authentication requires HTTPS")
            try:
                token = secret("token")
            except Exception:
                raise ConnectorError("Unity Catalog credentials are not configured") from None
            if not token:
                raise ConnectorError("Unity Catalog credentials are not configured")
            headers["Authorization"] = f"Bearer {token}"

        try:
            with requests.Session() as session:
                session.auth = _keep_request_headers

                def get(path, params=None):
                    budget.request()
                    timeout = min(self.request_timeout_s, budget.remaining())
                    with session.get(
                        f"{url}/{path}", headers=headers, params=params, timeout=timeout, allow_redirects=False
                    ) as response:
                        if response.status_code != 200:
                            raise ConnectorError("Unity Catalog metadata is unavailable")
                        data = response.json()
                    budget.remaining()
                    if not isinstance(data, dict):
                        raise ConnectorError("Unity Catalog metadata is invalid")
                    return data

                tables = []
                names = set()
                records = 1

                def add(table, expected_name=None):
                    nonlocal records
                    if (
                        not isinstance(table, dict)
                        or table.get("catalog_name") != options["catalog"]
                        or table.get("schema_name") != options["schema"]
                    ):
                        raise ConnectorError("Unity Catalog table scope is invalid")
                    name = table["name"]
                    if name in names or (expected_name is not None and name != expected_name):
                        raise ConnectorError("Unity Catalog table inventory is inconsistent")
                    # Some compatible list APIs omit column details. Fetch them
                    # explicitly rather than publishing an incomplete schema.
                    if "columns" not in table:
                        if expected_name is not None:
                            raise ConnectorError("Unity Catalog column inventory is missing")
                        full_name = ".".join((options["catalog"], options["schema"], name))
                        return add(get(f"tables/{quote(full_name, safe='')}"), expected_name=name)
                    columns = [
                        {"name": column["name"], "type": column["type_text"], "nullable": column.get("nullable")}
                        for column in table["columns"]
                    ]
                    records += 1 + len(columns)
                    budget.records(records)
                    names.add(name)
                    # UC OSS does not expose PK/FK constraints in TableInfo.
                    # Do not infer them from names, rows, or arbitrary properties.
                    tables.append({"name": name, "columns": columns, "primary_key": [], "relationships": []})

                if "tables" in options:
                    budget.records(1 + len(options["tables"]))
                    for name in sorted(options["tables"]):
                        full_name = ".".join((options["catalog"], options["schema"], name))
                        add(get(f"tables/{quote(full_name, safe='')}"), expected_name=name)
                else:
                    params = {"catalog_name": options["catalog"], "schema_name": options["schema"], "max_results": 50}
                    seen_tokens = set()
                    while True:
                        page = get("tables", params)
                        listed = page.get("tables", [])
                        if not isinstance(listed, list):
                            raise ConnectorError("Unity Catalog table inventory is invalid")
                        for table in listed:
                            add(table)
                        token = page.get("next_page_token")
                        if token is None or token == "":
                            break
                        if not isinstance(token, str) or token in seen_tokens:
                            raise ConnectorError("Unity Catalog pagination is inconsistent")
                        seen_tokens.add(token)
                        params = {**params, "page_token": token}
                catalog = build_catalog(
                    dataset.name, self.source, sorted(tables, key=lambda table: table["name"]), limits
                )
                budget.remaining()
                return catalog
        except ConnectorError:
            raise
        except requests.Timeout:
            raise ConnectorError("Unity Catalog metadata request timed out", budget=True) from None
        except Exception:
            # HTTP errors can contain endpoints, server bodies, or credentials.
            raise ConnectorError("Unity Catalog metadata is unavailable") from None
