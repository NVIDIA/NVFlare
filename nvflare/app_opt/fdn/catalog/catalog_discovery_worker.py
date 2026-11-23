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

"""Run a catalog read in a disposable interpreter with a hard deadline."""

import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from urllib.parse import parse_qsl, urlsplit

from nvflare.app_common.fdn.catalog.common.odcs import MAX_CONTRACT_BYTES
from nvflare.app_common.fdn.catalog.site.source_error import CatalogSourceError


def run_catalog_discovery(dataset, secret, budget, *, site):
    """Run discovery for one dataset in a child process bounded by the remaining site deadline."""
    uri = dataset.uri
    env = os.environ.copy()
    keys = ()
    if dataset.type == "sql" and dataset.path is None:
        keys = ("username", "password")
        url = urlsplit(uri)
        if url.query:
            params = parse_qsl(url.query, keep_blank_values=True, strict_parsing=True)
            modes = {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}
            if len(params) != 1 or params[0][0] != "sslmode" or params[0][1] not in modes:
                raise CatalogSourceError("unsupported PostgreSQL URI query")
            env["PGSSLMODE"] = params[0][1]
            uri = url._replace(query="").geturl()
    elif dataset.type == "unity-catalog" and "token" in dataset.secret_keys:
        keys = ("token",)
    elif dataset.options.get("metadata_source") in ("host", "uri"):
        keys = dataset.secret_keys

    budget.remaining()
    request = {
        "dataset": {
            "study": dataset.study,
            "name": dataset.name,
            "type": dataset.type,
            "uri": uri,
            "path": dataset.path,
            "options": dict(dataset.options),
            "secret_keys": keys,
        },
        "secrets": {key: secret(key) for key in keys},
        "site": site,
        "calls": budget.calls_left,
        "records": budget.records_left,
        "deadline": budget.deadline,
    }

    # DatasetConfig freezes nested mappings. Restore JSON objects (e.g. importer
    # settings) for the worker while retaining the encoder's checks on other types.
    def encode_mapping(value):
        if isinstance(value, Mapping):
            return dict(value)
        raise TypeError("catalog options must contain JSON values")

    data = json.dumps(request, default=encode_mapping).encode()
    # Parent owns temporary CSV/DDL files even when a blocking SDK call is killed.
    with tempfile.TemporaryDirectory() as temp:
        env.update(TMPDIR=temp, TEMP=temp, TMP=temp)
        with subprocess.Popen(
            [sys.executable, "-m", __name__],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
        ) as proc:
            budget.calls_left = 0  # restore the unused grant only after a complete reply
            try:
                output, _ = proc.communicate(data, timeout=budget.remaining())
            except subprocess.TimeoutExpired:
                raise CatalogSourceError("metadata read deadline exceeded", budget=True) from None
            finally:
                if proc.poll() is None:
                    proc.kill()
                proc.wait()

    if proc.returncode or len(output) > MAX_CONTRACT_BYTES + 1024:
        raise CatalogSourceError("catalog worker failed")
    reply = json.loads(output)
    budget.calls_left = reply["calls_left"]
    if "contract" not in reply:
        raise CatalogSourceError("source discovery failed", budget=reply["budget_exceeded"])
    budget.remaining()
    return reply["contract"]


def main():
    # Reserve the pipe for one JSON reply. Third-party diagnostics may include source details.
    with os.fdopen(os.dup(sys.stdout.fileno()), "wb") as reply_stream:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        from nvflare.apis.dataset_config_spec import DatasetConfig
        from nvflare.app_common.fdn.catalog.common.limits import ReadLimits
        from nvflare.app_opt.fdn.catalog._metadata import ReadBudget
        from nvflare.app_opt.fdn.catalog.odcs_contract_builder import build_odcs_contract

        request = json.load(sys.stdin.buffer)
        budget = ReadBudget(ReadLimits(0, request["calls"], request["records"]))
        budget.deadline = request["deadline"]
        try:
            contract = build_odcs_contract(
                DatasetConfig(**request["dataset"]), request["secrets"].__getitem__, budget, site=request["site"]
            )
            reply = {"contract": contract}
        except Exception as exc:
            reply = {"budget_exceeded": isinstance(exc, CatalogSourceError) and exc.budget}
        reply["calls_left"] = budget.calls_left
        reply_stream.write(json.dumps(reply, ensure_ascii=False, allow_nan=False).encode())


if __name__ == "__main__":
    main()
