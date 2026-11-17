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

"""The catalog result: a folder of the sites' ODCS contracts, with a status per site.

The server writes it into the job's workspace, so it is stored and downloaded as a job output::

    catalog/
      sites.json                 {"site-1": {"status": "OK", "unavailable": []}, "site-2": {"status": "FAILED", ...}}
      site-1.omop.odcs.yaml
      site-1.claims.odcs.yaml
"""

import json
import os
import re
import shutil
from pathlib import Path

from nvflare.apis.fl_context import FLContext

CATALOG_DIR = "catalog"
STATUS_FILE = "sites.json"

_SAFE_FILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def write_results(fl_ctx: FLContext, replies: dict) -> None:
    """Write the result of the running job. ``replies`` maps each site to its reply, or None if it gave none."""
    catalog_dir = os.path.join(fl_ctx.get_workspace().get_run_dir(fl_ctx.get_job_id()), CATALOG_DIR)
    os.makedirs(catalog_dir, exist_ok=True)
    sites = {}
    for site in sorted(replies):
        reply = replies[site]
        if reply is None:
            sites[site] = {"status": "FAILED", "unavailable": []}
            continue
        unavailable = set(reply["unavailable"])
        for name, contract in reply["contracts"].items():
            if _SAFE_FILE_NAME.fullmatch(name):
                Path(catalog_dir, f"{site}.{name}.odcs.yaml").write_text(contract, encoding="utf-8")
            else:
                unavailable.add(name)
        sites[site] = {"status": "OK", "unavailable": sorted(unavailable)}
    Path(catalog_dir, STATUS_FILE).write_text(json.dumps(sites, indent=2), encoding="utf-8")


def download_catalog(run, output_dir: str | None = None) -> str:
    """Wait for a catalog job, download its outputs and return the catalog folder; also copy it to ``output_dir``."""
    result_dir = run.get_result(clean_up=False, outputs_only=True)
    found = sorted(Path(result_dir).rglob(f"{CATALOG_DIR}/{STATUS_FILE}")) if result_dir else []
    if not found:
        raise RuntimeError(f"catalog job {run.job_id} has no catalog; its status is {run.get_status()}")
    catalog_dir = str(found[0].parent)
    if output_dir is None:
        return catalog_dir
    return shutil.copytree(catalog_dir, output_dir, dirs_exist_ok=True)
