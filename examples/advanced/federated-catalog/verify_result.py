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

"""Optionally check a completed demo's discovery output against its site policies."""

import argparse
import json
from pathlib import Path

from prepare_data import DEFAULT_DATA, SITES


def verify_result(data_root, store_root):
    data_root = Path(data_root)
    output = json.loads((Path(store_root) / "discovery.json").read_text())
    expected = {
        (site, "clinical")
        for site in SITES
        if json.loads((data_root / site / "local/fdn/default.json").read_text())["datasets"]["clinical"]["share"]
    }
    if {(e["client"], e["dataset"]) for e in output["entries"]} != expected:
        raise RuntimeError("Discovery does not match the current site publication policies")
    for entry in output["entries"]:
        metadata = entry.get("metadata")
        if metadata is None or metadata["owner"] is not None:
            raise RuntimeError("Expected available metadata with owner excluded")
        if {c["name"] for t in metadata["tables"] for c in t["columns"]} != {"age", "bmi"}:
            raise RuntimeError("Expected only the policy-approved age and bmi columns")
    print(
        f"Verified {len(output['entries'])} catalog entries: site selection, dataset withholding and column/owner exclusions."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA)
    parser.add_argument("--store-root", default="/tmp/nvflare/federated-catalog/store")
    args = parser.parse_args()
    verify_result(args.data_root, args.store_root)
