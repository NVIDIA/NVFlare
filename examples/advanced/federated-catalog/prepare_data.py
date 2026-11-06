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

"""Create tiny, deterministic site-owned Parquet datasets and publication policies."""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_DATA = "/tmp/nvflare/federated-catalog/data"
SITES = ["site-1", "site-2"]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare_data(data_root):
    root = Path(data_root).resolve()
    for i, site in enumerate(SITES):
        clinical = root / site / "datasets" / "clinical"
        internal = root / site / "datasets" / "internal"
        clinical.mkdir(parents=True, exist_ok=True)
        internal.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {"patient_id": [f"private-{i}-1", f"private-{i}-2"], "age": [40 + i, 60 + i], "bmi": [22.5, 27.0]}
            ),
            clinical / "patients.parquet",
        )
        pq.write_table(pa.table({"private_note": ["synthetic internal record"]}), internal / "records.parquet")
        write_json(
            clinical / "catalog.json",
            {
                "display_name": f"Clinical cohort at {site}",
                "description": "Synthetic cohort for catalog discovery",
                "owner": "private-owner",
                "modalities": ["tabular"],
                "operations": ["statistics"],
            },
        )
        local = root / site / "local"
        # JSON is valid YAML. Process-mode bindings point to site-owned absolute paths.
        write_json(
            local / "study_runtime.yaml",
            {
                "format_version": 2,
                "studies": {
                    "default": {
                        "datasets": {
                            "clinical": {"source": str(clinical), "mode": "ro"},
                            "internal": {"source": str(internal), "mode": "ro"},
                        }
                    }
                },
            },
        )
        write_json(local / "fdn" / "catalog_site.json", {"format_version": 1, "installed_operations": ["statistics"]})
        write_json(
            local / "fdn" / "default.json",
            {
                "format_version": 1,
                "datasets": {
                    "clinical": {
                        "share": True,
                        "exclude_columns": {"patients": ["patient_id"]},
                        "exclude_metadata": ["owner"],
                        "operations": ["statistics"],
                    },
                    "internal": {"share": False},
                },
            },
        )
    print(f"Prepared synthetic data and site policies in {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA)
    args = parser.parse_args()
    prepare_data(args.data_root)
