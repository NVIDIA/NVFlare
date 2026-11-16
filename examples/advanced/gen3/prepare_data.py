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

"""Generate reproducible synthetic regression data and its approved object manifest."""

import argparse
import hashlib
import json
import uuid
from pathlib import Path

import numpy as np


def prepare_data(output_dir, samples=128):
    if samples <= 0:
        raise ValueError("samples must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"format_version": 1, "sites": {}}
    for number, site in enumerate(("site-1", "site-2"), start=1):
        features = np.random.default_rng(2026 + number).normal(size=(samples, 2))
        labels = features @ np.array([2.0, -3.0]) + 0.5
        path = output / f"{site}.csv"
        np.savetxt(path, np.column_stack((features, labels)), delimiter=",", header="x0,x1,y", comments="", fmt="%.17g")
        manifest["sites"][site] = [
            {
                "object_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"nvflare-gen3-demo/{site}")),
                "file_name": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        ]
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data")
    parser.add_argument("--samples", type=int, default=128)
    args = parser.parse_args()
    prepare_data(args.output, args.samples)
