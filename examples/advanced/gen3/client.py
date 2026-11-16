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

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from gen3.auth import Gen3Auth
from gen3.file import Gen3File

import nvflare.client as flare
from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.np.constants import NPConstants


def load_arrays(paths):
    data = np.concatenate([np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2) for path in paths])
    if data.shape[1] != 3 or not len(data) or not np.isfinite(data).all():
        raise ValueError("Expected finite, nonempty CSV data with columns x0,x1,y")
    return np.column_stack((data[:, :2], np.ones(len(data)))), data[:, 2]


def train_round(weights, features, labels, learning_rate=0.05, steps=5):
    weights = np.array(weights, dtype=np.float64, copy=True)
    for _ in range(steps):
        error = features @ weights - labels
        weights -= learning_rate * (2.0 / len(labels)) * (features.T @ error)
    mse = float(np.mean((features @ weights - labels) ** 2))
    return weights, mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifest.json")
    parser.add_argument("--source", choices=("gen3", "local"), default="gen3")
    args = parser.parse_args()
    flare.init()
    site = flare.system_info()["site_name"]
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = Path(__file__).parent / manifest
    records = json.loads(manifest.read_text())["sites"][site]
    with tempfile.TemporaryDirectory(prefix="nvflare-gen3-") as data_dir:
        if args.source == "local":
            data_dir = os.environ["GEN3_LOCAL_DATA_DIR"]
        else:
            auth = Gen3Auth(
                endpoint=os.environ["GEN3_ENDPOINT"],
                client_credentials=(os.environ["GEN3_CLIENT_ID"], os.environ["GEN3_CLIENT_SECRET"]),
            )
            files = Gen3File(auth_provider=auth)
            for record in records:
                if not files.download_single(record["object_id"], data_dir):
                    raise RuntimeError(f"Download failed for {record['object_id']}")
        paths = [Path(data_dir) / record["file_name"] for record in records]
        features, labels = load_arrays(paths)
        evidence = {"site": site, "samples": len(labels), "objects": records, "source": args.source}
        Path("gen3-inputs.json").write_text(json.dumps(evidence, indent=2) + "\n")
        while flare.is_running():
            model = flare.receive()
            weights, mse = train_round(model.params[NPConstants.NUMPY_KEY], features, labels)
            flare.send(
                flare.FLModel(
                    params={NPConstants.NUMPY_KEY: weights},
                    metrics={"mse": mse},
                    meta={FLMetaKey.NUM_STEPS_CURRENT_ROUND: len(labels)},
                    current_round=model.current_round,
                )
            )


if __name__ == "__main__":
    main()
