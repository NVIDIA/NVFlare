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

"""Check the saved synthetic regression model against an independent closed-form result."""

import argparse
import json
from pathlib import Path

import numpy as np


def expected_model(manifest, data_dir, rounds=3):
    # For this synthetic dataset y = X @ truth. Five local gradient steps have
    # error transition (I - 2 * learning_rate * X.T @ X / n)^5. Weighted FedAvg
    # averages these transitions, so its round recurrence can be solved directly.
    transitions = []
    counts = []
    sites = json.loads(Path(manifest).read_text())["sites"]
    for site in ("site-1", "site-2"):
        paths = [Path(data_dir) / record["file_name"] for record in sites[site]]
        data = np.concatenate([np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2) for path in paths])
        features = np.column_stack((data[:, :2], np.ones(len(data))))
        np.testing.assert_allclose(features @ [2.0, -3.0, 0.5], data[:, 2], rtol=1e-12, atol=1e-12)
        transitions.append(np.linalg.matrix_power(np.eye(3) - 0.1 * features.T @ features / len(data), 5))
        counts.append(len(data))
    transition = np.average(transitions, axis=0, weights=counts)
    return (np.eye(3) - np.linalg.matrix_power(transition, rounds)) @ [2.0, -3.0, 0.5]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Saved server npy model")
    parser.add_argument("--control", help="Optional saved model from the pre-staged control")
    parser.add_argument("--data", required=True, help="Directory with synthetic CSVs and manifest.json")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    data = Path(args.data)
    expected = expected_model(data / "manifest.json", data, args.rounds)
    actual = np.load(args.model, allow_pickle=False)
    passed = actual.shape == expected.shape and np.allclose(actual, expected, rtol=1e-6, atol=1e-8)
    if args.control:
        control = np.load(args.control, allow_pickle=False)
        passed = passed and control.shape == actual.shape and np.allclose(actual, control, rtol=1e-6, atol=1e-8)
    print(json.dumps({"passed": bool(passed), "expected": expected.tolist(), "actual": actual.tolist()}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
