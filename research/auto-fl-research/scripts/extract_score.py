#!/usr/bin/env python3
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

"""Extract a comparable scalar score from NVFlare cross-site validation JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

PRIMARY_MODEL_KEY = "SRV_FL_global_model.pt"
METRIC_KEYS = ["accuracy", "val_accuracy", "test_accuracy", "token_f1"]


def find_json(result_root: Path) -> Path:
    candidate = result_root / "server" / "simulate_job" / "cross_site_val" / "cross_val_results.json"
    if candidate.exists():
        return candidate
    if result_root.name == "cross_val_results.json":
        return result_root
    raise FileNotFoundError(f"cross_val_results.json not found under {result_root}")


def extract_score(data) -> float:
    scores = []
    for site_payload in data.values():
        if not isinstance(site_payload, dict):
            continue

        metrics = site_payload.get(PRIMARY_MODEL_KEY)
        if isinstance(metrics, dict):
            score = extract_metric(metrics)
            if score is not None:
                scores.append(score)
                continue

    if not scores:
        raise ValueError(
            f"No comparable metric ({', '.join(METRIC_KEYS)}) found for "
            f"{PRIMARY_MODEL_KEY} in cross-site validation JSON"
        )

    return float(sum(scores) / len(scores))


def extract_metric(metrics) -> float | None:
    for metric_key in METRIC_KEYS:
        if metric_key in metrics:
            return float(metrics[metric_key])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", help="Result root dir or direct path to cross_val_results.json")
    args = parser.parse_args()

    json_path = find_json(Path(args.result_root))
    data = json.loads(json_path.read_text(encoding="utf-8"))
    score = extract_score(data)
    if math.isnan(score):
        raise ValueError("Extracted score is NaN")
    print(f"{score:.6f}")


if __name__ == "__main__":
    main()
