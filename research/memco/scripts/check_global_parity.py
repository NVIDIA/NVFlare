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

"""Rebuild global memory from ordered legacy locals and require exact parity."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memco.core import aggregate_local_payloads  # noqa: E402


def _read(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", type=Path, action="append", required=True)
    parser.add_argument("--expected-global", type=Path, required=True)
    parser.add_argument("--promotion-threshold", type=float, default=0.35)
    parser.add_argument("--batch-name", default="collab_dynamic_train")
    args = parser.parse_args()

    actual = aggregate_local_payloads(
        [_read(path) for path in args.local],
        promotion_threshold=args.promotion_threshold,
        batch_name=args.batch_name,
    )
    expected = _read(args.expected_global)
    if actual != expected:
        raise SystemExit("Global memory parity failed")
    print(
        "Exact global parity: "
        f"{actual['candidate_count']} candidates, "
        f"{actual['rule_count']} rules, "
        f"{actual['artifact_count']} artifacts"
    )


if __name__ == "__main__":
    main()
