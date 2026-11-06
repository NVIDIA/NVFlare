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

"""Fail unless legacy and migrated JSON outputs are exactly equivalent."""

import argparse
import json
from pathlib import Path


def _canonical(path: Path) -> str:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy", type=Path)
    parser.add_argument("migrated", type=Path)
    args = parser.parse_args()
    if _canonical(args.legacy) != _canonical(args.migrated):
        raise SystemExit(f"Parity check failed: {args.legacy} != {args.migrated}")
    print(f"Exact parity: {args.legacy} == {args.migrated}")


if __name__ == "__main__":
    main()
