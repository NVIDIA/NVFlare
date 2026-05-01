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

"""Static validator for the Auto-FL client contract."""

import argparse
import py_compile
from pathlib import Path

REQUIRED_SNIPPETS = [
    "flare.init()",
    "flare.receive()",
    "flare.send(",
    "strict=True",
    "compute_model_diff",
    "ParamsType.DIFF",
    '"NUM_STEPS_CURRENT_ROUND"',
    "flare.is_evaluate()",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="client.py")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"ERROR: file not found: {path}")
        return 1

    text = path.read_text(encoding="utf-8")
    missing = [snippet for snippet in REQUIRED_SNIPPETS if snippet not in text]
    if missing:
        print("ERROR: client contract validation failed. Missing:")
        for item in missing:
            print(f"  - {item}")
        return 2

    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as exc:
        print(f"ERROR: syntax validation failed: {exc}")
        return 3

    print(f"OK: static client contract checks passed for {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
