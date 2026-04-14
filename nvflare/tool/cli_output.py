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

import json
import sys
from typing import Any, Optional

SCHEMA_VERSION = "1"


def _render_table(data: Any) -> None:
    if isinstance(data, dict):
        for k, v in data.items():
            print(f"{k}: {v}")
    elif isinstance(data, list):
        if not data:
            return
        if isinstance(data[0], dict):
            keys = list(data[0].keys())
            widths = [max(len(k), max(len(str(r.get(k, ""))) for r in data)) for k in keys]
            header = "  ".join(k.ljust(w) for k, w in zip(keys, widths))
            print(header)
            print("-" * len(header))
            for row in data:
                print("  ".join(str(row.get(k, "")).ljust(w) for k, w in zip(keys, widths)))
        else:
            for item in data:
                print(item)
    else:
        print(str(data))


def output(data: Any, fmt: Optional[str]) -> None:
    """Print command result in requested format."""
    if fmt == "json":
        print(json.dumps({"schema_version": SCHEMA_VERSION, "status": "ok", "data": data}))
    elif fmt == "quiet":
        if isinstance(data, dict):
            print(next(iter(data.values()), ""))
        elif isinstance(data, list):
            print(data[0] if data else "")
        else:
            print(str(data))
    else:
        _render_table(data)


def output_error(
    error_code: str,
    message: str,
    hint: str,
    fmt: Optional[str],
    exit_code: int = 1,
) -> None:
    """Print error envelope and exit. Never returns."""
    if fmt == "json":
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "error",
                    "error_code": error_code,
                    "message": message,
                    "hint": hint,
                }
            )
        )
    else:
        print(f"ERROR_CODE: {error_code}", file=sys.stderr)
        print(message, file=sys.stderr)
        print(f"Hint: {hint}", file=sys.stderr)
    sys.exit(exit_code)
