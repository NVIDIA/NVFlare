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


def _render_dict(data: dict) -> None:
    for k, v in data.items():
        print(f"{k}: {v}")


def _human_output(data: Any) -> None:
    if isinstance(data, list):
        _render_table(data)
    elif isinstance(data, dict):
        _render_dict(data)
    else:
        print(str(data))


def output(data: Any, fmt: Optional[str]) -> None:
    """Print command result in requested format. Used by cert/package commands."""
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


def output_ok(data: Any, fmt: str = "json") -> None:
    """Print success envelope. fmt: 'json' (default) | 'txt'. Used by Phase 0+1 commands."""
    if fmt == "txt":
        _human_output(data)
    else:
        print(json.dumps({"schema_version": SCHEMA_VERSION, "status": "ok", "data": data}))


def output_error(
    error_code: str,
    message: str = None,
    hint: str = None,
    fmt: Optional[str] = None,
    exit_code: int = 1,
    detail: str = None,
    **kwargs,
) -> None:
    """Print error envelope and exit. Never returns.

    Two calling patterns:
    - Cert/package: output_error(code, message, hint, fmt, exit_code=N)
    - Phase 0+1:    output_error(code, fmt="json", job_id="abc", detail="...")
    """
    if message is None:
        # Phase 0+1: look up from ERROR_REGISTRY
        from nvflare.tool.cli_errors import ERROR_REGISTRY

        entry = ERROR_REGISTRY.get(error_code, {"message": error_code, "hint": ""})
        try:
            message = entry["message"].format_map(kwargs) if kwargs else entry["message"]
        except KeyError:
            message = entry["message"]
        if detail:
            message = f"{message} \u2014 {detail}"
        resolved_hint = entry["hint"]
        # Phase 0+1 default format is JSON
        resolved_fmt = fmt if fmt is not None else "json"
    else:
        # Cert/package: explicit message/hint provided
        resolved_hint = hint or ""
        if detail:
            message = f"{message} \u2014 {detail}"
        # fmt=None means text stderr for cert/package commands
        resolved_fmt = fmt if fmt is not None else "txt"

    if resolved_fmt == "json":
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "error",
                    "error_code": error_code,
                    "message": message,
                    "hint": resolved_hint,
                }
            )
        )
    else:
        print(f"ERROR_CODE: {error_code}", file=sys.stderr)
        print(message, file=sys.stderr)
        if resolved_hint:
            print(f"Hint: {resolved_hint}", file=sys.stderr)
    sys.exit(exit_code)


def print_human(msg: str) -> None:
    """Print a human-readable message to stderr (not captured by --output json)."""
    print(msg, file=sys.stderr)


def prompt_yn(question: str) -> bool:
    """Prompt the user with a yes/no question. Returns True for yes."""
    print_human(f"{question} [y/N] ")
    response = sys.stdin.readline().strip().lower()
    return response in ("y", "yes")
