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
from typing import List, Optional

from nvflare.tool.cli_contract import SCHEMA_VERSION

_PATH_KEYWORDS = ("dir", "path", "file", "output")


def _infer_type(action: argparse.Action) -> str:
    # Stage-1 schema is inferred from argparse only; richer explicit typing can be added later if
    # commands need stronger MCP/tool contracts than these naming heuristics provide.
    if action.option_strings:
        name = max(action.option_strings, key=len)
    else:
        name = action.dest

    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse._StoreConstAction)):
        return "boolean"
    if action.type is int:
        return "integer"
    if action.type is float:
        return "number"
    name_lower = name.lower()
    if any(kw in name_lower for kw in _PATH_KEYWORDS):
        return "path"
    return "string"


def parser_to_schema(
    parser: argparse.ArgumentParser,
    command: str,
    examples: list[str] | None = None,
    deprecated: bool = False,
    deprecated_message: str = "",
    streaming: bool | None = None,
    output_modes: list[str] | None = None,
    mutating: bool | None = None,
    idempotent: bool | None = None,
    retry_token: dict | None = None,
) -> dict:
    """Serialize an argparse parser to a JSON-compatible schema dict."""
    # argparse exposes parser structure via the private _actions list; this is the standard
    # introspection hook available for building a schema from parser definitions.
    args = []
    for action in parser._actions:
        if isinstance(action, (argparse._HelpAction, argparse._SubParsersAction)):
            continue
        if action.help == argparse.SUPPRESS:
            continue

        is_positional = not action.option_strings
        if is_positional:
            name = action.dest
            required = action.nargs not in (argparse.OPTIONAL, argparse.ZERO_OR_MORE)
        else:
            name = max(action.option_strings, key=len)
            required = bool(getattr(action, "required", False) or getattr(action, "schema_required", False))

        entry = {
            "name": name,
            "type": _infer_type(action),
            "required": required,
            "description": action.help or "",
        }

        if action.default is not None and action.default != argparse.SUPPRESS:
            entry["default"] = action.default
        else:
            entry["default"] = None

        if action.choices is not None:
            entry["choices"] = list(action.choices)

        if action.option_strings and len(action.option_strings) > 1:
            entry["aliases"] = action.option_strings[:-1]

        if action.nargs in ("*", "+", "?"):
            entry["nargs"] = action.nargs

        args.append(entry)

    result = {
        "schema_version": SCHEMA_VERSION,
        "command": command,
        "description": parser.description or "",
        "args": args,
        "examples": examples or [],
    }
    if deprecated:
        result["deprecated"] = True
        result["deprecated_message"] = deprecated_message
    if streaming is not None:
        result["streaming"] = streaming
    if output_modes is not None:
        result["output_modes"] = output_modes
    if mutating is not None:
        result["mutating"] = mutating
    if idempotent is not None:
        result["idempotent"] = idempotent
    if retry_token is not None:
        result["retry_token"] = retry_token
    return result


def handle_schema_flag(
    parser: argparse.ArgumentParser,
    command: str,
    examples: list[str],
    args_list: list[str],
    deprecated: bool = False,
    deprecated_message: str = "",
    streaming: bool | None = None,
    output_modes: list[str] | None = None,
    mutating: bool | None = None,
    idempotent: bool | None = None,
    retry_token: dict | None = None,
) -> None:
    """Handle the pre-parse --schema fast path.

    This must run before parser.parse_args() because many commands want schema discovery even when
    the rest of the required arguments are absent.
    """
    if "--schema" in args_list:
        if parser is None:
            schema = {
                "schema_version": SCHEMA_VERSION,
                "command": command,
                "args": [],
                "examples": examples or [],
            }
            if deprecated:
                schema["deprecated"] = True
                schema["deprecated_message"] = deprecated_message
            if streaming is not None:
                schema["streaming"] = streaming
            if output_modes is not None:
                schema["output_modes"] = output_modes
            if mutating is not None:
                schema["mutating"] = mutating
            if idempotent is not None:
                schema["idempotent"] = idempotent
            if retry_token is not None:
                schema["retry_token"] = retry_token
        else:
            schema = parser_to_schema(
                parser,
                command,
                examples,
                deprecated,
                deprecated_message,
                streaming=streaming,
                output_modes=output_modes,
                mutating=mutating,
                idempotent=idempotent,
                retry_token=retry_token,
            )
        # --schema intentionally bypasses the normal command-output envelope so agent/tool callers
        # always get the raw schema document.
        print(json.dumps(schema, indent=2))
        raise SystemExit(0)
