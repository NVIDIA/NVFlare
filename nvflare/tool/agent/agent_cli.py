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

"""Agent-facing CLI command group."""

import argparse
import sys
from typing import Optional

import nvflare
from nvflare.cli_unknown_cmd_exception import CLIUnknownCmdException
from nvflare.tool.agent.command_registry import agent_commands

CMD_AGENT_INFO = "info"
CMD_AGENT_INSPECT = "inspect"
CMD_AGENT_INSPECT_SOURCE = "source"
CMD_AGENT_INSPECT_DATA = "data"

_AGENT_OUTPUT_MODES = ["json"]
_AGENT_EXAMPLES = [
    "nvflare agent info --format json",
    "nvflare agent inspect source ./train.py --format json",
    "nvflare agent inspect data ./data --format json",
    "nvflare agent info --schema",
]
_agent_parser: Optional[argparse.ArgumentParser] = None
_agent_sub_cmd_parsers = {}


def def_agent_cli_parser(sub_cmd) -> dict:
    """Register the top-level `nvflare agent` command group."""
    global _agent_parser

    from nvflare.tool.agent.inspector import DEFAULT_MAX_FILE_BYTES, DEFAULT_MAX_FILES

    parser = sub_cmd.add_parser(
        "agent",
        description="Agent-facing NVFLARE command surface.",
        help="Agent-facing NVFLARE helpers.",
    )
    parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    agent_subparser = parser.add_subparsers(title="agent subcommands", metavar="", dest="agent_sub_cmd")

    info_parser = agent_subparser.add_parser(
        CMD_AGENT_INFO,
        description="Show the available NVFLARE agent command surface.",
        help="show agent command surface metadata",
    )
    info_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    _agent_sub_cmd_parsers[CMD_AGENT_INFO] = info_parser

    inspect_parser = agent_subparser.add_parser(
        CMD_AGENT_INSPECT,
        description=(
            "Statically inspect local code, FLARE job artifacts, or data directories for agent "
            "routing. Data directories are classified by reading bounded file headers "
            "(metadata only - names, dtypes, counts; never cell values)."
        ),
        help="statically inspect local code, job artifacts, or data directories",
    )
    inspect_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    _agent_sub_cmd_parsers[CMD_AGENT_INSPECT] = inspect_parser

    capability_parsers = inspect_parser.add_subparsers(
        title="inspection capabilities", metavar="", dest="agent_inspect_capability"
    )
    source_parser = capability_parsers.add_parser(
        CMD_AGENT_INSPECT_SOURCE,
        description="Statically inspect source ownership and existing NVFLARE integration.",
        help="inspect local Python source",
    )
    data_parser = capability_parsers.add_parser(
        CMD_AGENT_INSPECT_DATA,
        description="Inspect bounded dataset metadata without reading data values.",
        help="inspect a local data directory",
    )
    for capability_parser in (source_parser, data_parser):
        capability_parser.add_argument("path", help="existing local inspection target")
        capability_parser.add_argument(
            "--redact",
            choices=["on", "off"],
            default="on",
            help="redact secret-like literals and sensitive absolute paths (default: on)",
        )
        capability_parser.add_argument(
            "--max-files",
            type=_PositiveInt,
            default=DEFAULT_MAX_FILES,
            help="bounded walk limit (default: 250)",
        )
        capability_parser.add_argument(
            "--max-file-bytes",
            type=_PositiveInt,
            default=DEFAULT_MAX_FILE_BYTES,
            help="per-file read cap in bytes (default: 524288)",
        )
        capability_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    _agent_sub_cmd_parsers[f"{CMD_AGENT_INSPECT} {CMD_AGENT_INSPECT_SOURCE}"] = source_parser
    _agent_sub_cmd_parsers[f"{CMD_AGENT_INSPECT} {CMD_AGENT_INSPECT_DATA}"] = data_parser

    _agent_parser = parser
    return {"agent": parser}


class _PositiveInt(int):
    def __new__(cls, value: str):
        parsed = int(value)
        if parsed <= 0:
            raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
        return super().__new__(cls, parsed)


def _agent_info_data() -> dict:
    return {
        "nvflare_version": nvflare.__version__,
        "commands": agent_commands(),
    }


def handle_agent_cmd(args) -> None:
    from nvflare.tool.cli_output import output_error_message, output_ok
    from nvflare.tool.cli_schema import handle_schema_flag

    sub_cmd = getattr(args, "agent_sub_cmd", None)

    if sub_cmd is None:
        handle_schema_flag(
            _agent_parser,
            "nvflare agent",
            _AGENT_EXAMPLES,
            sys.argv[1:],
            streaming=False,
            output_modes=_AGENT_OUTPUT_MODES,
            mutating=False,
            idempotent=True,
        )
        output_error_message(
            "AGENT_SUBCOMMAND_REQUIRED",
            "Agent subcommand required.",
            "Run 'nvflare agent --help' or 'nvflare agent info --format json'.",
            exit_code=4,
            include_data=True,
        )
        return

    if sub_cmd == CMD_AGENT_INFO:
        handle_schema_flag(
            _agent_sub_cmd_parsers[CMD_AGENT_INFO],
            "nvflare agent info",
            _AGENT_EXAMPLES,
            sys.argv[1:],
            streaming=False,
            output_modes=_AGENT_OUTPUT_MODES,
            mutating=False,
            idempotent=True,
        )
        output_ok(
            _agent_info_data(),
            code="OK",
            message="NVFLARE agent command surface is available.",
            hint="Use --schema on agent-facing commands to inspect argument contracts.",
        )
        return

    if sub_cmd == CMD_AGENT_INSPECT:
        _handle_agent_inspect_cmd(args, handle_schema_flag, output_error_message, output_ok)
        return

    raise CLIUnknownCmdException(f"unknown agent subcommand: {sub_cmd}")


def _handle_agent_inspect_cmd(args, handle_schema_flag, output_error_message, output_ok) -> None:
    from nvflare.tool.agent.inspector import inspect_data, inspect_source

    capability = getattr(args, "agent_inspect_capability", None)
    if capability not in {None, CMD_AGENT_INSPECT_SOURCE, CMD_AGENT_INSPECT_DATA}:
        output_error_message(
            "AGENT_INSPECT_CAPABILITY_UNKNOWN",
            f"Unknown inspection capability: {capability}",
            "Use 'source' or 'data'.",
            exit_code=4,
            include_data=True,
        )
        return
    command_key = f"{CMD_AGENT_INSPECT} {capability}" if capability else CMD_AGENT_INSPECT
    handle_schema_flag(
        _agent_sub_cmd_parsers[command_key],
        f"nvflare agent {command_key}",
        _AGENT_EXAMPLES,
        sys.argv[1:],
        streaming=False,
        output_modes=_AGENT_OUTPUT_MODES,
        mutating=False,
        idempotent=True,
    )
    if capability is None:
        output_error_message(
            "AGENT_INSPECT_CAPABILITY_REQUIRED",
            "Inspection capability required.",
            "Run 'nvflare agent inspect source --help' or 'nvflare agent inspect data --help'.",
            exit_code=4,
            include_data=True,
        )
        return
    inspect_kwargs = {"redact": getattr(args, "redact", "on") != "off"}
    if getattr(args, "max_files", None):
        inspect_kwargs["max_files"] = args.max_files
    if getattr(args, "max_file_bytes", None):
        inspect_kwargs["max_file_bytes"] = args.max_file_bytes
    try:
        inspector = inspect_source if capability == CMD_AGENT_INSPECT_SOURCE else inspect_data
        data = inspector(args.path, **inspect_kwargs)
    except FileNotFoundError as e:
        output_error_message(
            "AGENT_INSPECT_PATH_NOT_FOUND",
            str(e),
            "Pass an existing local file or directory to inspect.",
            exit_code=4,
            include_data=True,
            recovery_category="FIXABLE_BY_CONFIG",
        )
        return
    except ValueError as e:
        output_error_message(
            "AGENT_INSPECT_INVALID_TARGET",
            str(e),
            "Use a source file or directory for source mode and a real directory for data mode.",
            exit_code=4,
            include_data=True,
            recovery_category="FIXABLE_BY_CONFIG",
        )
        return
    except Exception as e:
        output_error_message(
            "AGENT_INSPECT_FAILED",
            "Static inspection failed.",
            "Check file permissions or reduce the inspected path scope.",
            exit_code=1,
            detail=str(e) if inspect_kwargs["redact"] is False else None,
            include_data=True,
            recovery_category="ENVIRONMENT_FAILURE",
        )
        return

    output_ok(
        data,
        code="OK",
        message="NVFLARE agent inspect completed.",
        hint="Use routing.recommended_skill and routing.reason to choose the next workflow.",
    )
