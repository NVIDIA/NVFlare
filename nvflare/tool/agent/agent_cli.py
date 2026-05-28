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

CMD_AGENT_INFO = "info"
CMD_AGENT_INSPECT = "inspect"
CMD_AGENT_DOCTOR = "doctor"
CMD_AGENT_SKILLS = "skills"
CMD_AGENT_SKILLS_INSTALL = "install"
CMD_AGENT_SKILLS_LIST = "list"

_AGENT_OUTPUT_MODES = ["json"]
_AGENT_EXAMPLES = [
    "nvflare agent info --format json",
    "nvflare agent inspect ./train.py --format json",
    "nvflare agent doctor --format json",
    "nvflare agent doctor --online --format json",
    "nvflare agent skills install --agent codex --dry-run --format json",
    "nvflare agent skills list --agent claude --format json",
    "nvflare agent info --schema",
]
_AGENT_SKILLS_EXAMPLES = [
    "nvflare agent skills install --agent codex --dry-run --format json",
    "nvflare agent skills install --agent claude --skill nvflare-orient --format json",
    "nvflare agent skills list --agent codex --format json",
]
_agent_parser: Optional[argparse.ArgumentParser] = None
_agent_sub_cmd_parsers = {}
_agent_skills_sub_cmd_parsers = {}


def def_agent_cli_parser(sub_cmd) -> dict:
    """Register the top-level `nvflare agent` command group."""
    global _agent_parser

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
        description="Statically inspect local code or FLARE job artifacts for agent routing.",
        help="statically inspect local code or FLARE job artifacts",
    )
    inspect_parser.add_argument("path", help="local file or directory to inspect without executing user code")
    inspect_parser.add_argument(
        "--redact",
        choices=["on", "off"],
        default="on",
        help="redact secret-like literals and sensitive absolute paths (default: on)",
    )
    inspect_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")

    doctor_parser = agent_subparser.add_parser(
        CMD_AGENT_DOCTOR,
        description="Check local NVFLARE agent readiness without modifying state.",
        help="check local NVFLARE agent readiness",
    )
    doctor_parser.add_argument(
        "--online",
        action="store_true",
        help="also run a bounded read-only status check through the selected startup kit",
    )
    _add_startup_kit_selection_args(doctor_parser)
    doctor_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")

    skills_parser = agent_subparser.add_parser(
        CMD_AGENT_SKILLS,
        description="Install and list NVFLARE-owned agent skills.",
        help="install and list NVFLARE-owned agent skills",
    )
    skills_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    skills_subparser = skills_parser.add_subparsers(
        title="agent skills subcommands", metavar="", dest="agent_skills_sub_cmd"
    )
    install_parser = skills_subparser.add_parser(
        CMD_AGENT_SKILLS_INSTALL,
        description="Install NVFLARE-owned skills into a local agent skill directory.",
        help="install NVFLARE-owned skills",
    )
    _add_agent_target_args(install_parser)
    install_parser.add_argument("--skill", help="install one skill by name; omit to install all available skills")
    install_parser.add_argument("--dry-run", action="store_true", help="show the install plan without copying files")
    install_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")

    list_parser = skills_subparser.add_parser(
        CMD_AGENT_SKILLS_LIST,
        description="List available and installed NVFLARE-owned skills for an agent target.",
        help="list NVFLARE-owned skills",
    )
    _add_agent_target_args(list_parser)
    list_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")

    _agent_sub_cmd_parsers[CMD_AGENT_INSPECT] = inspect_parser
    _agent_sub_cmd_parsers[CMD_AGENT_DOCTOR] = doctor_parser
    _agent_sub_cmd_parsers[CMD_AGENT_SKILLS] = skills_parser
    _agent_skills_sub_cmd_parsers[CMD_AGENT_SKILLS_INSTALL] = install_parser
    _agent_skills_sub_cmd_parsers[CMD_AGENT_SKILLS_LIST] = list_parser

    _agent_parser = parser
    return {"agent": parser}


def _add_agent_target_args(parser) -> None:
    from nvflare.tool.agent.skill_manager import SUPPORTED_AGENT_TARGETS

    parser.add_argument(
        "--agent", choices=list(SUPPORTED_AGENT_TARGETS), required=True, help="agent skill target to manage"
    )
    parser.add_argument("--target", help="override the resolved agent skill directory")


def _add_startup_kit_selection_args(parser) -> None:
    from nvflare.tool.cli_session import add_startup_kit_selection_args

    add_startup_kit_selection_args(parser)


def _agent_info_data() -> dict:
    return {
        "nvflare_version": nvflare.__version__,
        "commands": [
            {
                "name": "info",
                "command": "nvflare agent info",
                "status": "available",
                "mutating": False,
                "streaming": False,
            },
            {
                "name": "inspect",
                "command": "nvflare agent inspect",
                "status": "available",
                "mutating": False,
                "streaming": False,
            },
            {
                "name": "doctor",
                "command": "nvflare agent doctor",
                "status": "available",
                "mutating": False,
                "streaming": False,
            },
            {
                "name": "skills install",
                "command": "nvflare agent skills install",
                "status": "available",
                "mutating": True,
                "streaming": False,
            },
            {
                "name": "skills list",
                "command": "nvflare agent skills list",
                "status": "available",
                "mutating": False,
                "streaming": False,
            },
        ],
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

    if sub_cmd == CMD_AGENT_DOCTOR:
        _handle_agent_doctor_cmd(args, handle_schema_flag, output_error_message, output_ok)
        return

    if sub_cmd == CMD_AGENT_SKILLS:
        _handle_agent_skills_cmd(args, handle_schema_flag, output_error_message, output_ok)
        return

    raise CLIUnknownCmdException(f"unknown agent subcommand: {sub_cmd}")


def _handle_agent_inspect_cmd(args, handle_schema_flag, output_error_message, output_ok) -> None:
    from nvflare.tool.agent.inspector import inspect_path

    handle_schema_flag(
        _agent_sub_cmd_parsers[CMD_AGENT_INSPECT],
        "nvflare agent inspect",
        _AGENT_EXAMPLES,
        sys.argv[1:],
        streaming=False,
        output_modes=_AGENT_OUTPUT_MODES,
        mutating=False,
        idempotent=True,
    )
    try:
        data = inspect_path(args.path, redact=getattr(args, "redact", "on") != "off")
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
    except Exception as e:
        output_error_message(
            "AGENT_INSPECT_FAILED",
            "Static inspection failed.",
            "Check file permissions or reduce the inspected path scope.",
            exit_code=1,
            detail=str(e),
            include_data=True,
            recovery_category="ENVIRONMENT_FAILURE",
        )
        return

    output_ok(
        data,
        code="OK",
        message="NVFLARE agent inspect completed.",
        hint="Use the framework and conversion_state fields to choose the next skill.",
    )


def _handle_agent_doctor_cmd(args, handle_schema_flag, output_error_message, output_ok) -> None:
    from nvflare.tool.agent.doctor import doctor_environment

    handle_schema_flag(
        _agent_sub_cmd_parsers[CMD_AGENT_DOCTOR],
        "nvflare agent doctor",
        _AGENT_EXAMPLES,
        sys.argv[1:],
        streaming=False,
        output_modes=_AGENT_OUTPUT_MODES,
        mutating=False,
        idempotent=True,
    )
    try:
        data = doctor_environment(online=getattr(args, "online", False), args=args)
    except Exception as e:
        output_error_message(
            "AGENT_DOCTOR_FAILED",
            "NVFLARE agent doctor failed.",
            "Review local NVFLARE installation and startup-kit configuration.",
            exit_code=1,
            detail=str(e),
            include_data=True,
            recovery_category="ENVIRONMENT_FAILURE",
        )
        return

    output_ok(
        data,
        code="OK",
        message="NVFLARE agent doctor completed.",
        hint="Resolve warning/error findings before production or online workflows.",
    )


def _handle_agent_skills_cmd(args, handle_schema_flag, output_error_message, output_ok) -> None:
    from nvflare.tool.agent.skill_manager import SUPPORTED_AGENT_TARGETS, install_skills, list_skills

    skills_sub_cmd = getattr(args, "agent_skills_sub_cmd", None)
    schema_sub_cmd = _schema_agent_skills_sub_cmd(getattr(args, "_argv", sys.argv[1:]))

    if skills_sub_cmd is None and schema_sub_cmd in _agent_skills_sub_cmd_parsers:
        skills_sub_cmd = schema_sub_cmd

    if skills_sub_cmd is None:
        handle_schema_flag(
            _agent_sub_cmd_parsers[CMD_AGENT_SKILLS],
            "nvflare agent skills",
            _AGENT_SKILLS_EXAMPLES,
            sys.argv[1:],
            streaming=False,
            output_modes=_AGENT_OUTPUT_MODES,
            mutating=False,
            idempotent=True,
        )
        output_error_message(
            "AGENT_SKILLS_SUBCOMMAND_REQUIRED",
            "Agent skills subcommand required.",
            "Run 'nvflare agent skills --help' or 'nvflare agent skills list --agent codex --format json'.",
            exit_code=4,
            include_data=True,
        )
        return

    if skills_sub_cmd == CMD_AGENT_SKILLS_INSTALL:
        handle_schema_flag(
            _agent_skills_sub_cmd_parsers[CMD_AGENT_SKILLS_INSTALL],
            "nvflare agent skills install",
            _AGENT_SKILLS_EXAMPLES,
            sys.argv[1:],
            streaming=False,
            output_modes=_AGENT_OUTPUT_MODES,
            mutating=True,
            idempotent=True,
        )
        try:
            plan = install_skills(
                agent=args.agent,
                skill_name=getattr(args, "skill", None),
                dry_run=getattr(args, "dry_run", False),
                target_dir=getattr(args, "target", None),
            )
        except ValueError as e:
            _output_agent_skill_target_error(output_error_message, getattr(args, "target", None), e)
            return
        if plan["missing"]:
            output_error_message(
                "AGENT_SKILL_NOT_FOUND",
                f"NVFLARE skill not found: {', '.join(plan['missing'])}.",
                "Run 'nvflare agent skills list --agent <codex|claude> --format json' to inspect available skills.",
                exit_code=4,
                data=plan,
                recovery_category="FIXABLE_BY_CONFIG",
            )
            return
        if plan["errors"]:
            output_error_message(
                "AGENT_SKILL_INSTALL_FAILED",
                "One or more NVFLARE skills failed to install.",
                "Review data.errors and rerun the install after fixing the reported filesystem issue.",
                exit_code=1,
                data=plan,
                recovery_category="FIXABLE_BY_ENV",
            )
            return
        output_ok(
            plan,
            code="OK",
            message=(
                "NVFLARE agent skills install plan completed."
                if plan["applied"]
                else "NVFLARE agent skills install dry run completed."
            ),
            hint="Review conflicts before relying on skipped skills." if plan["conflicts"] else "",
        )
        return

    if skills_sub_cmd == CMD_AGENT_SKILLS_LIST:
        handle_schema_flag(
            _agent_skills_sub_cmd_parsers[CMD_AGENT_SKILLS_LIST],
            "nvflare agent skills list",
            _AGENT_SKILLS_EXAMPLES,
            sys.argv[1:],
            streaming=False,
            output_modes=_AGENT_OUTPUT_MODES,
            mutating=False,
            idempotent=True,
        )
        try:
            data = list_skills(agent=args.agent, target_dir=getattr(args, "target", None))
        except ValueError as e:
            _output_agent_skill_target_error(output_error_message, getattr(args, "target", None), e)
            return
        output_ok(
            data,
            code="OK",
            message="NVFLARE agent skills listed.",
            hint=f"Supported agent targets: {', '.join(SUPPORTED_AGENT_TARGETS)}.",
        )
        return

    raise CLIUnknownCmdException(f"unknown agent skills subcommand: {skills_sub_cmd}")


def _output_agent_skill_target_error(output_error_message, target, error: ValueError) -> None:
    output_error_message(
        "AGENT_SKILL_TARGET_INVALID",
        "Invalid agent skill target.",
        "Choose a target directory without symlink components. On macOS, use real paths such as /private/tmp "
        "instead of /tmp.",
        exit_code=4,
        detail=str(error),
        data={"target": target},
        recovery_category="FIXABLE_BY_CONFIG",
    )


def _schema_agent_skills_sub_cmd(argv: list[str]) -> Optional[str]:
    # The top-level CLI bypasses nested argparse parsing for --schema, so infer
    # the third-level agent skills command from argv until that parser path is generalized.
    if "--schema" not in argv or CMD_AGENT_SKILLS not in argv:
        return None
    index = argv.index(CMD_AGENT_SKILLS) + 1
    while index < len(argv):
        token = argv[index]
        if token in _agent_skills_sub_cmd_parsers:
            return token
        index += 1
    return None
