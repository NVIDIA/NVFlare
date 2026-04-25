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

"""nvflare config kit command parser and handlers."""

import os
import sys
from typing import Callable, Dict

from nvflare.cli_unknown_cmd_exception import CLIUnknownCmdException
from nvflare.tool.cli_output import is_json_mode, output_error_message, output_ok, print_human
from nvflare.tool.cli_schema import handle_schema_flag
from nvflare.tool.kit.kit_config import (
    NVFLARE_STARTUP_KIT_DIR,
    STARTUP_KIT_KIND_ADMIN,
    StartupKitConfigError,
    add_startup_kit_entry,
    get_active_startup_kit_id,
    get_cli_config_path,
    get_startup_kit_entries,
    get_startup_kit_status,
    inspect_startup_kit_metadata,
    load_cli_config,
    remove_startup_kit_entry,
    save_cli_config,
    set_active_startup_kit,
)

CMD_KIT_ADD = "add"
CMD_KIT_USE = "use"
CMD_KIT_SHOW = "show"
CMD_KIT_LIST = "list"
CMD_KIT_REMOVE = "remove"
KIT_COMMAND = "nvflare config kit"

_kit_root_parser = None
_kit_sub_cmd_parsers = {}


def _emit_kit_error(e: StartupKitConfigError, exit_code: int = 4):
    output_error_message("INVALID_ARGS", str(e), hint=e.hint, exit_code=exit_code)


def _metadata_for_output(path: str) -> Dict[str, str]:
    metadata = inspect_startup_kit_metadata(path)
    return {
        "identity": metadata.get("identity") or "-",
        "cert_role": metadata.get("cert_role") or "-",
    }


def cmd_kit_add(args):
    handle_schema_flag(
        _kit_sub_cmd_parsers[CMD_KIT_ADD],
        f"{KIT_COMMAND} add",
        [f"{KIT_COMMAND} add cancer_lead /secure/startup_kits/cancer/lead@nvidia.com"],
        sys.argv[1:],
    )

    kit_id = args.kit_id.strip()
    try:
        config = load_cli_config()
        config = add_startup_kit_entry(config, kit_id, args.startup_kit_dir, force=args.force)
        save_cli_config(config)
    except StartupKitConfigError as e:
        _emit_kit_error(e)

    output_ok(
        {
            "registered_startup_kit": kit_id,
            "path": get_startup_kit_entries(config)[kit_id],
            "next_step": f"{KIT_COMMAND} use {kit_id}",
        }
    )


def cmd_kit_use(args):
    handle_schema_flag(
        _kit_sub_cmd_parsers[CMD_KIT_USE],
        f"{KIT_COMMAND} use",
        [f"{KIT_COMMAND} use cancer_lead"],
        sys.argv[1:],
    )

    kit_id = args.kit_id.strip()
    try:
        config = load_cli_config()
        entries = get_startup_kit_entries(config)
        config = set_active_startup_kit(config, kit_id)
        save_cli_config(config)
    except StartupKitConfigError as e:
        _emit_kit_error(e)

    path = entries[kit_id]
    metadata = _metadata_for_output(path)
    output_ok(
        {
            "active_startup_kit": kit_id,
            "identity": metadata["identity"],
            "cert_role": metadata["cert_role"],
            "path": path,
        }
    )


def _print_env_warning():
    env_path = os.getenv(NVFLARE_STARTUP_KIT_DIR)
    if env_path:
        print_human(f"warning: {NVFLARE_STARTUP_KIT_DIR} is set ({env_path})")
        print_human("         normal commands will use this path instead of the active kit above")


def cmd_kit_show(args):
    handle_schema_flag(
        _kit_sub_cmd_parsers[CMD_KIT_SHOW],
        f"{KIT_COMMAND} show",
        [f"{KIT_COMMAND} show"],
        sys.argv[1:],
    )

    try:
        config = load_cli_config()
    except StartupKitConfigError as e:
        _emit_kit_error(e)

    active = get_active_startup_kit_id(config)
    entries = get_startup_kit_entries(config)

    if not active:
        _print_env_warning()
        if is_json_mode():
            output_ok({"active": None, "config_file": str(get_cli_config_path())})
        else:
            print_human("No active startup kit.")
            print_human(f"Hint: Run {KIT_COMMAND} use <id>.")
        return

    path = entries.get(active)
    data = {"active": active, "path": path or "-", "config_file": str(get_cli_config_path())}
    if path is None:
        data.update({"status": "unregistered", "hint": f"run {KIT_COMMAND} list, then {KIT_COMMAND} use <id>"})
    else:
        status, normalized_path, metadata = get_startup_kit_status(path)
        data["status"] = status
        if normalized_path:
            data["path"] = normalized_path
        if status == "ok":
            data["identity"] = metadata.get("identity") or "-"
            data["cert_role"] = metadata.get("cert_role") or "-"
        else:
            data["identity"] = "-"
            data["cert_role"] = "-"
            data["hint"] = f"run {KIT_COMMAND} use <id> or {KIT_COMMAND} remove {active}"

    _print_env_warning()
    output_ok(data)


def cmd_kit_list(args):
    handle_schema_flag(
        _kit_sub_cmd_parsers[CMD_KIT_LIST],
        f"{KIT_COMMAND} list",
        [f"{KIT_COMMAND} list"],
        sys.argv[1:],
    )

    try:
        config = load_cli_config()
    except StartupKitConfigError as e:
        _emit_kit_error(e)

    active = get_active_startup_kit_id(config)
    entries = get_startup_kit_entries(config)
    rows = []
    for kit_id, path in sorted(entries.items()):
        status, normalized_path, metadata = get_startup_kit_status(path)
        # Valid site/server kits are service identities, not CLI user identities.
        # Keep missing/invalid rows visible so users can clean stale registrations.
        if metadata.get("kind") != STARTUP_KIT_KIND_ADMIN and status == "ok":
            continue
        rows.append(
            {
                "active": "*" if kit_id == active else "",
                "id": kit_id,
                "status": status,
                "identity": metadata.get("identity") or "-",
                "cert_role": metadata.get("cert_role") or "-",
                "path": normalized_path or path,
            }
        )

    if not rows and not is_json_mode():
        print_human("No startup kits registered.")
    output_ok(rows)


def cmd_kit_remove(args):
    handle_schema_flag(
        _kit_sub_cmd_parsers[CMD_KIT_REMOVE],
        f"{KIT_COMMAND} remove",
        [f"{KIT_COMMAND} remove cancer_lead"],
        sys.argv[1:],
    )

    kit_id = args.kit_id.strip()
    try:
        config = load_cli_config()
        was_active = get_active_startup_kit_id(config) == kit_id
        config = remove_startup_kit_entry(config, kit_id)
        save_cli_config(config)
    except StartupKitConfigError as e:
        _emit_kit_error(e)

    data = {"removed_startup_kit": kit_id}
    if was_active:
        data["warning"] = "no active startup kit is configured"
        data["next_step"] = f"{KIT_COMMAND} use <id>"
    output_ok(data)


_KIT_HANDLERS: Dict[str, Callable] = {
    CMD_KIT_ADD: cmd_kit_add,
    CMD_KIT_USE: cmd_kit_use,
    CMD_KIT_SHOW: cmd_kit_show,
    CMD_KIT_LIST: cmd_kit_list,
    CMD_KIT_REMOVE: cmd_kit_remove,
}


def def_kit_cli_parser(sub_cmd):
    global _kit_root_parser
    parser = sub_cmd.add_parser("kit", help="manage local startup kit registrations")
    _kit_root_parser = parser
    kit_subparser = parser.add_subparsers(title="kit subcommands", metavar="", dest="kit_sub_cmd")

    add_parser = kit_subparser.add_parser(CMD_KIT_ADD, help="register a startup kit path")
    add_parser.add_argument("kit_id", help="local startup kit ID")
    add_parser.add_argument("startup_kit_dir", help="admin startup kit directory")
    add_parser.add_argument("--force", action="store_true", help="replace an existing local registration")
    add_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    _kit_sub_cmd_parsers[CMD_KIT_ADD] = add_parser

    use_parser = kit_subparser.add_parser(CMD_KIT_USE, help="activate a registered startup kit")
    use_parser.add_argument("kit_id", help="local startup kit ID")
    use_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    _kit_sub_cmd_parsers[CMD_KIT_USE] = use_parser

    show_parser = kit_subparser.add_parser(CMD_KIT_SHOW, help="show the configured active startup kit")
    show_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    _kit_sub_cmd_parsers[CMD_KIT_SHOW] = show_parser

    list_parser = kit_subparser.add_parser(CMD_KIT_LIST, help="list registered startup kits")
    list_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    _kit_sub_cmd_parsers[CMD_KIT_LIST] = list_parser

    remove_parser = kit_subparser.add_parser(CMD_KIT_REMOVE, help="remove a local startup kit registration")
    remove_parser.add_argument("kit_id", help="local startup kit ID")
    remove_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    _kit_sub_cmd_parsers[CMD_KIT_REMOVE] = remove_parser

    return {"kit": parser}


def handle_kit_cmd(args):
    sub_cmd = getattr(args, "kit_sub_cmd", None)
    if "--schema" in getattr(args, "_argv", []) and sub_cmd is None:
        handle_schema_flag(
            _kit_root_parser,
            KIT_COMMAND,
            [
                f"{KIT_COMMAND} add cancer_lead /secure/startup_kits/cancer/lead@nvidia.com",
                f"{KIT_COMMAND} use cancer_lead",
                f"{KIT_COMMAND} list",
            ],
            sys.argv[1:],
        )

    handler = _KIT_HANDLERS.get(sub_cmd)
    if handler:
        handler(args)
    elif sub_cmd is None:
        _kit_root_parser.print_help()
    else:
        raise CLIUnknownCmdException("invalid kit command")
