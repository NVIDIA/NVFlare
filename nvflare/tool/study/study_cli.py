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
import os
import sys
from contextlib import contextmanager

from nvflare.apis.job_def import DEFAULT_STUDY
from nvflare.cli_unknown_cmd_exception import CLIUnknownCmdException
from nvflare.fuel.flare_api.api_spec import (
    AuthenticationError,
    AuthorizationError,
    CommandError,
    InternalError,
    InvalidArgumentError,
    NoConnection,
)
from nvflare.tool.cli_output import output_error, output_error_message, output_ok, output_usage_error
from nvflare.tool.cli_session import add_startup_kit_selection_args, resolve_startup_kit_info_for_args

CMD_STUDY_REGISTER = "register"
CMD_STUDY_ADD_SITE = "add-site"
CMD_STUDY_REMOVE_SITE = "remove-site"
CMD_STUDY_REMOVE = "remove"
CMD_STUDY_LIST = "list"
CMD_STUDY_SHOW = "show"
CMD_STUDY_ADD_USER = "add-user"
CMD_STUDY_REMOVE_USER = "remove-user"
POC_DEFAULT_ORG = "nvidia"

_study_sub_cmd_parsers = {}
_study_handlers = {}
_study_root_parser = None
_JSON_OUTPUT_MODES = ["json"]
_NO_RETRY_TOKEN_SCHEMA = {"supported": False}


class _WideSubcmdFormatter(argparse.HelpFormatter):
    """Ensures subcommand help text starts on the same line as the longest name."""

    def add_arguments(self, actions):
        super().add_arguments(actions)
        # "remove-site"/"remove-user" (11 chars) at indent-4 needs _action_max_length >= 15
        # so that help_position = 17 and action_width = 17-4-2 = 11 >= 11.
        self._action_max_length = max(self._action_max_length, 15)


def _ensure_study_parsers():
    global _study_root_parser
    if _study_sub_cmd_parsers and _study_root_parser is not None:
        return
    root = argparse.ArgumentParser(
        prog="nvflare study",
        formatter_class=_WideSubcmdFormatter,
    )
    _define_study_subcommands(root)
    _study_root_parser = root


def _resolve_startup_kit_dir_for_args(args=None) -> str:
    from nvflare.tool.cli_session import resolve_startup_kit_dir_for_args

    return resolve_startup_kit_dir_for_args(args)


@contextmanager
def _study_session(args):
    from nvflare.tool.cli_output import get_connect_timeout
    from nvflare.tool.cli_session import new_cli_session_for_args

    try:
        sess = new_cli_session_for_args(args=args, timeout=get_connect_timeout(), study=DEFAULT_STUDY)
    except ValueError as e:
        output_error(
            "STARTUP_KIT_MISSING",
            exit_code=4,
            detail=str(e),
            hint=getattr(e, "hint", None),
        )
        raise SystemExit(4)
    try:
        yield sess
    finally:
        sess.close()


def _handle_command_error(e: Exception):
    if isinstance(e, CommandError):
        output_error_message(e.error_code, e.message, hint=e.hint, exit_code=e.exit_code)
    elif isinstance(e, AuthorizationError):
        output_error_message(
            "NOT_AUTHORIZED",
            str(e) or "not authorized",
            hint="Use a startup kit with the required admin role.",
            exit_code=1,
        )
    elif isinstance(e, AuthenticationError):
        output_error("AUTH_FAILED", exit_code=2, detail=str(e))
    elif isinstance(e, NoConnection):
        output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))
    elif isinstance(e, InternalError):
        output_error("INTERNAL_ERROR", exit_code=5, detail=str(e))
    else:
        raise e


def _get_caller_role_from_startup_kit(admin_user_dir: str) -> str:
    from nvflare.apis.workspace import Workspace
    from nvflare.fuel.hci.client.api_spec import AdminConfigKey
    from nvflare.fuel.hci.client.config import secure_load_admin_config
    from nvflare.fuel.hci.security import IdentityKey, get_identity_info
    from nvflare.lighter.utils import cert_to_dict
    from nvflare.private.fed.utils.identity_utils import load_cert_file

    workspace = Workspace(root_dir=admin_user_dir)
    conf = secure_load_admin_config(workspace)
    admin_config = conf.get_admin_config()
    if not admin_config:
        return ""
    cert_path = admin_config.get(AdminConfigKey.CLIENT_CERT)
    if not cert_path or not os.path.isfile(cert_path):
        return ""
    cert = load_cert_file(cert_path)
    identity = get_identity_info(cert_to_dict(cert))
    return (identity or {}).get(IdentityKey.ROLE) or ""


def _try_get_caller_role(args) -> str:
    try:
        admin_user_dir = _resolve_startup_kit_dir_for_args(args)
        return _get_caller_role_from_startup_kit(admin_user_dir)
    except Exception:
        return ""


def _parse_sites_arg(sites_arg):
    if sites_arg is None:
        raise ValueError("--sites is required")
    raw_items = sites_arg if isinstance(sites_arg, list) else [sites_arg]
    sites = []
    for raw_item in raw_items:
        sites.extend(s.strip() for s in str(raw_item).split(",") if s.strip())
    if not sites:
        raise ValueError("--sites must contain at least one site")
    return sites


def _parse_site_org_args(site_org_args):
    if not site_org_args:
        raise ValueError("--site-org is required")
    values = []
    for item in site_org_args:
        item = item.strip()
        if not item:
            continue
        values.append(item)
    if not values:
        raise ValueError("--site-org must contain at least one org group")
    return values


def _output_invalid_lifecycle_args(detail: str, hint: str = "Run with -h for usage."):
    output_error_message("INVALID_ARGS", "Invalid arguments.", hint=hint, exit_code=4, detail=detail)


def _project_admin_site_org_hint(command_label: str, study_name: str) -> str:
    return (
        f"Use: nvflare study {command_label} {study_name} --site-org {POC_DEFAULT_ORG}:site-1,site-2 "
        f"(POC default org: {POC_DEFAULT_ORG})"
    )


def _resolve_lifecycle_inputs(args, command_label: str):
    if args.sites and args.site_org:
        _output_invalid_lifecycle_args("--sites and --site-org are mutually exclusive; provide only one")

    caller_role = _try_get_caller_role(args)
    if caller_role == "org_admin":
        if args.site_org:
            _output_invalid_lifecycle_args(
                "org_admin must use --sites, not --site-org",
                hint=f"Use: nvflare study {command_label} {args.name} --sites site-1 site-2",
            )
        try:
            sites = _parse_sites_arg(args.sites)
        except ValueError:
            _output_invalid_lifecycle_args(
                "org_admin must provide --sites",
                hint=f"Use: nvflare study {command_label} {args.name} --sites site-1 site-2",
            )
        return sites, None

    if caller_role == "project_admin":
        if args.sites:
            _output_invalid_lifecycle_args(
                "project_admin must use --site-org, not --sites",
                hint=_project_admin_site_org_hint(command_label, args.name),
            )
        try:
            site_orgs = _parse_site_org_args(args.site_org)
        except ValueError:
            _output_invalid_lifecycle_args(
                "project_admin must provide --site-org",
                hint=_project_admin_site_org_hint(command_label, args.name),
            )
        return None, site_orgs

    try:
        if args.site_org:
            return None, _parse_site_org_args(args.site_org)
        if args.sites:
            return _parse_sites_arg(args.sites), None
    except ValueError as e:
        _output_invalid_lifecycle_args(str(e))

    _output_invalid_lifecycle_args("provide --sites for org_admin or --site-org for project_admin")


def _run_with_payload(args, func, *func_args, parser=None, output_func=output_ok):
    try:
        with _study_session(args) as sess:
            data = func(sess, *func_args)
        output_func(data)
    except (ValueError, InvalidArgumentError) as e:
        output_usage_error(parser, detail=str(e), exit_code=4)
    except Exception as e:
        _handle_command_error(e)


def _with_startup_kit_info(args, data: dict) -> dict:
    data = dict(data or {})
    data["startup_kit"] = resolve_startup_kit_info_for_args(args)
    return data


def _render_study_list_human(data: dict):
    from nvflare.tool.cli_output import print_human

    data = data or {}
    identity = data.get("identity") or {}
    startup_kit = data.get("startup_kit") or {}
    study_details = data.get("study_details") or []
    studies = data.get("studies") or []

    identity_name = identity.get("name") or "unknown"
    identity_parts = []
    if identity.get("role"):
        identity_parts.append(f"role: {identity['role']}")
    if identity.get("org"):
        identity_parts.append(f"org: {identity['org']}")
    identity_suffix = f" ({', '.join(identity_parts)})" if identity_parts else ""
    print_human(f"Identity: {identity_name}{identity_suffix}")

    startup_label = startup_kit.get("id") or startup_kit.get("path") or "unknown"
    startup_source = startup_kit.get("source")
    if startup_source:
        print_human(f"Startup kit: {startup_label} ({startup_source})")
    else:
        print_human(f"Startup kit: {startup_label}")

    rows = []
    if isinstance(study_details, list):
        rows.extend(detail for detail in study_details if isinstance(detail, dict))
    known_names = {row.get("name") for row in rows}
    for name in studies:
        if name not in known_names:
            rows.append({"name": name})

    if not rows:
        print_human("Studies: none")
        return

    print_human("")
    print_human("Studies:")
    headers = ("NAME", "ROLE", "CAN SUBMIT", "REASON")
    table_rows = []
    for row in rows:
        can_submit = row.get("can_submit_job")
        if can_submit is True:
            can_submit_text = "yes"
        elif can_submit is False:
            can_submit_text = "no"
        else:
            can_submit_text = "-"
        table_rows.append(
            (
                str(row.get("name") or "-"),
                str(row.get("role") or "-"),
                can_submit_text,
                str(row.get("reason") or "-"),
            )
        )

    widths = [len(header) for header in headers]
    for row in table_rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    print_human("  " + "  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print_human("  " + "  ".join("-" * width for width in widths))
    for row in table_rows:
        print_human("  " + "  ".join(cell.ljust(width) for cell, width in zip(row, widths)))


def _output_study_list(data: dict):
    from nvflare.tool.cli_output import is_json_mode

    if is_json_mode():
        output_ok(data)
    else:
        _render_study_list_human(data)


def cmd_register(args):
    from nvflare.tool.cli_schema import handle_schema_flag

    _ensure_study_parsers()
    handle_schema_flag(
        _study_sub_cmd_parsers[CMD_STUDY_REGISTER],
        "nvflare study register",
        ["nvflare study register cancer-research --site-org nvidia:site-1,site-2"],
        sys.argv[1:],
    )
    parser = _study_sub_cmd_parsers[CMD_STUDY_REGISTER]
    sites, site_orgs = _resolve_lifecycle_inputs(args, CMD_STUDY_REGISTER)
    _run_with_payload(
        args,
        lambda s, name, sites, site_orgs: s.register_study(name, sites=sites, site_orgs=site_orgs),
        args.name,
        sites,
        site_orgs,
        parser=parser,
    )


def cmd_add_site(args):
    from nvflare.tool.cli_schema import handle_schema_flag

    _ensure_study_parsers()
    handle_schema_flag(
        _study_sub_cmd_parsers[CMD_STUDY_ADD_SITE],
        "nvflare study add-site",
        ["nvflare study add-site cancer-research --sites hospital-b"],
        sys.argv[1:],
    )
    parser = _study_sub_cmd_parsers[CMD_STUDY_ADD_SITE]
    sites, site_orgs = _resolve_lifecycle_inputs(args, CMD_STUDY_ADD_SITE)
    _run_with_payload(
        args,
        lambda s, name, sites, site_orgs: s.add_study_site(name, sites=sites, site_orgs=site_orgs),
        args.name,
        sites,
        site_orgs,
        parser=parser,
    )


def cmd_remove_site(args):
    from nvflare.tool.cli_schema import handle_schema_flag

    _ensure_study_parsers()
    handle_schema_flag(
        _study_sub_cmd_parsers[CMD_STUDY_REMOVE_SITE],
        "nvflare study remove-site",
        ["nvflare study remove-site cancer-research --sites hospital-b"],
        sys.argv[1:],
    )
    parser = _study_sub_cmd_parsers[CMD_STUDY_REMOVE_SITE]
    sites, site_orgs = _resolve_lifecycle_inputs(args, CMD_STUDY_REMOVE_SITE)
    _run_with_payload(
        args,
        lambda s, name, sites, site_orgs: s.remove_study_site(name, sites=sites, site_orgs=site_orgs),
        args.name,
        sites,
        site_orgs,
        parser=parser,
    )


def cmd_remove(args):
    from nvflare.tool.cli_schema import handle_schema_flag

    _ensure_study_parsers()
    handle_schema_flag(
        _study_sub_cmd_parsers[CMD_STUDY_REMOVE],
        "nvflare study remove",
        ["nvflare study remove cancer-research"],
        sys.argv[1:],
    )
    _run_with_payload(
        args,
        lambda s, name: s.remove_study(name),
        args.name,
        parser=_study_sub_cmd_parsers[CMD_STUDY_REMOVE],
    )


def cmd_list(args):
    from nvflare.tool.cli_schema import handle_schema_flag

    _ensure_study_parsers()
    handle_schema_flag(
        _study_sub_cmd_parsers[CMD_STUDY_LIST],
        "nvflare study list",
        ["nvflare study list"],
        sys.argv[1:],
        output_modes=_JSON_OUTPUT_MODES,
        streaming=False,
        mutating=False,
        idempotent=True,
        retry_token=_NO_RETRY_TOKEN_SCHEMA,
    )
    _run_with_payload(
        args,
        lambda s: _with_startup_kit_info(args, s.list_studies()),
        parser=_study_sub_cmd_parsers[CMD_STUDY_LIST],
        output_func=_output_study_list,
    )


def cmd_show(args):
    from nvflare.tool.cli_schema import handle_schema_flag

    _ensure_study_parsers()
    handle_schema_flag(
        _study_sub_cmd_parsers[CMD_STUDY_SHOW],
        "nvflare study show",
        ["nvflare study show cancer-research"],
        sys.argv[1:],
    )
    _run_with_payload(
        args,
        lambda s, name: s.show_study(name),
        args.name,
        parser=_study_sub_cmd_parsers[CMD_STUDY_SHOW],
    )


def cmd_add_user(args):
    from nvflare.tool.cli_schema import handle_schema_flag

    _ensure_study_parsers()
    handle_schema_flag(
        _study_sub_cmd_parsers[CMD_STUDY_ADD_USER],
        "nvflare study add-user",
        ["nvflare study add-user cancer-research trainer@org_a.com"],
        sys.argv[1:],
    )
    _run_with_payload(
        args,
        lambda s, study, user: s.add_study_user(study, user),
        args.study,
        args.user,
        parser=_study_sub_cmd_parsers[CMD_STUDY_ADD_USER],
    )


def cmd_remove_user(args):
    from nvflare.tool.cli_schema import handle_schema_flag

    _ensure_study_parsers()
    handle_schema_flag(
        _study_sub_cmd_parsers[CMD_STUDY_REMOVE_USER],
        "nvflare study remove-user",
        ["nvflare study remove-user cancer-research trainer@org_a.com"],
        sys.argv[1:],
    )
    _run_with_payload(
        args,
        lambda s, study, user: s.remove_study_user(study, user),
        args.study,
        args.user,
        parser=_study_sub_cmd_parsers[CMD_STUDY_REMOVE_USER],
    )


def _define_study_subcommands(parser):
    sub = parser.add_subparsers(title="study subcommands", metavar="", dest="study_sub_cmd")

    p = sub.add_parser(CMD_STUDY_REGISTER, help="create or merge a study")
    p.add_argument("name")
    p.add_argument("--sites", nargs="+", required=False)
    p.add_argument("--site-org", action="append", default=[])
    add_startup_kit_selection_args(p)
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    _study_sub_cmd_parsers[CMD_STUDY_REGISTER] = p
    _study_handlers[CMD_STUDY_REGISTER] = cmd_register

    p = sub.add_parser(CMD_STUDY_ADD_SITE, help="add sites to a study")
    p.add_argument("name")
    p.add_argument("--sites", nargs="+", required=False)
    p.add_argument("--site-org", action="append", default=[])
    add_startup_kit_selection_args(p)
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    _study_sub_cmd_parsers[CMD_STUDY_ADD_SITE] = p
    _study_handlers[CMD_STUDY_ADD_SITE] = cmd_add_site

    p = sub.add_parser(CMD_STUDY_REMOVE_SITE, help="remove sites from a study")
    p.add_argument("name")
    p.add_argument("--sites", nargs="+", required=False)
    p.add_argument("--site-org", action="append", default=[])
    add_startup_kit_selection_args(p)
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    _study_sub_cmd_parsers[CMD_STUDY_REMOVE_SITE] = p
    _study_handlers[CMD_STUDY_REMOVE_SITE] = cmd_remove_site

    p = sub.add_parser(CMD_STUDY_REMOVE, help="remove a study")
    p.add_argument("name")
    add_startup_kit_selection_args(p)
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    _study_sub_cmd_parsers[CMD_STUDY_REMOVE] = p
    _study_handlers[CMD_STUDY_REMOVE] = cmd_remove

    p = sub.add_parser(CMD_STUDY_LIST, help="list visible studies")
    add_startup_kit_selection_args(p)
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    _study_sub_cmd_parsers[CMD_STUDY_LIST] = p
    _study_handlers[CMD_STUDY_LIST] = cmd_list

    p = sub.add_parser(CMD_STUDY_SHOW, help="show a study")
    p.add_argument("name")
    add_startup_kit_selection_args(p)
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    _study_sub_cmd_parsers[CMD_STUDY_SHOW] = p
    _study_handlers[CMD_STUDY_SHOW] = cmd_show

    p = sub.add_parser(CMD_STUDY_ADD_USER, help="add a study user")
    p.add_argument("study")
    p.add_argument("user")
    add_startup_kit_selection_args(p)
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    _study_sub_cmd_parsers[CMD_STUDY_ADD_USER] = p
    _study_handlers[CMD_STUDY_ADD_USER] = cmd_add_user

    p = sub.add_parser(CMD_STUDY_REMOVE_USER, help="remove a study user")
    p.add_argument("study")
    p.add_argument("user")
    add_startup_kit_selection_args(p)
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    _study_sub_cmd_parsers[CMD_STUDY_REMOVE_USER] = p
    _study_handlers[CMD_STUDY_REMOVE_USER] = cmd_remove_user


def def_study_cli_parser(sub_cmd):
    global _study_root_parser
    parser = sub_cmd.add_parser(
        "study",
        help="manage study registry entries and study-scoped users",
        formatter_class=_WideSubcmdFormatter,
    )
    _study_root_parser = parser
    _define_study_subcommands(parser)
    return {"study": parser}


def handle_study_cmd(args):
    sub_cmd = getattr(args, "study_sub_cmd", None)
    if sub_cmd is None:
        _ensure_study_parsers()
        _study_root_parser.print_help()
        return
    handler = _study_handlers.get(sub_cmd)
    if handler is None:
        raise CLIUnknownCmdException(f"Unknown study subcommand: {sub_cmd}")
    handler(args)
