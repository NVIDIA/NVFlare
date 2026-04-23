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
from nvflare.tool.job.job_cli import _resolve_admin_user_and_dir_from_startup_kit
from nvflare.utils.cli_utils import get_startup_kit_dir_for_target

CMD_STUDY_REGISTER = "register"
CMD_STUDY_ADD_SITE = "add-site"
CMD_STUDY_REMOVE_SITE = "remove-site"
CMD_STUDY_REMOVE = "remove"
CMD_STUDY_LIST = "list"
CMD_STUDY_SHOW = "show"
CMD_STUDY_ADD_USER = "add-user"
CMD_STUDY_REMOVE_USER = "remove-user"

_study_sub_cmd_parsers = {}
_study_handlers = {}


def _ensure_study_parsers():
    if _study_sub_cmd_parsers:
        return
    root = argparse.ArgumentParser(prog="nvflare study")
    sub = root.add_subparsers(dest="study_sub_cmd")
    def_study_cli_parser(sub)


def _add_connection_args(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--startup-target",
        choices=["poc", "prod"],
        default=None,
        dest="startup_target",
        help="startup kit target from ~/.nvflare/config.conf",
    )
    group.add_argument(
        "--startup-kit",
        dest="startup_kit",
        default=None,
        help="explicit startup kit location; mutually exclusive with --startup-target",
    )


def _resolve_session_inputs(args):
    startup_target = getattr(args, "startup_target", None)
    startup_override = getattr(args, "startup_kit", None)
    if not startup_override and not startup_target and not os.environ.get("NVFLARE_STARTUP_KIT_DIR"):
        raise ValueError("startup kit must be resolved via --startup-kit, --startup-target, or NVFLARE_STARTUP_KIT_DIR")
    startup = get_startup_kit_dir_for_target(startup_kit_dir=startup_override, target=startup_target)
    return _resolve_admin_user_and_dir_from_startup_kit(startup)


@contextmanager
def _study_session(args):
    from nvflare.tool.cli_output import get_connect_timeout
    from nvflare.tool.cli_session import new_cli_session

    try:
        username, startup_dir = _resolve_session_inputs(args)
    except ValueError as e:
        detail = str(e)
        if "config.conf" in detail or "config file" in detail or "not configured" in detail:
            target = getattr(args, "startup_target", None) or "poc"
            output_error("STARTUP_KIT_NOT_CONFIGURED", exit_code=4, detail=detail, target=target)
        output_error("STARTUP_KIT_MISSING", exit_code=4, detail=detail)
        raise SystemExit(4)

    sess = new_cli_session(
        username=username,
        startup_kit_location=startup_dir,
        timeout=get_connect_timeout(),
    )
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
    from nvflare.fuel.hci.client.api_spec import AdminConfigKey
    from nvflare.fuel.hci.client.config import secure_load_admin_config
    from nvflare.fuel.hci.security import IdentityKey, get_identity_from_cert
    from nvflare.lighter.utils import cert_to_dict
    from nvflare.private.common.workspace import Workspace
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
    identity = get_identity_from_cert(cert_to_dict(cert))
    return (identity or {}).get(IdentityKey.ROLE) or ""


def _try_get_caller_role(args) -> str:
    try:
        _, admin_user_dir = _resolve_session_inputs(args)
        return _get_caller_role_from_startup_kit(admin_user_dir)
    except Exception:
        return ""


def _parse_sites_arg(sites_arg: str):
    if sites_arg is None:
        raise ValueError("--sites is required")
    sites = [s.strip() for s in sites_arg.split(",") if s.strip()]
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


def _run_with_payload(args, func, *func_args, parser=None):
    try:
        with _study_session(args) as sess:
            data = func(sess, *func_args)
        output_ok(data)
    except (ValueError, InvalidArgumentError) as e:
        output_usage_error(parser, detail=str(e), exit_code=4)
    except Exception as e:
        _handle_command_error(e)


def cmd_register(args):
    from nvflare.tool.cli_schema import handle_schema_flag

    _ensure_study_parsers()
    handle_schema_flag(
        _study_sub_cmd_parsers[CMD_STUDY_REGISTER],
        "nvflare study register",
        ["nvflare study register cancer-research --sites hospital-a --startup-target prod"],
        sys.argv[1:],
    )
    parser = _study_sub_cmd_parsers[CMD_STUDY_REGISTER]
    if args.sites and args.site_org:
        output_usage_error(parser, detail="--sites and --site-org are mutually exclusive; provide only one")
        return
    caller_role = _try_get_caller_role(args)
    if caller_role == "org_admin" and args.site_org:
        output_usage_error(parser, detail="org_admin must use --sites, not --site-org")
        return
    if caller_role == "project_admin" and args.sites:
        output_usage_error(parser, detail="project_admin must use --site-org, not --sites")
        return
    try:
        if args.site_org:
            _parse_site_org_args(args.site_org)
        else:
            _parse_sites_arg(args.sites)
    except ValueError as e:
        output_usage_error(parser, detail=str(e), exit_code=4)
        return
    _run_with_payload(
        args,
        lambda s, name, sites, site_orgs: s.register_study(name, sites=sites, site_orgs=site_orgs),
        args.name,
        [s.strip() for s in args.sites.split(",") if s.strip()] if args.sites else None,
        args.site_org or None,
        parser=_study_sub_cmd_parsers[CMD_STUDY_REGISTER],
    )


def cmd_add_site(args):
    from nvflare.tool.cli_schema import handle_schema_flag

    _ensure_study_parsers()
    handle_schema_flag(
        _study_sub_cmd_parsers[CMD_STUDY_ADD_SITE],
        "nvflare study add-site",
        ["nvflare study add-site cancer-research --sites hospital-b --startup-target prod"],
        sys.argv[1:],
    )
    parser = _study_sub_cmd_parsers[CMD_STUDY_ADD_SITE]
    if args.sites and args.site_org:
        output_usage_error(parser, detail="--sites and --site-org are mutually exclusive; provide only one")
        return
    caller_role = _try_get_caller_role(args)
    if caller_role == "org_admin" and args.site_org:
        output_usage_error(parser, detail="org_admin must use --sites, not --site-org")
        return
    if caller_role == "project_admin" and args.sites:
        output_usage_error(parser, detail="project_admin must use --site-org, not --sites")
        return
    try:
        if args.site_org:
            _parse_site_org_args(args.site_org)
        else:
            _parse_sites_arg(args.sites)
    except ValueError as e:
        output_usage_error(parser, detail=str(e), exit_code=4)
        return
    _run_with_payload(
        args,
        lambda s, name, sites, site_orgs: s.add_study_site(name, sites=sites, site_orgs=site_orgs),
        args.name,
        [s.strip() for s in args.sites.split(",") if s.strip()] if args.sites else None,
        args.site_org or None,
        parser=_study_sub_cmd_parsers[CMD_STUDY_ADD_SITE],
    )


def cmd_remove_site(args):
    from nvflare.tool.cli_schema import handle_schema_flag

    _ensure_study_parsers()
    handle_schema_flag(
        _study_sub_cmd_parsers[CMD_STUDY_REMOVE_SITE],
        "nvflare study remove-site",
        ["nvflare study remove-site cancer-research --sites hospital-b --startup-target prod"],
        sys.argv[1:],
    )
    parser = _study_sub_cmd_parsers[CMD_STUDY_REMOVE_SITE]
    if args.sites and args.site_org:
        output_usage_error(parser, detail="--sites and --site-org are mutually exclusive; provide only one")
        return
    caller_role = _try_get_caller_role(args)
    if caller_role == "org_admin" and args.site_org:
        output_usage_error(parser, detail="org_admin must use --sites, not --site-org")
        return
    if caller_role == "project_admin" and args.sites:
        output_usage_error(parser, detail="project_admin must use --site-org, not --sites")
        return
    try:
        if args.site_org:
            _parse_site_org_args(args.site_org)
        else:
            _parse_sites_arg(args.sites)
    except ValueError as e:
        output_usage_error(parser, detail=str(e), exit_code=4)
        return
    _run_with_payload(
        args,
        lambda s, name, sites, site_orgs: s.remove_study_site(name, sites=sites, site_orgs=site_orgs),
        args.name,
        [s.strip() for s in args.sites.split(",") if s.strip()] if args.sites else None,
        args.site_org or None,
        parser=_study_sub_cmd_parsers[CMD_STUDY_REMOVE_SITE],
    )


def cmd_remove(args):
    from nvflare.tool.cli_schema import handle_schema_flag

    _ensure_study_parsers()
    handle_schema_flag(
        _study_sub_cmd_parsers[CMD_STUDY_REMOVE],
        "nvflare study remove",
        ["nvflare study remove cancer-research --startup-target prod"],
        sys.argv[1:],
    )
    _run_with_payload(
        args, lambda s, name: s.remove_study(name), args.name, parser=_study_sub_cmd_parsers[CMD_STUDY_REMOVE]
    )


def cmd_list(args):
    from nvflare.tool.cli_schema import handle_schema_flag

    _ensure_study_parsers()
    handle_schema_flag(
        _study_sub_cmd_parsers[CMD_STUDY_LIST],
        "nvflare study list",
        ["nvflare study list --startup-target prod"],
        sys.argv[1:],
    )
    _run_with_payload(args, lambda s: s.list_studies(), parser=_study_sub_cmd_parsers[CMD_STUDY_LIST])


def cmd_show(args):
    from nvflare.tool.cli_schema import handle_schema_flag

    _ensure_study_parsers()
    handle_schema_flag(
        _study_sub_cmd_parsers[CMD_STUDY_SHOW],
        "nvflare study show",
        ["nvflare study show cancer-research --startup-target prod"],
        sys.argv[1:],
    )
    _run_with_payload(
        args, lambda s, name: s.show_study(name), args.name, parser=_study_sub_cmd_parsers[CMD_STUDY_SHOW]
    )


def cmd_add_user(args):
    from nvflare.tool.cli_schema import handle_schema_flag

    _ensure_study_parsers()
    handle_schema_flag(
        _study_sub_cmd_parsers[CMD_STUDY_ADD_USER],
        "nvflare study add-user",
        ["nvflare study add-user cancer-research trainer@org_a.com --startup-target prod"],
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
        ["nvflare study remove-user cancer-research trainer@org_a.com --startup-target prod"],
        sys.argv[1:],
    )
    _run_with_payload(
        args,
        lambda s, study, user: s.remove_study_user(study, user),
        args.study,
        args.user,
        parser=_study_sub_cmd_parsers[CMD_STUDY_REMOVE_USER],
    )


def def_study_cli_parser(sub_cmd):
    parser = sub_cmd.add_parser("study", help="manage study registry entries and study-scoped users")
    sub = parser.add_subparsers(title="study subcommands", metavar="", dest="study_sub_cmd")

    p = sub.add_parser(CMD_STUDY_REGISTER, help="create or merge a study")
    _add_connection_args(p)
    p.add_argument("name")
    p.add_argument("--sites", required=False)
    p.add_argument("--site-org", action="append", default=[])
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    _study_sub_cmd_parsers[CMD_STUDY_REGISTER] = p
    _study_handlers[CMD_STUDY_REGISTER] = cmd_register

    p = sub.add_parser(CMD_STUDY_ADD_SITE, help="add sites to a study")
    _add_connection_args(p)
    p.add_argument("name")
    p.add_argument("--sites", required=False)
    p.add_argument("--site-org", action="append", default=[])
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    _study_sub_cmd_parsers[CMD_STUDY_ADD_SITE] = p
    _study_handlers[CMD_STUDY_ADD_SITE] = cmd_add_site

    p = sub.add_parser(CMD_STUDY_REMOVE_SITE, help="remove sites from a study")
    _add_connection_args(p)
    p.add_argument("name")
    p.add_argument("--sites", required=False)
    p.add_argument("--site-org", action="append", default=[])
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    _study_sub_cmd_parsers[CMD_STUDY_REMOVE_SITE] = p
    _study_handlers[CMD_STUDY_REMOVE_SITE] = cmd_remove_site

    p = sub.add_parser(CMD_STUDY_REMOVE, help="remove a study")
    _add_connection_args(p)
    p.add_argument("name")
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    _study_sub_cmd_parsers[CMD_STUDY_REMOVE] = p
    _study_handlers[CMD_STUDY_REMOVE] = cmd_remove

    p = sub.add_parser(CMD_STUDY_LIST, help="list visible studies")
    _add_connection_args(p)
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    _study_sub_cmd_parsers[CMD_STUDY_LIST] = p
    _study_handlers[CMD_STUDY_LIST] = cmd_list

    p = sub.add_parser(CMD_STUDY_SHOW, help="show a study")
    _add_connection_args(p)
    p.add_argument("name")
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    _study_sub_cmd_parsers[CMD_STUDY_SHOW] = p
    _study_handlers[CMD_STUDY_SHOW] = cmd_show

    p = sub.add_parser(CMD_STUDY_ADD_USER, help="add a study user")
    _add_connection_args(p)
    p.add_argument("study")
    p.add_argument("user")
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    _study_sub_cmd_parsers[CMD_STUDY_ADD_USER] = p
    _study_handlers[CMD_STUDY_ADD_USER] = cmd_add_user

    p = sub.add_parser(CMD_STUDY_REMOVE_USER, help="remove a study user")
    _add_connection_args(p)
    p.add_argument("study")
    p.add_argument("user")
    p.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    _study_sub_cmd_parsers[CMD_STUDY_REMOVE_USER] = p
    _study_handlers[CMD_STUDY_REMOVE_USER] = cmd_remove_user
    return {"study": parser}


def handle_study_cmd(args):
    sub_cmd = getattr(args, "study_sub_cmd", None)
    if sub_cmd is None:
        raise CLIUnknownCmdException("study subcommand required")
    handler = _study_handlers.get(sub_cmd)
    if handler is None:
        raise CLIUnknownCmdException(f"Unknown study subcommand: {sub_cmd}")
    handler(args)
