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

import sys
import time
from contextlib import contextmanager

import nvflare
from nvflare.tool.cli_output import output_error, output_error_message, output_ok, output_usage_error

CMD_SYSTEM_STATUS = "status"
CMD_SYSTEM_RESOURCES = "resources"
CMD_SYSTEM_SHUTDOWN = "shutdown"
CMD_SYSTEM_RESTART = "restart"
CMD_SYSTEM_REMOVE_CLIENT = "remove-client"
CMD_SYSTEM_VERSION = "version"
CMD_SYSTEM_LOG_CONFIG = "log-config"

_system_sub_cmd_parsers = {}


def _add_system_connection_args(parser):
    parser.add_argument(
        "--startup-kit",
        "--startup_kit",  # backward compat
        dest="startup_kit",
        default=None,
        help="path to the admin startup kit directory (overrides target-based config lookup)",
    )
    parser.add_argument(
        "--startup-target",
        "--startup_target",  # backward compat
        choices=["poc", "prod"],
        default=None,
        dest="startup_target",
        help="startup kit target to use from config.conf when --startup-kit is not supplied",
    )


def def_system_cli_parser(system_parser):
    """system_parser is already created in cli.py — add subcommands here."""
    _add_system_connection_args(system_parser)
    sub = system_parser.add_subparsers(title="system subcommands", metavar="", dest="system_sub_cmd")

    # status
    p = sub.add_parser(CMD_SYSTEM_STATUS, help="show server and client status")
    _add_system_connection_args(p)
    p.add_argument("target", nargs="?", choices=["server", "client"], default=None)
    p.add_argument("client_names", nargs="*", default=[])
    p.add_argument("--schema", action="store_true")
    _system_sub_cmd_parsers[CMD_SYSTEM_STATUS] = p

    # resources
    p = sub.add_parser(CMD_SYSTEM_RESOURCES, help="show server and client resource usage")
    _add_system_connection_args(p)
    p.add_argument("target", nargs="?", choices=["server", "client"], default=None)
    p.add_argument("client_names", nargs="*", default=[])
    p.add_argument("--schema", action="store_true")
    _system_sub_cmd_parsers[CMD_SYSTEM_RESOURCES] = p

    # shutdown
    p = sub.add_parser(CMD_SYSTEM_SHUTDOWN, help="shut down server, clients, or all")
    _add_system_connection_args(p)
    p.add_argument("target", choices=["server", "client", "all"])
    p.add_argument("client_names", nargs="*", default=[])
    p.add_argument("--force", action="store_true")
    p.add_argument("--schema", action="store_true")
    _system_sub_cmd_parsers[CMD_SYSTEM_SHUTDOWN] = p

    # restart
    p = sub.add_parser(CMD_SYSTEM_RESTART, help="restart server, clients, or all")
    _add_system_connection_args(p)
    p.add_argument("target", choices=["server", "client", "all"])
    p.add_argument("client_names", nargs="*", default=[])
    p.add_argument("--force", action="store_true")
    p.add_argument("--schema", action="store_true")
    _system_sub_cmd_parsers[CMD_SYSTEM_RESTART] = p

    # remove-client
    p = sub.add_parser(CMD_SYSTEM_REMOVE_CLIENT, help="remove a client from the federation")
    _add_system_connection_args(p)
    p.add_argument("client_name", help="name of the client to remove")
    p.add_argument("--force", action="store_true")
    p.add_argument("--schema", action="store_true")
    _system_sub_cmd_parsers[CMD_SYSTEM_REMOVE_CLIENT] = p

    # version
    p = sub.add_parser(CMD_SYSTEM_VERSION, help="show NVFlare version on each remote site")
    _add_system_connection_args(p)
    p.add_argument("--site", default="all", help="server, a client name, or all")
    p.add_argument("--schema", action="store_true")
    _system_sub_cmd_parsers[CMD_SYSTEM_VERSION] = p

    # log-config
    p = sub.add_parser(CMD_SYSTEM_LOG_CONFIG, help="change logging level on server or client sites")
    _add_system_connection_args(p)
    p.add_argument(
        "level",
        nargs="?",
        default=None,
        help="DEBUG, INFO, WARNING, ERROR, CRITICAL, concise, msg_only, full, verbose, reload",
    )
    p.add_argument("--site", default="all", help="server, a client name, or all")
    p.add_argument("--schema", action="store_true")
    _system_sub_cmd_parsers[CMD_SYSTEM_LOG_CONFIG] = p

    return _system_sub_cmd_parsers


def _confirm_or_force(prompt, args):
    """Prompt for confirmation unless --force is set."""
    if args.force:
        return
    if not sys.stdin.isatty():
        output_error(
            "INVALID_ARGS",
            exit_code=4,
            detail="non-interactive mode requires --force",
        )
        raise SystemExit(4)
    from nvflare.tool.cli_output import prompt_yn

    if not prompt_yn(prompt):
        raise SystemExit(0)


def _get_system_session(args=None):
    """Create a secure session using the startup kit."""
    from nvflare.tool.cli_output import get_connect_timeout
    from nvflare.tool.cli_session import new_cli_session
    from nvflare.utils.cli_utils import get_startup_kit_dir_for_target

    username = None
    startup = None

    try:
        from nvflare.tool.job.job_cli import _resolve_admin_user_and_dir_from_startup_kit

        startup_target = getattr(args, "startup_target", None) or "poc"
        startup_override = getattr(args, "startup_kit", None)
        startup = get_startup_kit_dir_for_target(startup_kit_dir=startup_override, target=startup_target)
        username, startup = _resolve_admin_user_and_dir_from_startup_kit(startup)
    except ValueError as e:
        output_error("STARTUP_KIT_MISSING", exit_code=4, detail=str(e))
        raise SystemExit(4)
    except Exception:
        output_error(
            "STARTUP_KIT_MISSING",
            exit_code=4,
            detail="admin username could not be resolved from the startup kit",
        )
        raise SystemExit(4)

    timeout = get_connect_timeout()
    return new_cli_session(username=username, startup_kit_location=startup, timeout=timeout)


@contextmanager
def _system_session(args=None):
    sess = _get_system_session(args)
    try:
        yield sess
    finally:
        if sess is not None:
            sess.close()


def _fmt_ts(ts):
    if not ts:
        return "unknown"
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(ts)))
    except Exception:
        return str(ts)


def _render_status_human(result, target_type):
    from nvflare.tool.cli_output import print_human

    jobs = result.get("jobs") or []
    clients = result.get("clients") or []
    client_status = result.get("client_status") or []

    client_info_by_name = {}
    for c in clients:
        if isinstance(c, dict):
            client_info_by_name[c.get("client_name")] = c

    client_status_by_name = {}
    for c in client_status:
        if isinstance(c, dict):
            client_status_by_name[c.get("client_name")] = c

    if target_type in ("all", "server"):
        print_human(f"Engine status: {result.get('server_status', 'unknown')}")
        print_human(f"Start time: {_fmt_ts(result.get('server_start_time'))}")
        job_id_w = max([len("JOB_ID")] + [len(str(job.get("job_id", "?"))) for job in jobs if isinstance(job, dict)])
        app_name_w = max(
            [len("APP NAME")] + [len(str(job.get("app_name", "?"))) for job in jobs if isinstance(job, dict)]
        )
        separator = f"+-{'-' * job_id_w}-+-{'-' * app_name_w}-+"
        print_human(separator)
        print_human(f"| {'JOB_ID':<{job_id_w}} | {'APP NAME':<{app_name_w}} |")
        print_human(separator)
        for job in jobs:
            if isinstance(job, dict):
                print_human(
                    f"| {str(job.get('job_id', '?')):<{job_id_w}} | {str(job.get('app_name', '?')):<{app_name_w}} |"
                )
        print_human(separator)

    if target_type in ("all", "client"):
        if target_type == "all":
            client_count = len(clients)
        else:
            client_count = len(clients) if clients else len(client_status_by_name)

        if target_type == "all":
            print_human(f"Registered clients: {client_count}")
        else:
            print_human(f"Clients: {client_count}")

        if clients or client_status:
            print_human("")
            names = []
            seen = set()
            for c in clients:
                name = c.get("client_name") if isinstance(c, dict) else None
                if name and name not in seen:
                    names.append(name)
                    seen.add(name)
            for c in client_status:
                name = c.get("client_name") if isinstance(c, dict) else None
                if name and name not in seen:
                    names.append(name)
                    seen.add(name)

            client_w = max([len("CLIENT")] + [len(name) for name in names]) if names else len("CLIENT")
            fqcn_w = (
                max([len("FQCN")] + [len((client_info_by_name.get(name, {}) or {}).get("fqcn", "-")) for name in names])
                if names
                else len("FQCN")
            )
            last_seen_w = len("LAST CONNECT TIME")
            print_human(f"{'CLIENT':<{client_w}} {'FQCN':<{fqcn_w}} {'LAST CONNECT TIME':<{last_seen_w}} STATUS")

            for name in names:
                info = client_info_by_name.get(name, {})
                status = client_status_by_name.get(name, {}).get("status", "unknown")
                fqcn = info.get("fqcn", "-")
                last_seen = _fmt_ts(info.get("client_last_conn_time")) if info else "-"
                print_human(f"{name:<{client_w}} {fqcn:<{fqcn_w}} {last_seen:<{last_seen_w}} {status}")


def _output_system_status(result, target_type):
    from nvflare.tool.cli_output import is_json_mode

    if is_json_mode():
        output_ok(result)
    else:
        _render_status_human(result, target_type)


def _render_version_human(result):
    from nvflare.tool.cli_output import print_human

    sites = result.get("sites") or []
    compatible = result.get("compatible")
    mismatched_sites = result.get("mismatched_sites")
    admin_version = result.get("admin_version", "unknown")

    print_human("Versions")
    print_human(f"  {'SITE':<16} VERSION")
    for entry in sites:
        if isinstance(entry, dict):
            print_human(f"  {entry.get('site', '?'):<16} {entry.get('version', 'unknown')}")

    print_human("")
    print_human(f"Admin version: {admin_version}")
    if compatible is not None:
        print_human(f"Compatible: {'yes' if compatible else 'no'}")
        if mismatched_sites:
            print_human(f"Mismatched sites: {', '.join(mismatched_sites)}")
        else:
            print_human("Mismatched sites: none")


def _output_system_version(result):
    from nvflare.tool.cli_output import is_json_mode

    if is_json_mode():
        output_ok(result)
    else:
        _render_version_human(result)


def cmd_system_status(args):
    from nvflare.fuel.flare_api.api_spec import AuthenticationError, NoConnection
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        _system_sub_cmd_parsers.get(CMD_SYSTEM_STATUS),
        "nvflare system status",
        ["nvflare system status", "nvflare system status server"],
        sys.argv[1:],
    )

    target_type = getattr(args, "target", None) or "all"
    client_names = getattr(args, "client_names", [])

    try:
        with _system_session(args) as sess:
            result = sess.check_status(target_type, client_names if client_names else None)
    except AuthenticationError:
        raise
    except NoConnection as e:
        output_error_message(
            "CONNECTION_FAILED",
            message="Could not connect to the FLARE server.",
            hint="Start the server or verify the admin startup kit endpoint.",
            exit_code=2,
            detail=str(e),
        )
        raise SystemExit(2)
    except Exception as e:
        output_error("INTERNAL_ERROR", exit_code=5, detail=str(e))
        raise SystemExit(5)

    _output_system_status(result, target_type)


def cmd_system_resources(args):
    from nvflare.fuel.flare_api.api_spec import AuthenticationError, NoConnection
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        _system_sub_cmd_parsers.get(CMD_SYSTEM_RESOURCES),
        "nvflare system resources",
        ["nvflare system resources", "nvflare system resources client"],
        sys.argv[1:],
    )

    target_type = getattr(args, "target", None) or "all"
    client_names = getattr(args, "client_names", [])

    try:
        with _system_session(args) as sess:
            result = sess.report_resources(target_type, client_names if client_names else None)
    except (AuthenticationError, NoConnection):
        raise
    except Exception as e:
        output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))
        raise SystemExit(2)

    if not result:
        from nvflare.tool.cli_output import is_json_mode, print_human

        if not is_json_mode():
            print_human("No resources specified.")
            return

    output_ok(result)


def cmd_system_shutdown(args):
    from nvflare.fuel.flare_api.api_spec import AuthenticationError, InvalidTarget, NoConnection
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        _system_sub_cmd_parsers.get(CMD_SYSTEM_SHUTDOWN),
        "nvflare system shutdown",
        ["nvflare system shutdown all --force"],
        sys.argv[1:],
    )

    target = args.target
    client_names = getattr(args, "client_names", [])

    _confirm_or_force(f"Really shutdown {target}?", args)

    try:
        with _system_session(args) as sess:
            result = sess.shutdown(target, client_names=client_names or None)
    except (AuthenticationError, NoConnection):
        raise
    except InvalidTarget as e:
        output_error("INVALID_ARGS", exit_code=4, detail=str(e))
        raise SystemExit(4)
    except Exception as e:
        output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))
        raise SystemExit(2)

    output_ok({"target": target, "status": "shutdown initiated", "result": result})


def cmd_system_restart(args):
    from nvflare.fuel.flare_api.api_spec import AuthenticationError, InvalidTarget, NoConnection
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        _system_sub_cmd_parsers.get(CMD_SYSTEM_RESTART),
        "nvflare system restart",
        ["nvflare system restart server --force"],
        sys.argv[1:],
    )

    target = args.target
    client_names = getattr(args, "client_names", [])

    _confirm_or_force(f"Really restart {target}?", args)

    try:
        with _system_session(args) as sess:
            result = sess.restart(target, client_names=client_names or None)
    except (AuthenticationError, NoConnection):
        raise
    except InvalidTarget as e:
        output_error("INVALID_ARGS", exit_code=4, detail=str(e))
        raise SystemExit(4)
    except Exception as e:
        output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))
        raise SystemExit(2)

    output_ok({"target": target, "status": "restart initiated", "result": result})


def cmd_system_version(args):
    from nvflare.fuel.flare_api.api_spec import AuthenticationError, NoConnection
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        _system_sub_cmd_parsers.get(CMD_SYSTEM_VERSION),
        "nvflare system version",
        ["nvflare system version", "nvflare system version --site server"],
        sys.argv[1:],
    )

    site = getattr(args, "site", "all")

    admin_version = getattr(nvflare, "__version__", "unknown")

    target_type = "all" if site == "all" else "server" if site == "server" else "client"

    try:
        with _system_session(args) as sess:
            if target_type == "server":
                known_sites = ["server"]
                targets = None
            else:
                sys_info = sess.get_system_info()
                known_sites = ["server"] + [client.name for client in sys_info.client_info]

                if site != "all" and site not in known_sites:
                    output_error("SITE_NOT_FOUND", site=site)
                    raise SystemExit(1)

                targets = [site] if target_type == "client" else None
            raw_versions = sess.report_version(target_type, targets)
    except (AuthenticationError, NoConnection):
        raise
    except Exception as e:
        output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))
        raise SystemExit(2)

    sites = [site] if site != "all" else known_sites
    versions = []
    for s in sites:
        payload = raw_versions.get(s, {}) if isinstance(raw_versions, dict) else {}
        version = payload.get("version") if isinstance(payload, dict) else None
        versions.append({"site": s, "version": version or "unknown"})

    result = {"sites": versions, "admin_version": admin_version}
    server_version = next((v["version"] for v in versions if v["site"] == "server"), None)
    if server_version is not None:
        result["compatible"] = all(v["version"] == server_version for v in versions)
        result["mismatched_sites"] = [v["site"] for v in versions if v["version"] != server_version]

    _output_system_version(result)


def cmd_system_log(args):
    from nvflare.fuel.flare_api.api_spec import AuthenticationError, NoConnection
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        _system_sub_cmd_parsers.get(CMD_SYSTEM_LOG_CONFIG),
        "nvflare system log-config",
        [
            "nvflare system log-config DEBUG",
            "nvflare system log-config concise",
        ],
        sys.argv[1:],
    )

    level = getattr(args, "level", None)
    site = getattr(args, "site", "all")

    if not level:
        parser = _system_sub_cmd_parsers.get(CMD_SYSTEM_LOG_CONFIG)
        output_usage_error(
            None if getattr(args, "schema", False) else parser,
            "specify a log level or mode",
            exit_code=4,
            error_code="LOG_CONFIG_INVALID",
            message="Log config is not a recognised log mode.",
            hint="Supply one of: DEBUG, INFO, WARNING, ERROR, CRITICAL, concise, msg_only, full, verbose, reload.",
        )
        raise SystemExit(4)

    try:
        with _system_session(args) as sess:
            sess.configure_site_log(level, target=site)
    except (AuthenticationError, NoConnection):
        raise
    except Exception as e:
        output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))
        raise SystemExit(2)

    output_ok({"site": site, "log_config": level, "status": "applied"})


def cmd_system_remove_client(args):
    from nvflare.fuel.flare_api.api_spec import AuthenticationError, InvalidTarget, NoConnection
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        _system_sub_cmd_parsers.get(CMD_SYSTEM_REMOVE_CLIENT),
        "nvflare system remove-client",
        ["nvflare system remove-client site-1 --force"],
        sys.argv[1:],
    )

    client_name = args.client_name
    _confirm_or_force(f"Really remove client '{client_name}'?", args)

    try:
        with _system_session(args) as sess:
            sess.remove_client(client_name)
    except (AuthenticationError, NoConnection):
        raise
    except InvalidTarget as e:
        output_error("INVALID_ARGS", exit_code=4, detail=str(e))
        raise SystemExit(4)
    except Exception as e:
        output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))
        raise SystemExit(2)

    output_ok({"client_name": client_name, "status": "removed"})


_system_handlers = {
    CMD_SYSTEM_STATUS: cmd_system_status,
    CMD_SYSTEM_RESOURCES: cmd_system_resources,
    CMD_SYSTEM_SHUTDOWN: cmd_system_shutdown,
    CMD_SYSTEM_RESTART: cmd_system_restart,
    CMD_SYSTEM_REMOVE_CLIENT: cmd_system_remove_client,
    CMD_SYSTEM_VERSION: cmd_system_version,
    CMD_SYSTEM_LOG_CONFIG: cmd_system_log,
}


def handle_system_cmd(args):
    sub_cmd = getattr(args, "system_sub_cmd", None)
    if sub_cmd is None:
        from nvflare.cli_unknown_cmd_exception import CLIUnknownCmdException

        raise CLIUnknownCmdException("system subcommand required")
    handler = _system_handlers.get(sub_cmd)
    if handler is None:
        from nvflare.cli_unknown_cmd_exception import CLIUnknownCmdException

        raise CLIUnknownCmdException(f"Unknown system subcommand: {sub_cmd}")
    handler(args)
