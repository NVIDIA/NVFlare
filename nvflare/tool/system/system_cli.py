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
import os
import sys
import time
from contextlib import contextmanager

from nvflare.tool.cli_output import output_error, output_ok, output_usage_error

CMD_SYSTEM_STATUS = "status"
CMD_SYSTEM_RESOURCES = "resources"
CMD_SYSTEM_SHUTDOWN = "shutdown"
CMD_SYSTEM_RESTART = "restart"
CMD_SYSTEM_VERSION = "version"
CMD_SYSTEM_LOG = "log"

_system_sub_cmd_parsers = {}


def def_system_cli_parser(system_parser):
    """system_parser is already created in cli.py — add subcommands here."""
    sub = system_parser.add_subparsers(title="system subcommands", metavar="", dest="system_sub_cmd")

    # status
    p = sub.add_parser(CMD_SYSTEM_STATUS, help="show server and client status")
    p.add_argument("target", nargs="?", choices=["server", "client"], default=None)
    p.add_argument("client_names", nargs="*", default=[])
    p.add_argument("--schema", action="store_true")
    _system_sub_cmd_parsers[CMD_SYSTEM_STATUS] = p

    # resources
    p = sub.add_parser(CMD_SYSTEM_RESOURCES, help="show server and client resource usage")
    p.add_argument("target", nargs="?", choices=["server", "client"], default=None)
    p.add_argument("client_names", nargs="*", default=[])
    p.add_argument("--schema", action="store_true")
    _system_sub_cmd_parsers[CMD_SYSTEM_RESOURCES] = p

    # shutdown
    p = sub.add_parser(CMD_SYSTEM_SHUTDOWN, help="shut down server, client(s), or all")
    p.add_argument("target", choices=["server", "client", "all"])
    p.add_argument("client_names", nargs="*", default=[])
    p.add_argument("--force", action="store_true")
    p.add_argument("--schema", action="store_true")
    _system_sub_cmd_parsers[CMD_SYSTEM_SHUTDOWN] = p

    # restart
    p = sub.add_parser(CMD_SYSTEM_RESTART, help="restart server, client(s), or all")
    p.add_argument("target", choices=["server", "client", "all"])
    p.add_argument("client_names", nargs="*", default=[])
    p.add_argument("--force", action="store_true")
    p.add_argument("--schema", action="store_true")
    _system_sub_cmd_parsers[CMD_SYSTEM_RESTART] = p

    # version
    p = sub.add_parser(CMD_SYSTEM_VERSION, help="show NVFlare version on each remote site")
    p.add_argument("--site", default="all", help="server, a client name, or all")
    p.add_argument("--schema", action="store_true")
    _system_sub_cmd_parsers[CMD_SYSTEM_VERSION] = p

    # log
    p = sub.add_parser(CMD_SYSTEM_LOG, help="change logging level on server or client sites")
    p.add_argument(
        "level",
        nargs="?",
        default=None,
        help="DEBUG, INFO, WARNING, ERROR, CRITICAL, concise, full, verbose, reload",
    )
    p.add_argument("--config", default=None, help="path to dictConfig JSON file or inline JSON")
    p.add_argument("--site", default="all", help="server, a client name, or all")
    p.add_argument("--schema", action="store_true")
    _system_sub_cmd_parsers[CMD_SYSTEM_LOG] = p

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
    from nvflare.tool.cli_output import prompt_yn

    if not prompt_yn(prompt):
        raise SystemExit(0)


def resolve_log_config(level_str, config_str):
    """Resolve log configuration from level string or config string/file.

    Returns a str level or dict dictConfig, or None on parse failure.
    """
    if config_str:
        if os.path.isfile(config_str):
            with open(config_str) as f:
                return json.load(f)
        try:
            return json.loads(config_str)
        except json.JSONDecodeError:
            return None
    return level_str


def _get_system_session():
    """Create a secure session using the startup kit."""
    from nvflare.tool.cli_session import new_cli_session
    from nvflare.utils.cli_utils import get_hidden_config, get_startup_kit_dir_for_target

    try:
        from nvflare.tool.job.job_cli import find_admin_user_and_dir

        username, startup = find_admin_user_and_dir()
    except Exception:
        _, nvflare_config = get_hidden_config()
        startup = nvflare_config.get("poc.startup_kit", None) if nvflare_config else None
        username = None

    if not startup:
        startup = get_startup_kit_dir_for_target(target="poc")

    from nvflare.tool.cli_output import get_connect_timeout

    timeout = get_connect_timeout()
    return new_cli_session(username=username, startup_kit_location=startup, timeout=timeout)


@contextmanager
def _system_session():
    sess = _get_system_session()
    try:
        yield sess
    finally:
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
        print_human("---------------------")
        print_human("| JOB_ID | APP NAME |")
        print_human("---------------------")
        for job in jobs:
            if isinstance(job, dict):
                print_human(f"| {job.get('job_id', '?')} | {job.get('app_name', '?')} |")
        print_human("---------------------")

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
            print_human(f"{'CLIENT':<16} {'FQCN':<20} {'LAST CONNECT TIME':<19} STATUS")
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

            for name in names:
                info = client_info_by_name.get(name, {})
                status = client_status_by_name.get(name, {}).get("status", "unknown")
                fqcn = info.get("fqcn", "-")
                last_seen = _fmt_ts(info.get("client_last_conn_time")) if info else "-"
                print_human(f"{name:<16} {fqcn:<20} {last_seen:<19} {status}")


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
    mismatched_sites = result.get("mismatched_sites") or []
    admin_version = result.get("admin_version", "unknown")

    print_human("Versions")
    print_human(f"  {'SITE':<16} VERSION")
    for entry in sites:
        if isinstance(entry, dict):
            print_human(f"  {entry.get('site', '?'):<16} {entry.get('version', 'unknown')}")

    print_human("")
    print_human(f"Admin version: {admin_version}")
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
        with _system_session() as sess:
            result = sess.check_status(target_type, client_names if client_names else None)
    except Exception as e:
        output_error(
            "CONNECTION_FAILED",
            message="Could not connect to the FLARE server.",
            hint="Start the server or verify the admin startup kit endpoint.",
            exit_code=2,
            detail=str(e),
        )

    _output_system_status(result, target_type)


def cmd_system_resources(args):
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
        with _system_session() as sess:
            result = sess.report_resources(target_type, client_names if client_names else None)
    except Exception as e:
        output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))

    if not result:
        from nvflare.tool.cli_output import is_json_mode, print_human

        if not is_json_mode():
            print_human("No resources specified.")

    output_ok(result)


def cmd_system_shutdown(args):
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        _system_sub_cmd_parsers.get(CMD_SYSTEM_SHUTDOWN),
        "nvflare system shutdown",
        ["nvflare system shutdown all --force", "nvflare system shutdown server --force"],
        sys.argv[1:],
    )

    target = getattr(args, "target", "all")
    client_names = getattr(args, "client_names", [])

    _confirm_or_force(f"Really shutdown {target}?", args)

    try:
        with _system_session() as sess:
            sess.shutdown(target, client_names if client_names else None)
    except Exception as e:
        output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))

    output_ok({"target": target, "status": "shutdown initiated"})


def cmd_system_restart(args):
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        _system_sub_cmd_parsers.get(CMD_SYSTEM_RESTART),
        "nvflare system restart",
        ["nvflare system restart all --force", "nvflare system restart client --force"],
        sys.argv[1:],
    )

    target = getattr(args, "target", "all")
    client_names = getattr(args, "client_names", [])

    _confirm_or_force(f"Really restart {target}?", args)

    try:
        with _system_session() as sess:
            result = sess.restart(target, client_names if client_names else None)
    except Exception as e:
        output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))

    output_ok({"target": target, "status": "restart initiated", "result": result})


def cmd_system_version(args):
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        _system_sub_cmd_parsers.get(CMD_SYSTEM_VERSION),
        "nvflare system version",
        ["nvflare system version", "nvflare system version --site server"],
        sys.argv[1:],
    )

    site = getattr(args, "site", "all")

    try:
        import nvflare

        admin_version = nvflare.__version__
    except Exception:
        admin_version = "unknown"

    target_type = "all" if site == "all" else "server" if site == "server" else "client"

    try:
        with _system_session() as sess:
            sys_info = sess.get_system_info()
            known_sites = ["server"] + [client.name for client in sys_info.client_info]

            if site != "all":
                if site not in known_sites:
                    output_error("SITE_NOT_FOUND", detail=f"site '{site}' not found")

            targets = [site] if target_type == "client" else None
            raw_versions = sess.report_version(target_type, targets)
    except Exception as e:
        output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))

    sites = [site] if site != "all" else known_sites
    versions = []
    for s in sites:
        payload = raw_versions.get(s, {}) if isinstance(raw_versions, dict) else {}
        version = payload.get("version") if isinstance(payload, dict) else None
        versions.append({"site": s, "version": version or "unknown"})

    server_ver = next((v["version"] for v in versions if v["site"] == "server"), admin_version)
    compatible = all(v["version"] == server_ver for v in versions)
    mismatched_sites = [v["site"] for v in versions if v["version"] != server_ver]

    _output_system_version(
        {
            "sites": versions,
            "compatible": compatible,
            "mismatched_sites": mismatched_sites,
            "admin_version": admin_version,
        }
    )


def cmd_system_log(args):
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        _system_sub_cmd_parsers.get(CMD_SYSTEM_LOG),
        "nvflare system log",
        [
            "nvflare system log DEBUG",
            "nvflare system log --config /path/to/logging.json",
        ],
        sys.argv[1:],
    )

    level = getattr(args, "level", None)
    config_str = getattr(args, "config", None)
    site = getattr(args, "site", "all")

    if level and config_str:
        output_error("INVALID_ARGS", exit_code=4, detail="level and --config are mutually exclusive")

    if not level and not config_str:
        parser = _system_sub_cmd_parsers.get(CMD_SYSTEM_LOG)
        output_usage_error(
            None if getattr(args, "schema", False) else parser, "specify a log level or --config JSON/file", exit_code=4
        )

    log_config = resolve_log_config(level, config_str)
    if log_config is None:
        parser = _system_sub_cmd_parsers.get(CMD_SYSTEM_LOG)
        output_usage_error(
            None if getattr(args, "schema", False) else parser,
            "provide a valid level name or --config JSON/file",
            exit_code=1,
            error_code="LOG_CONFIG_INVALID",
            message="Log config is not valid JSON or a recognised log mode.",
            hint="Supply a valid dictConfig JSON file or one of: DEBUG, INFO, WARNING, ERROR, CRITICAL, concise, full, verbose, reload.",
        )

    try:
        with _system_session() as sess:
            sess.configure_site_log(log_config, target=site)
    except Exception as e:
        output_error("CONNECTION_FAILED", exit_code=2, detail=str(e))

    output_ok({"site": site, "log_config": log_config, "status": "applied"})


_system_handlers = {
    CMD_SYSTEM_STATUS: cmd_system_status,
    CMD_SYSTEM_RESOURCES: cmd_system_resources,
    CMD_SYSTEM_SHUTDOWN: cmd_system_shutdown,
    CMD_SYSTEM_RESTART: cmd_system_restart,
    CMD_SYSTEM_VERSION: cmd_system_version,
    CMD_SYSTEM_LOG: cmd_system_log,
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
