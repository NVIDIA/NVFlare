# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import os
import sys
import traceback

from pyhocon import ConfigFactory as CF

from nvflare.cli_exception import CLIException
from nvflare.cli_unknown_cmd_exception import CLIUnknownCmdException
from nvflare.dashboard.cli import define_dashboard_parser, handle_dashboard
from nvflare.fuel.flare_api.api_spec import AuthenticationError, NoConnection
from nvflare.fuel.hci.tools.authz_preview import define_authz_preview_parser, run_command
from nvflare.lighter.provision import define_provision_parser, handle_provision
from nvflare.private.fed.app.simulator.simulator import define_simulator_parser, run_simulator
from nvflare.private.fed.app.utils import version_check
from nvflare.tool.cert.cert_cli import def_cert_cli_parser, handle_cert_cmd
from nvflare.tool.job.job_cli import def_job_cli_parser, handle_job_cli_cmd
from nvflare.tool.package.package_cli import def_package_cli_parser, handle_package_cmd
from nvflare.tool.poc.poc_commands import def_poc_parser, handle_poc_cmd
from nvflare.tool.preflight_check import check_packages, define_preflight_check_parser
from nvflare.tool.recipe.recipe_cli import def_recipe_parser, handle_recipe_cmd
from nvflare.tool.system.system_cli import def_system_cli_parser, handle_system_cmd
from nvflare.utils.cli_utils import (
    TARGET_POC,
    TARGET_PROD,
    backup_hidden_config_file,
    create_job_template_config,
    create_poc_workspace_config,
    create_startup_kit_config,
    ensure_hidden_config_migrated,
    load_hidden_config_state,
    print_hidden_config_migration_notice,
    save_config,
)

CMD_POC = "poc"
CMD_PROVISION = "provision"
CMD_PREFLIGHT_CHECK = "preflight_check"
CMD_SIMULATOR = "simulator"
CMD_DASHBOARD = "dashboard"
CMD_AUTHZ_PREVIEW = "authz_preview"
CMD_JOB = "job"
CMD_RECIPE = "recipe"
CMD_CONFIG = "config"
CMD_CERT = "cert"
CMD_PACKAGE = "package"
CMD_SYSTEM = "system"


def def_provision_parser(sub_cmd):
    cmd = CMD_PROVISION
    provision_parser = sub_cmd.add_parser(cmd, help="provision a project")
    define_provision_parser(provision_parser)
    return {cmd: provision_parser}


def def_dashboard_parser(sub_cmd):
    cmd = CMD_DASHBOARD
    dashboard_parser = sub_cmd.add_parser(cmd, help="start the NVFlare dashboard")
    define_dashboard_parser(dashboard_parser)
    return {cmd: dashboard_parser}


def def_preflight_check_parser(sub_cmd):
    cmd = CMD_PREFLIGHT_CHECK
    checker_parser = sub_cmd.add_parser(
        cmd,
        aliases=["preflight"],
        help="check a provisioned package before deployment",
    )
    define_preflight_check_parser(checker_parser)
    return {cmd: checker_parser}


def def_simulator_parser(sub_cmd):
    cmd = CMD_SIMULATOR
    simulator_parser = sub_cmd.add_parser(cmd, help="[deprecated] run a job in local simulator")
    define_simulator_parser(simulator_parser)
    simulator_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    return {cmd: simulator_parser}


def handle_simulator_cmd(simulator_args):
    from nvflare.tool.cli_output import print_human
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        None,
        "nvflare simulator",
        ["nvflare simulator -n 2 -t 2 /path/to/job"],
        sys.argv[1:],
        deprecated=True,
        deprecated_message="Use 'python job.py' with SimEnv instead.",
    )
    print_human("WARNING: 'nvflare simulator' is deprecated. Use 'python job.py' with SimEnv instead.")
    status = run_simulator(simulator_args)
    # make sure the script terminate after run
    if status:
        sys.exit(status)


def def_authz_preview_parser(sub_cmd):
    cmd = CMD_AUTHZ_PREVIEW
    authz_preview_parser = sub_cmd.add_parser(cmd, help="[deprecated] preview authorization policy")
    define_authz_preview_parser(authz_preview_parser)
    authz_preview_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    return {cmd: authz_preview_parser}


def handle_authz_preview(args):
    from nvflare.tool.cli_output import print_human
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        None,
        "nvflare authz_preview",
        ["nvflare authz_preview -p /path/to/policy.json"],
        sys.argv[1:],
        deprecated=True,
        deprecated_message="This command is deprecated and will be removed in a future release.",
    )
    print_human("WARNING: 'nvflare authz_preview' is deprecated and will be removed in a future release.")
    run_command(args)


_config_parser = None


def def_config_parser(sub_cmd):
    global _config_parser
    cmd = "config"
    config_parser = sub_cmd.add_parser(cmd, help="configure local NVFlare settings (startup kit path, POC workspace)")
    _config_parser = config_parser
    config_parser.add_argument(
        "--poc.startup_kit",
        dest="poc_startup_kit_dir",
        type=str,
        nargs="?",
        default=None,
        help="POC startup kit location",
    )
    config_parser.add_argument(
        "--prod.startup_kit",
        dest="prod_startup_kit_dir",
        type=str,
        nargs="?",
        default=None,
        help="production startup kit location",
    )
    config_parser.add_argument(
        "-d",
        "--startup_kit_dir",
        dest="legacy_startup_kit_dir",
        type=str,
        nargs="?",
        default=None,
        help=argparse.SUPPRESS,
    )
    config_parser.add_argument(
        "--poc.workspace",
        dest="poc_workspace_dir",
        type=str,
        nargs="?",
        default=None,
        help="POC workspace location",
    )
    config_parser.add_argument(
        "-jt", "--job_templates_dir", type=str, nargs="?", default=None, help="job templates location"
    )
    config_parser.add_argument("-debug", "--debug", action="store_true", help="debug is on")
    config_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    return {cmd: config_parser}


def handle_config_cmd(args):
    from nvflare.tool.cli_output import output_error, output_ok, print_human
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        _config_parser,
        "nvflare config",
        [
            "nvflare config --poc.startup_kit /path/to/poc_startup",
            "nvflare config --prod.startup_kit /path/to/prod_startup",
            "nvflare config --poc.workspace /path/to/poc_workspace",
        ],
        sys.argv[1:],
    )

    config_file_path, loaded_config, migration_needed = load_hidden_config_state()
    nvflare_config = loaded_config or CF.parse_string("{}")
    requested_poc_startup = args.poc_startup_kit_dir
    requested_prod_startup = args.prod_startup_kit_dir
    requested_poc_workspace = args.poc_workspace_dir
    legacy_startup_kit_dir = getattr(args, "legacy_startup_kit_dir", None)

    if legacy_startup_kit_dir is not None:
        if requested_poc_startup is not None or requested_prod_startup is not None:
            output_error(
                "INVALID_ARGS",
                exit_code=4,
                detail="--startup_kit_dir cannot be used together with --poc.startup_kit or --prod.startup_kit",
            )
            return
        requested_poc_startup = legacy_startup_kit_dir
        print_human(
            "WARNING: 'nvflare config -d/--startup_kit_dir' is deprecated. "
            "Use '--poc.startup_kit' for the same setting."
        )

    if (
        requested_poc_startup is None
        and requested_prod_startup is None
        and requested_poc_workspace is None
        and args.job_templates_dir is None
    ):
        # Read-only: print existing config
        poc_startup_kit_dir = nvflare_config.get("poc.startup_kit", None) if nvflare_config else None
        prod_startup_kit_dir = nvflare_config.get("prod.startup_kit", None) if nvflare_config else None
        startup_kit_dir = poc_startup_kit_dir or prod_startup_kit_dir
        poc_workspace_dir = nvflare_config.get("poc.workspace", None) if nvflare_config else None
        job_templates_dir = nvflare_config.get("job_template.path", None) if nvflare_config else None
        output_ok(
            {
                "config_file": config_file_path,
                "startup_kit_dir": startup_kit_dir,
                "poc_startup_kit_dir": poc_startup_kit_dir,
                "prod_startup_kit_dir": prod_startup_kit_dir,
                "poc_workspace_dir": poc_workspace_dir,
                "job_templates_dir": job_templates_dir,
            }
        )
        return

    try:
        nvflare_config = create_startup_kit_config(nvflare_config, TARGET_POC, requested_poc_startup)
        nvflare_config = create_startup_kit_config(nvflare_config, TARGET_PROD, requested_prod_startup)
        nvflare_config = create_poc_workspace_config(nvflare_config, requested_poc_workspace)
        nvflare_config = create_job_template_config(nvflare_config, args.job_templates_dir)
    except ValueError as e:
        output_error("INVALID_ARGS", exit_code=4, detail=str(e))
        return

    backup_path = backup_hidden_config_file(config_file_path) if migration_needed else None
    save_config(nvflare_config, config_file_path)
    if backup_path:
        print_hidden_config_migration_notice(config_file_path, backup_path)

    poc_startup_kit_dir = nvflare_config.get("poc.startup_kit", None) if nvflare_config else requested_poc_startup
    prod_startup_kit_dir = nvflare_config.get("prod.startup_kit", None) if nvflare_config else requested_prod_startup
    startup_kit_dir = poc_startup_kit_dir or prod_startup_kit_dir
    poc_workspace_dir = nvflare_config.get("poc.workspace", None) if nvflare_config else requested_poc_workspace
    job_templates_dir = nvflare_config.get("job_template.path", None) if nvflare_config else args.job_templates_dir

    output_ok(
        {
            "config_file": config_file_path,
            "startup_kit_dir": startup_kit_dir,
            "poc_startup_kit_dir": poc_startup_kit_dir,
            "prod_startup_kit_dir": prod_startup_kit_dir,
            "poc_workspace_dir": poc_workspace_dir,
            "job_templates_dir": job_templates_dir,
        }
    )


def _get_subcommand_choices(parser):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return sorted(action.choices.keys())
    return []


def _emit_argparse_error_json(parser, message):
    from nvflare.tool.cli_output import SCHEMA_VERSION

    # Parser errors intentionally expose usage/choices inline because they are generated before any
    # command handler runs and therefore sit outside the normal command data envelope.
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "error",
        "exit_code": 4,
        "error_code": "INVALID_ARGS",
        "message": message,
        "hint": "Run with --help to see usage.",
        "data": {
            "usage": parser.format_usage().strip(),
            "choices": _get_subcommand_choices(parser),
        },
    }
    print(json.dumps(payload))
    parser.exit(4)


def _emit_argparse_error_human(parser, message, exit_code: int = 4):
    from nvflare.tool.cli_output import output_usage_error

    output_usage_error(parser, message, exit_code=exit_code)


def _patch_help_on_error(parser, json_mode: bool = False):
    """Recursively patch every parser in the tree to print help before error-exit.

    When argparse detects a missing required argument it calls parser.error(),
    which prints a terse usage line and exits 2.  By wrapping error() we ensure
    the full help is printed first so users see the complete syntax.
    """

    def _error_with_help(message):
        if json_mode:
            _emit_argparse_error_json(parser, message)
        else:
            _emit_argparse_error_human(parser, message, exit_code=4)

    parser.error = _error_with_help
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for sub in action.choices.values():
                _patch_help_on_error(sub, json_mode=json_mode)


def _build_global_arg_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--version", "-V", action="store_true", help="print nvflare version")
    parser.add_argument(
        "--out-format",
        dest="out_format",
        choices=["txt", "json"],
        default="txt",
        help="output format: 'txt' (default, human-readable to stdout/stderr) or 'json' for machine-readable JSON envelope on stdout",
    )
    parser.add_argument(
        "--connect-timeout",
        dest="connect_timeout",
        type=float,
        default=5.0,
        help="seconds to wait for server connection (default: 5.0)",
    )
    return parser


def _normalize_global_args(argv, global_parser):
    """Move supported global options ahead of the subcommand without parsing them.

    This keeps argparse as the single source of truth while preserving the CLI's
    existing behavior of accepting global flags after the subcommand.
    """

    option_actions = global_parser._option_string_actions
    global_args = []
    remaining_args = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        option, has_inline_value = (arg.split("=", 1)[0], True) if arg.startswith("--") and "=" in arg else (arg, False)
        action = option_actions.get(option)
        if action is None:
            remaining_args.append(arg)
            i += 1
            continue

        global_args.append(arg)
        if action.nargs != 0 and not has_inline_value:
            if i + 1 < len(argv):
                global_args.append(argv[i + 1])
                i += 2
            else:
                i += 1
        else:
            i += 1

    return global_args + remaining_args


def parse_args(prog_name: str):
    global_parser = _build_global_arg_parser()
    normalized_argv = _normalize_global_args(sys.argv[1:], global_parser)
    global_args, _ = global_parser.parse_known_args(normalized_argv)
    _parser = argparse.ArgumentParser(description=prog_name, parents=[global_parser])
    sub_cmd = _parser.add_subparsers(title="commands", metavar="", dest="sub_command")
    sub_cmd_parsers = {}
    sub_cmd_parsers.update(def_poc_parser(sub_cmd))
    sub_cmd_parsers.update(def_preflight_check_parser(sub_cmd))
    sub_cmd_parsers.update(def_provision_parser(sub_cmd))
    sub_cmd_parsers.update(def_simulator_parser(sub_cmd))
    sub_cmd_parsers.update(def_dashboard_parser(sub_cmd))
    sub_cmd_parsers.update(def_authz_preview_parser(sub_cmd))
    sub_cmd_parsers.update(def_job_cli_parser(sub_cmd))
    sub_cmd_parsers.update(def_recipe_parser(sub_cmd))
    sub_cmd_parsers.update(def_config_parser(sub_cmd))
    sub_cmd_parsers.update(def_cert_cli_parser(sub_cmd))
    sub_cmd_parsers.update(def_package_cli_parser(sub_cmd))
    system_parser = sub_cmd.add_parser(CMD_SYSTEM, help="FL system operations (status, shutdown, version, ...)")
    sub_cmd_parsers.update({CMD_SYSTEM: system_parser})
    def_system_cli_parser(system_parser)

    # Normalize CLI aliases so the handlers dict can use canonical names.
    _CMD_ALIASES = {"preflight": CMD_PREFLIGHT_CHECK}

    if "--schema" in sys.argv:
        # When --schema is present, bypass argparse entirely to avoid required-arg
        # validation failures. Build just enough namespace to route to the handler;
        # the handler calls handle_schema_flag() as its first line and exits before
        # accessing any command-specific args.
        positionals = [a for a in normalized_argv if not a.startswith("-")]
        ns = argparse.Namespace()
        raw_cmd = positionals[0] if positionals else None
        ns._raw_sub_command = raw_cmd
        ns._argv = list(sys.argv[1:])
        ns.sub_command = _CMD_ALIASES.get(raw_cmd, raw_cmd)
        # Two-level dispatch: each multi-level handler reads a different dest attribute.
        sub_sub = positionals[1] if len(positionals) > 1 else None
        ns.job_sub_cmd = sub_sub
        ns.poc_sub_cmd = sub_sub
        ns.system_sub_cmd = sub_sub
        ns.cert_sub_command = sub_sub
        ns.recipe_sub_cmd = sub_sub
        ns.out_format = global_args.out_format
        ns.connect_timeout = global_args.connect_timeout
        ns.version = global_args.version
        return _parser, ns, sub_cmd_parsers

    # Patch every parser so it prints full help before exiting on error.
    _patch_help_on_error(_parser, json_mode=global_args.out_format == "json")

    args, unknown = _parser.parse_known_args(normalized_argv)
    args._raw_sub_command = args.__dict__.get("sub_command")
    args._argv = list(sys.argv[1:])
    cmd = _CMD_ALIASES.get(args.__dict__.get("sub_command"), args.__dict__.get("sub_command"))
    args.sub_command = cmd
    sub_cmd_parser = sub_cmd_parsers.get(cmd)
    if unknown:
        msg = f"unrecognized arguments: {' '.join(unknown)}"
        if args.out_format == "json":
            _emit_argparse_error_json(sub_cmd_parser or _parser, f"{prog_name} {cmd}: {msg}")
        else:
            _emit_argparse_error_human(sub_cmd_parser or _parser, msg, exit_code=4)
    return _parser, args, sub_cmd_parsers


handlers = {
    CMD_POC: handle_poc_cmd,
    CMD_PROVISION: handle_provision,
    CMD_PREFLIGHT_CHECK: check_packages,
    CMD_SIMULATOR: handle_simulator_cmd,
    CMD_DASHBOARD: handle_dashboard,
    CMD_AUTHZ_PREVIEW: handle_authz_preview,
    CMD_JOB: handle_job_cli_cmd,
    CMD_RECIPE: handle_recipe_cmd,
    CMD_CONFIG: handle_config_cmd,
    CMD_CERT: handle_cert_cmd,
    CMD_PACKAGE: handle_package_cmd,
    CMD_SYSTEM: handle_system_cmd,
}


def _auth_hint_from_detail(detail: str, auth_code: str = None) -> str:
    if auth_code == "AUTH_UNKNOWN_STUDY" or auth_code == "AUTH_STUDY_NOT_CONFIGURED":
        return "Add the study under 'studies:' in project.yml with api_version: 4, reprovision, redeploy or restart the server, then try again."
    if auth_code == "AUTH_STUDY_USER_NOT_MAPPED":
        return "Add this user under the study's admins mapping in project.yml, reprovision, redeploy or restart the server, then try again."
    if auth_code in {"AUTH_INVALID_STUDY_NAME", "AUTH_INVALID_STUDY"}:
        return "Use a valid study name in project.yml, reprovision, redeploy or restart the server, then try again."

    detail = (detail or "").lower()
    if "unknown study" in detail or "not configured on the server" in detail:
        return "Add the study under 'studies:' in project.yml with api_version: 4, reprovision, redeploy or restart the server, then try again."
    if "not mapped to study" in detail:
        return "Add this user under the study's admins mapping in project.yml, reprovision, redeploy or restart the server, then try again."
    if "invalid study name" in detail:
        return "Use a valid study name in project.yml, reprovision, redeploy or restart the server, then try again."
    if "certificate validation failed" in detail or "error_cert" in detail:
        return "Check that the startup kit certificate, key, and root CA match the server trust chain."
    return "Check startup kit credentials."


def run(prog_name):
    cwd = os.getcwd()
    sys.path.append(cwd)
    prog_parser, prog_args, sub_cmd_parsers = parse_args(prog_name)

    from nvflare.tool.cli_output import set_connect_timeout, set_output_format

    set_output_format(getattr(prog_args, "out_format", "txt"))
    set_connect_timeout(getattr(prog_args, "connect_timeout", 5.0))
    _suppress_cli_connector_noise()

    sub_cmd = None
    try:
        sub_cmd = prog_args.sub_command
        if sub_cmd:
            handlers[sub_cmd](prog_args)
        elif prog_args.version:
            print_nvflare_version()
        else:
            prog_parser.print_help()
    except CLIUnknownCmdException as e:
        from nvflare.tool.cli_output import output_usage_error

        parser = sub_cmd_parsers.get(sub_cmd) if sub_cmd else prog_parser
        output_usage_error(parser, str(e).strip(), exit_code=4)
    except CLIException as e:
        from nvflare.tool.cli_output import output_error

        output_error("CLI_ERROR", exit_code=1, detail=str(e))
    except NoConnection:
        from nvflare.tool.cli_output import output_error

        output_error("CONNECTION_FAILED", exit_code=2)
    except AuthenticationError as e:
        from nvflare.tool.cli_output import output_error_message

        output_error_message(
            "AUTH_FAILED",
            message="Authentication failed.",
            hint=_auth_hint_from_detail(str(e), getattr(e, "auth_code", None)),
            exit_code=2,
            detail=str(e),
        )
    except TimeoutError:
        from nvflare.tool.cli_output import output_error

        output_error("TIMEOUT", exit_code=3)
    except SystemExit:
        raise
    except Exception as e:
        from nvflare.tool.cli_output import output_error

        if hasattr(prog_args, "debug") and prog_args.debug:
            print(traceback.format_exc(), file=sys.stderr)
        output_error("INTERNAL_ERROR", exit_code=5, detail=str(e))


def _suppress_cli_connector_noise():
    """Reduce noisy connector retry logs for CLI invocations only."""
    import logging

    noisy_loggers = [
        "nvflare.fuel.f3.sfm.conn_manager",
        "nvflare.fuel.f3.cellnet",
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.CRITICAL)


def print_nvflare_version():
    import nvflare

    print(f"NVFlare version is {nvflare.__version__}")


def main():
    version_check()
    ensure_hidden_config_migrated()
    run("nvflare")


if __name__ == "__main__":
    main()
