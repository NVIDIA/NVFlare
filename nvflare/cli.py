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
import os
import sys
import traceback

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
from nvflare.utils.cli_utils import (
    create_job_template_config,
    create_poc_workspace_config,
    create_startup_kit_config,
    get_hidden_config,
    save_config,
)

CMD_POC = "poc"
CMD_PROVISION = "provision"
CMD_PREFLIGHT_CHECK = "preflight_check"
CMD_SIMULATOR = "simulator"
CMD_DASHBOARD = "dashboard"
CMD_AUTHZ_PREVIEW = "authz_preview"
CMD_JOB = "job"
CMD_CONFIG = "config"
CMD_CERT = "cert"
CMD_PACKAGE = "package"


def def_provision_parser(sub_cmd):
    cmd = CMD_PROVISION
    provision_parser = sub_cmd.add_parser(cmd)
    define_provision_parser(provision_parser)
    return {cmd: provision_parser}


def def_dashboard_parser(sub_cmd):
    cmd = CMD_DASHBOARD
    dashboard_parser = sub_cmd.add_parser(cmd)
    define_dashboard_parser(dashboard_parser)
    return {cmd: dashboard_parser}


def def_preflight_check_parser(sub_cmd):
    cmd = CMD_PREFLIGHT_CHECK
    checker_parser = sub_cmd.add_parser(cmd)
    define_preflight_check_parser(checker_parser)
    return {cmd: checker_parser}


def def_simulator_parser(sub_cmd):
    cmd = CMD_SIMULATOR
    simulator_parser = sub_cmd.add_parser(cmd)
    define_simulator_parser(simulator_parser)
    return {cmd: simulator_parser}


def handle_simulator_cmd(simulator_args):
    print("WARNING: 'nvflare simulator' is deprecated. Use 'python job.py' with SimEnv instead.", file=sys.stderr)
    status = run_simulator(simulator_args)
    # make sure the script terminate after run
    if status:
        sys.exit(status)


def def_authz_preview_parser(sub_cmd):
    cmd = CMD_AUTHZ_PREVIEW
    authz_preview_parser = sub_cmd.add_parser(cmd)
    define_authz_preview_parser(authz_preview_parser)
    return {cmd: authz_preview_parser}


def handle_authz_preview(args):
    print("WARNING: 'nvflare authz_preview' is deprecated and will be removed in a future release.", file=sys.stderr)
    run_command(args)


_config_parser = None


def def_config_parser(sub_cmd):
    global _config_parser
    cmd = "config"
    config_parser = sub_cmd.add_parser(cmd)
    _config_parser = config_parser
    config_parser.add_argument(
        "-d", "--startup_kit_dir", type=str, nargs="?", default=None, help="startup kit location"
    )
    config_parser.add_argument(
        "-pw", "--poc_workspace_dir", type=str, nargs="?", default=None, help="POC workspace location"
    )
    config_parser.add_argument(
        "-jt", "--job_templates_dir", type=str, nargs="?", default=None, help="job templates location"
    )
    config_parser.add_argument("-debug", "--debug", action="store_true", help="debug is on")
    config_parser.add_argument("--output", choices=["json", "txt"], default="json", help="output format")
    config_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    return {cmd: config_parser}


def handle_config_cmd(args):
    from nvflare.tool.cli_output import output_ok
    from nvflare.tool.cli_schema import handle_schema_flag

    fmt = getattr(args, "output", "json")
    handle_schema_flag(
        _config_parser,
        "nvflare config",
        [
            "nvflare config -d /path/to/startup",
            "nvflare config -pw /path/to/poc --output json",
        ],
        sys.argv[1:],
    )

    config_file_path, nvflare_config = get_hidden_config()

    if args.startup_kit_dir is None and args.poc_workspace_dir is None and args.job_templates_dir is None:
        # Read-only: print existing config
        startup_kit_dir = nvflare_config.get("startup_kit.path", None) if nvflare_config else None
        poc_workspace_dir = nvflare_config.get("poc_workspace.path", None) if nvflare_config else None
        job_templates_dir = nvflare_config.get("job_template.path", None) if nvflare_config else None
        output_ok(
            {
                "config_file": config_file_path,
                "startup_kit_dir": startup_kit_dir,
                "poc_workspace_dir": poc_workspace_dir,
                "job_templates_dir": job_templates_dir,
            },
            fmt,
        )
        return

    nvflare_config = create_startup_kit_config(nvflare_config, args.startup_kit_dir)
    nvflare_config = create_poc_workspace_config(nvflare_config, args.poc_workspace_dir)
    nvflare_config = create_job_template_config(nvflare_config, args.job_templates_dir)

    save_config(nvflare_config, config_file_path)

    startup_kit_dir = nvflare_config.get("startup_kit.path", None) if nvflare_config else args.startup_kit_dir
    poc_workspace_dir = nvflare_config.get("poc_workspace.path", None) if nvflare_config else args.poc_workspace_dir
    job_templates_dir = nvflare_config.get("job_template.path", None) if nvflare_config else args.job_templates_dir

    output_ok(
        {
            "config_file": config_file_path,
            "startup_kit_dir": startup_kit_dir,
            "poc_workspace_dir": poc_workspace_dir,
            "job_templates_dir": job_templates_dir,
        },
        fmt,
    )


def parse_args(prog_name: str):
    _parser = argparse.ArgumentParser(description=prog_name)
    _parser.add_argument("--version", "-V", action="store_true", help="print nvflare version")
    sub_cmd = _parser.add_subparsers(description="sub command parser", dest="sub_command")
    sub_cmd_parsers = {}
    sub_cmd_parsers.update(def_poc_parser(sub_cmd))
    sub_cmd_parsers.update(def_preflight_check_parser(sub_cmd))
    sub_cmd_parsers.update(def_provision_parser(sub_cmd))
    sub_cmd_parsers.update(def_simulator_parser(sub_cmd))
    sub_cmd_parsers.update(def_dashboard_parser(sub_cmd))
    sub_cmd_parsers.update(def_authz_preview_parser(sub_cmd))
    sub_cmd_parsers.update(def_job_cli_parser(sub_cmd))
    sub_cmd_parsers.update(def_config_parser(sub_cmd))
    sub_cmd_parsers.update(def_cert_cli_parser(sub_cmd))
    sub_cmd_parsers.update(def_package_cli_parser(sub_cmd))

    args, argv = _parser.parse_known_args(None, None)
    cmd = args.__dict__.get("sub_command")
    sub_cmd_parser = sub_cmd_parsers.get(cmd)
    if argv:
        msg = f"{prog_name} {cmd}: unrecognized arguments: {' '.join(argv)}\n"
        print(f"\nerror: {msg}")
        if sub_cmd_parser:
            sub_cmd_parser.print_help()
        _parser.exit(2, "\n")
    return _parser, _parser.parse_args(), sub_cmd_parsers


handlers = {
    CMD_POC: handle_poc_cmd,
    CMD_PROVISION: handle_provision,
    CMD_PREFLIGHT_CHECK: check_packages,
    CMD_SIMULATOR: handle_simulator_cmd,
    CMD_DASHBOARD: handle_dashboard,
    CMD_AUTHZ_PREVIEW: handle_authz_preview,
    CMD_JOB: handle_job_cli_cmd,
    CMD_CONFIG: handle_config_cmd,
    CMD_CERT: handle_cert_cmd,
    CMD_PACKAGE: handle_package_cmd,
}


def run(prog_name):
    cwd = os.getcwd()
    sys.path.append(cwd)
    prog_parser, prog_args, sub_cmd_parsers = parse_args(prog_name)
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
        print(e)
        print_help(prog_parser, sub_cmd, sub_cmd_parsers)
        sys.exit(1)
    except CLIException as e:
        print(e)
        sys.exit(1)
    except NoConnection:
        from nvflare.tool.cli_output import output_error

        output_error("CONNECTION_FAILED", getattr(prog_args, "output", "json"), exit_code=2)
    except AuthenticationError:
        from nvflare.tool.cli_output import output_error

        output_error("AUTH_FAILED", getattr(prog_args, "output", "json"), exit_code=2)
    except TimeoutError:
        from nvflare.tool.cli_output import output_error

        output_error("TIMEOUT", getattr(prog_args, "output", "json"), exit_code=3)
    except SystemExit:
        raise
    except Exception as e:
        from nvflare.tool.cli_output import output_error

        if hasattr(prog_args, "debug") and prog_args.debug:
            print(traceback.format_exc())
        output_error("INTERNAL_ERROR", getattr(prog_args, "output", "json"), exit_code=5, detail=str(e))


def print_help(prog_parser, sub_cmd, sub_cmd_parsers):
    if sub_cmd:
        sub_parser = sub_cmd_parsers[sub_cmd]
        if sub_parser:
            print(f"Usage for subcommand '{sub_cmd}':\n")
            sub_parser.print_help()
        else:
            prog_parser.print_help()
    else:
        prog_parser.print_help()


def print_nvflare_version():
    import nvflare

    print(f"NVFlare version is {nvflare.__version__}")


def main():
    version_check()
    run("nvflare")


if __name__ == "__main__":
    main()
