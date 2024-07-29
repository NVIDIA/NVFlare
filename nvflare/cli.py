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
from nvflare.fuel.hci.tools.authz_preview import define_authz_preview_parser, run_command
from nvflare.lighter.provision import define_provision_parser, handle_provision
from nvflare.private.fed.app.simulator.simulator import define_simulator_parser, run_simulator
from nvflare.tool.job.job_cli import def_job_cli_parser, handle_job_cli_cmd
from nvflare.tool.poc.poc_commands import def_poc_parser, handle_poc_cmd
from nvflare.tool.preflight_check import check_packages, define_preflight_check_parser
from nvflare.utils.cli_utils import (
    create_job_template_config,
    create_poc_workspace_config,
    create_startup_kit_config,
    get_hidden_config,
    print_hidden_config,
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


def check_python_version():
    if sys.version_info >= (3, 11):
        raise RuntimeError("Python versions 3.11 and above are not yet supported. Please use Python 3.8, 3.9 or 3.10.")
    if sys.version_info < (3, 8):
        raise RuntimeError("Python versions 3.7 and below are not supported. Please use Python 3.8, 3.9 or 3.10")


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
    run_command(args)


def def_config_parser(sub_cmd):
    cmd = "config"
    config_parser = sub_cmd.add_parser(cmd)
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
    return {cmd: config_parser}


def handle_config_cmd(args):
    config_file_path, nvflare_config = get_hidden_config()

    if args.startup_kit_dir is None and args.poc_workspace_dir is None and args.job_templates_dir is None:
        print(f"not specifying any directory. print existing config at {config_file_path}")
        print_hidden_config(config_file_path, nvflare_config)
        return

    nvflare_config = create_startup_kit_config(nvflare_config, args.startup_kit_dir)
    nvflare_config = create_poc_workspace_config(nvflare_config, args.poc_workspace_dir)
    nvflare_config = create_job_template_config(nvflare_config, args.job_templates_dir)

    save_config(nvflare_config, config_file_path)
    print(f"new config at {config_file_path}")
    print_hidden_config(config_file_path, nvflare_config)


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
    except Exception as e:
        print(f"\nUnable to handle command: {sub_cmd} due to: {e} \n")
        if hasattr(prog_args, "debug"):
            if prog_args.debug:
                print(traceback.format_exc())
        else:
            print_help(prog_parser, sub_cmd, sub_cmd_parsers)


def print_help(prog_parser, sub_cmd, sub_cmd_parsers):
    if sub_cmd:
        sub_parser = sub_cmd_parsers[sub_cmd]
        if sub_parser:
            print(f"sub_parser is: {sub_parser}")
            sub_parser.print_help()
        else:
            prog_parser.print_help()
    else:
        prog_parser.print_help()


def print_nvflare_version():
    import nvflare

    print(f"NVFlare version is {nvflare.__version__}")


def main():
    check_python_version()
    run("nvflare")


if __name__ == "__main__":
    main()
