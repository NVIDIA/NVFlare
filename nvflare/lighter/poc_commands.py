# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
import random
import sys

from nvflare.lighter.poc import generate_poc
from nvflare.lighter.service_constants import FlareServiceConstants as SC

DEFAULT_WORKSPACE = "/tmp/nvflare/poc"


def get_package_command(cmd_type: str, poc_workspace: str, package_dir) -> str:
    cmd_map = {
        SC.CMD_START: {
            SC.FLARE_CONSOLE: get_cmd_path(poc_workspace, package_dir, "fl_admin.sh"),
            "*": get_cmd_path(poc_workspace, package_dir, "start.sh"),
        },
        SC.CMD_STOP: {
            "*": get_stop_cmd(poc_workspace, package_dir),
        },
    }

    default_cmd = cmd_map[cmd_type]["*"]
    return cmd_map[cmd_type].get(package_dir, default_cmd)


def get_stop_cmd(poc_workspace: str, service_dir_name: str):
    service_dir = os.path.join(poc_workspace, service_dir_name)
    stop_file = os.path.join(service_dir, "shutdown.fl")
    return f"touch {stop_file}"


def get_nvflare_home() -> str:
    nvflare_home = os.getenv("NVFLARE_HOME")
    if not nvflare_home:
        print("NVFLARE_HOME environment variable is not set. Please set NVFLARE_HOME=<NVFLARE install dir>")
        sys.exit(1)

    if nvflare_home.endswith("/"):
        nvflare_home = nvflare_home[:-1]
    return nvflare_home


def get_upload_dir(poc_workspace: str) -> str:
    console_config_path = os.path.join(poc_workspace, f"{SC.FLARE_CONSOLE}/{SC.STARTUP}/fed_admin.json")
    try:
        with open(console_config_path, "r") as f:
            console_config = json.load(f)
            upload_dir = console_config[SC.FLARE_CONSOLE]["upload_dir"]
    except IOError as e:
        print(f"failed to load {console_config_path} {e}")
        sys.exit(5)
    except json.decoder.JSONDecodeError as e:
        print(f"failed to load {console_config_path}, please double check the configuration {e}")
        sys.exit(5)

    return upload_dir


def prepare_examples(poc_workspace: str):
    src = os.path.join(get_nvflare_home(), SC.EXAMPLES)
    dst = os.path.join(poc_workspace, f"{SC.FLARE_CONSOLE}/{get_upload_dir(poc_workspace)}")
    print(f"link examples from {src} to {dst}")
    os.symlink(src, dst)


def prepare_poc(number_of_clients: int, poc_workspace: str):
    print(f"prepare_poc at {poc_workspace} for {number_of_clients} clients")
    generate_poc(number_of_clients, poc_workspace)
    prepare_examples(poc_workspace)


def sort_package_cmds(package_cmds: list) -> list:
    def sort_first(val):
        return val[0]

    order_packages = []
    for package_name, cmd_path in package_cmds:
        if package_name == SC.FLARE_SERVER:
            order_packages.append((0, package_name, cmd_path))
        elif package_name == SC.FLARE_CONSOLE:
            order_packages.append((sys.maxsize, package_name, cmd_path))
        else:
            if len(package_cmds) == 1:
                order_packages.append((0, package_name, cmd_path))
            else:
                order_packages.append((random.randint(2, len(package_cmds)), package_name, cmd_path))

    order_packages.sort(key=sort_first)
    return [(package_name, cmd_path) for n, package_name, cmd_path in order_packages]


def get_cmd_path(poc_workspace, service_name, cmd):
    service_dir = os.path.join(poc_workspace, service_name)
    bin_dir = os.path.join(service_dir, SC.STARTUP)
    cmd_path = os.path.join(bin_dir, cmd)
    return cmd_path


def is_poc_ready(poc_workspace: str):
    # check server and admin directories exist
    console_dir = os.path.join(poc_workspace, SC.FLARE_CONSOLE)
    server_dir = os.path.join(poc_workspace, SC.FLARE_SERVER)
    return os.path.isdir(server_dir) and os.path.isdir(console_dir)


def validate_poc_workspace(poc_workspace: str):
    if not is_poc_ready(poc_workspace):
        print(f"workspace {poc_workspace} is not ready, please use poc --prepare to prepare poc workspace")
        sys.exit(2)


def start_poc(poc_workspace: str, white_list: list = []):
    print(f"start_poc at {poc_workspace}, white_list={white_list}")
    validate_poc_workspace(poc_workspace)
    _run_poc(SC.CMD_START, poc_workspace, excluded=[SC.FLARE_OVERSEER], white_list=white_list)


def stop_poc(poc_workspace: str, white_list: list = []):
    print(f"stop_poc at {poc_workspace}")
    validate_poc_workspace(poc_workspace)
    _run_poc(SC.CMD_STOP, poc_workspace, excluded=[SC.FLARE_OVERSEER], white_list=white_list)


def _build_commands(cmd_type: str, poc_workspace: str, excluded: list, white_list: list = []):
    """
    :param cmd_type: start/stop
    :param poc_workspace:  poc workspace directory path
    :param white_list: whitelist, package name. If empty, include every package
    :param excluded: excluded package namae
    :return:
    """

    package_commands = []
    for root, dirs, files in os.walk(poc_workspace):
        if root == poc_workspace:
            for package_dir_name in dirs:
                if package_dir_name not in excluded:
                    if len(white_list) == 0 or package_dir_name in white_list:
                        cmd = get_package_command(cmd_type, poc_workspace, package_dir_name)
                        if cmd:
                            package_commands.append((package_dir_name, cmd))
    return sort_package_cmds(package_commands)


def async_process(cmd_path):
    import subprocess
    import time

    subprocess.Popen(cmd_path.split(" "))
    time.sleep(3)


def sync_process(cmd_path):
    import subprocess

    subprocess.run(cmd_path.split(" "))


def _run_poc(cmd_type: str, poc_workspace: str, excluded: list, white_list=[]):
    package_commands = _build_commands(cmd_type, poc_workspace, excluded, white_list)
    for package_name, cmd_path in package_commands:
        print(f"{cmd_type}: package: {package_name}, executing {cmd_path}")
        if package_name == SC.FLARE_CONSOLE:
            sync_process(cmd_path)
        else:
            async_process(cmd_path)


def clean_poc(poc_workspace: str):
    import shutil

    if is_poc_ready(poc_workspace):
        shutil.rmtree(poc_workspace, ignore_errors=True)
        print(f"{poc_workspace} is removed")
    else:
        print(f"{poc_workspace} is not valid poc directory")
        exit(1)


def def_poc_parser(sub_cmd, prog_name: str):
    poc_parser = sub_cmd.add_parser("poc")
    poc_parser.add_argument(
        "-n", "--number_of_clients", type=int, nargs="?", default=2, help="number of sites or clients, default to 2"
    )
    poc_parser.add_argument(
        "-p",
        "--package",
        type=str,
        nargs="?",
        default="all",
        help="package directory, default to all = all packages, only used for start/stop-poc commands when specified",
    )
    poc_parser.add_argument(
        "--prepare", dest="prepare_poc", action="store_const", const=prepare_poc, help="prepare poc workspace"
    )
    poc_parser.add_argument("--start", dest="start_poc", action="store_const", const=start_poc, help="start poc")
    poc_parser.add_argument("--stop", dest="stop_poc", action="store_const", const=stop_poc, help="stop poc")
    poc_parser.add_argument(
        "--clean", dest="clean_poc", action="store_const", const=clean_poc, help="cleanup poc workspace"
    )


def is_poc(cmd_args) -> bool:
    return (
        hasattr(cmd_args, "start_poc")
        or hasattr(cmd_args, "prepare_poc")
        or hasattr(cmd_args, "stop_poc")
        or hasattr(cmd_args, "clean_poc")
    )


def handle_poc_cmd(cmd_args):
    if cmd_args.package != "all":
        white_list = [cmd_args.package]
    else:
        white_list = []

    poc_workspace = os.getenv("NVFLARE_POC_WORKSPACE")
    if poc_workspace is None or len(poc_workspace.strip()) == 0:
        poc_workspace = DEFAULT_WORKSPACE
    if cmd_args.start_poc:
        start_poc(poc_workspace, white_list)
    elif cmd_args.prepare_poc:
        prepare_poc(cmd_args.number_of_clients, poc_workspace)
    elif cmd_args.stop_poc:
        stop_poc(poc_workspace, white_list)
    elif cmd_args.clean_poc:
        clean_poc(poc_workspace)
    else:
        print(f"unable to handle poc command:{cmd_args}")
        sys.exit(3)
