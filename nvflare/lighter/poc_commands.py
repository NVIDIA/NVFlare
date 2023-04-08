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

import json
import os
import random
import subprocess
import sys
import time
from typing import Dict, List, Optional

from nvflare.cli_exception import CLIException
from nvflare.fuel.utils.gpu_utils import get_host_gpu_ids
from nvflare.lighter.poc import generate_poc
from nvflare.lighter.service_constants import FlareServiceConstants as SC
from nvflare.lighter.utils import update_storage_locations
from nvflare.tool.api_utils import shutdown_system

DEFAULT_WORKSPACE = "/tmp/nvflare/poc"


def client_gpu_assignments(clients: List[str], gpu_ids: List[int]) -> Dict[str, List[int]]:
    n_gpus = len(gpu_ids)
    n_clients = len(clients)
    gpu_assignments = {}
    if n_gpus == 0:
        for client in clients:
            gpu_assignments[client] = []

    if 0 < n_gpus <= n_clients:
        for client_id, client in enumerate(clients):
            gpu_index = client_id % n_gpus
            gpu_assignments[client] = [gpu_ids[gpu_index]]
    elif n_gpus > n_clients > 0:
        client_name_map = {}
        for client_id, client in enumerate(clients):
            client_name_map[client_id] = client

        for gpu_index, gpu_id in enumerate(gpu_ids):
            client_id = gpu_index % n_clients
            client = client_name_map[client_id]
            if client not in gpu_assignments:
                gpu_assignments[client] = []
            gpu_assignments[client].append(gpu_ids[gpu_index])
    return gpu_assignments


def get_package_command(cmd_type: str, poc_workspace: str, package_dir) -> str:
    if cmd_type == SC.CMD_START:
        if package_dir == SC.FLARE_CONSOLE:
            cmd = get_cmd_path(poc_workspace, package_dir, "fl_admin.sh")
        elif package_dir == SC.FLARE_SERVER:
            cmd = get_cmd_path(poc_workspace, package_dir, "start.sh")
        else:
            cmd = get_cmd_path(poc_workspace, package_dir, "start.sh")

    elif cmd_type == SC.CMD_STOP:
        cmd = get_stop_cmd(poc_workspace, package_dir)
    else:
        raise ValueError("unknown cmd_type :", cmd_type)
    return cmd


def get_stop_cmd(poc_workspace: str, service_dir_name: str):
    service_dir = os.path.join(poc_workspace, service_dir_name)
    stop_file = os.path.join(service_dir, "shutdown.fl")
    return f"touch {stop_file}"


def get_nvflare_home() -> str:
    nvflare_home = os.getenv("NVFLARE_HOME")
    if nvflare_home:
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
        raise CLIException(f"failed to load {console_config_path} {e}")
    except json.decoder.JSONDecodeError as e:
        raise CLIException(f"failed to load {console_config_path}, please double check the configuration {e}")

    return upload_dir


def prepare_examples(poc_workspace: str):
    nvflare_home = get_nvflare_home()
    if nvflare_home:
        src = os.path.join(nvflare_home, SC.EXAMPLES)
        dst = os.path.join(poc_workspace, f"{SC.FLARE_CONSOLE}/{get_upload_dir(poc_workspace)}")
        print(f"link examples from {src} to {dst}")
        os.symlink(src, dst)


def prepare_poc(number_of_clients: int, poc_workspace: str):
    print(f"prepare_poc at {poc_workspace} for {number_of_clients} clients")
    ret_code = generate_poc(number_of_clients, poc_workspace)
    if poc_workspace != DEFAULT_WORKSPACE:
        update_storage_locations(
            local_dir=f"{poc_workspace}/server/local", default_resource_name="resources.json", workspace=poc_workspace
        )
    if ret_code:
        prepare_examples(poc_workspace)


def sort_package_cmds(cmd_type, package_cmds: list) -> list:
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
    if cmd_type == SC.CMD_STOP:
        order_packages.reverse()
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
        raise CLIException(f"workspace {poc_workspace} is not ready, please use poc --prepare to prepare poc workspace")


def validate_gpu_ids(gpu_ids: list, host_gpu_ids: list):
    for gpu_id in gpu_ids:
        if gpu_id not in host_gpu_ids:
            raise CLIException(
                f"gpu_id provided is not available in the host machine, available GPUs are {host_gpu_ids}"
            )


def get_gpu_ids(user_input_gpu_ids, host_gpu_ids) -> List[int]:
    if type(user_input_gpu_ids) == int and user_input_gpu_ids == -1:
        gpu_ids = host_gpu_ids
    else:
        gpu_ids = user_input_gpu_ids
        validate_gpu_ids(gpu_ids, host_gpu_ids)
    return gpu_ids


def start_poc(poc_workspace: str, gpu_ids: List[int], excluded=None, white_list=None):
    if white_list is None:
        white_list = []
    if excluded is None:
        excluded = []
    print(f"start_poc at {poc_workspace}, gpu_ids={gpu_ids}, excluded = {excluded}, white_list={white_list}")
    validate_poc_workspace(poc_workspace)
    _run_poc(SC.CMD_START, poc_workspace, gpu_ids, excluded=excluded, white_list=white_list)


def stop_poc(poc_workspace: str, excluded=None, white_list=None):
    if white_list is None:
        white_list = []
    if excluded is None:
        excluded = [SC.FLARE_CONSOLE]
    else:
        excluded.append(SC.FLARE_CONSOLE)

    print("start shutdown NVFLARE")
    validate_poc_workspace(poc_workspace)
    gpu_ids: List[int] = []
    shutdown_system(poc_workspace)
    _run_poc(SC.CMD_STOP, poc_workspace, gpu_ids, excluded=excluded, white_list=white_list)


def _get_clients(package_commands: list) -> List[str]:
    clients = [
        package_dir_name
        for package_dir_name, _ in package_commands
        if package_dir_name != SC.FLARE_CONSOLE and package_dir_name != SC.FLARE_SERVER
    ]
    return clients


def _build_commands(cmd_type: str, poc_workspace: str, excluded: list, white_list=None) -> list:
    """
    :param cmd_type: start/stop
    :param poc_workspace:  poc workspace directory path
    :param excluded: excluded package namae
    :param white_list: whitelist, package name. If empty, include every package
    :return:
    """

    def is_fl_package_dir(p_dir_name: str) -> bool:
        return p_dir_name == "admin" or p_dir_name == "server" or p_dir_name.startswith("site-")

    if white_list is None:
        white_list = []
    package_commands = []
    for root, dirs, files in os.walk(poc_workspace):
        if root == poc_workspace:
            fl_dirs = [d for d in dirs if is_fl_package_dir(d)]
            for package_dir_name in fl_dirs:
                if package_dir_name not in excluded:
                    if len(white_list) == 0 or package_dir_name in white_list:
                        cmd = get_package_command(cmd_type, poc_workspace, package_dir_name)
                        if cmd:
                            package_commands.append((package_dir_name, cmd))
    return sort_package_cmds(cmd_type, package_commands)


def prepare_env(gpu_ids: Optional[List[int]] = None):
    import os

    if gpu_ids:
        my_env = os.environ.copy()
        if gpu_ids and len(gpu_ids) > 0:
            my_env["CUDA_VISIBLE_DEVICES"] = ",".join([str(gid) for gid in gpu_ids])
            return my_env

    return None


def async_process(cmd_path, gpu_ids: Optional[List[int]] = None):
    my_env = prepare_env(gpu_ids)
    if my_env:
        subprocess.Popen(cmd_path.split(" "), env=my_env)
    else:
        subprocess.Popen(cmd_path.split(" "))

    time.sleep(3)


def sync_process(cmd_path):
    subprocess.run(cmd_path.split(" "))


def _run_poc(cmd_type: str, poc_workspace: str, gpu_ids: List[int], excluded: list, white_list=None):
    if white_list is None:
        white_list = []
    package_commands = _build_commands(cmd_type, poc_workspace, excluded, white_list)

    clients = _get_clients(package_commands)
    gpu_assignments: Dict[str, List[int]] = client_gpu_assignments(clients, gpu_ids)

    for package_name, cmd_path in package_commands:
        print(f"{cmd_type}: package: {package_name}, executing {cmd_path}")
        if package_name == SC.FLARE_CONSOLE:
            sync_process(cmd_path)
        elif package_name == SC.FLARE_SERVER:
            async_process(cmd_path, None)
        else:
            async_process(cmd_path, gpu_assignments[package_name])


def clean_poc(poc_workspace: str):
    import shutil

    if is_poc_ready(poc_workspace):
        shutil.rmtree(poc_workspace, ignore_errors=True)
        print(f"{poc_workspace} is removed")
    else:
        raise CLIException(f"{poc_workspace} is not valid poc directory")


def def_poc_parser(sub_cmd):
    cmd = "poc"
    poc_parser = sub_cmd.add_parser(cmd)
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
        "-ex",
        "--exclude",
        type=str,
        nargs="?",
        default="",
        help="exclude package directory during --start or --stop, default to " ", i.e. nothing to exclude",
    )
    poc_parser.add_argument(
        "-gpu",
        "--gpu",
        type=int,
        nargs="*",
        default="-1",
        help="gpu device ids will be used as CUDA_VISIBLE_DEVICES. used for poc start command",
    )
    poc_parser.add_argument(
        "--prepare",
        dest="prepare_poc",
        action="store_const",
        const=prepare_poc,
        help="prepare poc workspace. "
        + "export NVFLARE_HOME=<NVFLARE github cloned directory> to setup examples with prepare command",
    )
    poc_parser.add_argument("--start", dest="start_poc", action="store_const", const=start_poc, help="start poc")
    poc_parser.add_argument("--stop", dest="stop_poc", action="store_const", const=stop_poc, help="stop poc")
    poc_parser.add_argument(
        "--clean", dest="clean_poc", action="store_const", const=clean_poc, help="cleanup poc workspace"
    )
    return {cmd: poc_parser}


def is_poc(cmd_args) -> bool:
    return (
        hasattr(cmd_args, "start_poc")
        or hasattr(cmd_args, "prepare_poc")
        or hasattr(cmd_args, "stop_poc")
        or hasattr(cmd_args, "clean_poc")
    )


def get_local_host_gpu_ids():
    try:
        return get_host_gpu_ids()
    except Exception as e:
        raise CLIException(f"Failed to get host gpu ids:{e}")


def handle_poc_cmd(cmd_args):
    if cmd_args.package != "all":
        white_list = [cmd_args.package]
    else:
        white_list = []

    excluded = None
    if cmd_args.exclude != "":
        excluded = [cmd_args.exclude]

    poc_workspace = os.getenv("NVFLARE_POC_WORKSPACE")
    if poc_workspace is None or len(poc_workspace.strip()) == 0:
        poc_workspace = DEFAULT_WORKSPACE

    if cmd_args.start_poc:
        gpu_ids = get_gpu_ids(cmd_args.gpu, get_local_host_gpu_ids())
        start_poc(poc_workspace, gpu_ids, excluded, white_list)
    elif cmd_args.prepare_poc:
        prepare_poc(cmd_args.number_of_clients, poc_workspace)
    elif cmd_args.stop_poc:
        stop_poc(poc_workspace, excluded, white_list)
    elif cmd_args.clean_poc:
        clean_poc(poc_workspace)
    else:
        raise Exception(f"unable to handle poc command:{cmd_args}")
