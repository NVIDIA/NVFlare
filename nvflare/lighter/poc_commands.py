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
import shutil
import socket
import subprocess
import sys
from typing import Dict, List, Optional, OrderedDict

import yaml

from nvflare.cli_exception import CLIException
from nvflare.fuel.utils.class_utils import instantiate_class
from nvflare.fuel.utils.gpu_utils import get_host_gpu_ids
from nvflare.lighter.provision import gen_default_project_config, prepare_project
from nvflare.lighter.service_constants import FlareServiceConstants as SC
from nvflare.lighter.spec import Provisioner
from nvflare.lighter.utils import load_yaml, update_project_server_name_config, update_storage_locations
from nvflare.tool.api_utils import shutdown_system

DEFAULT_WORKSPACE = "/tmp/nvflare/poc"
DEFAULT_PROJECT_NAME = "example_project"
global_packages = {}


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


def get_package_command(cmd_type: str, prod_dir: str, package_dir) -> str:
    cmd = ""
    if cmd_type == SC.CMD_START:
        admin_package = global_packages.get(SC.FLARE_PROJ_ADMIN, SC.FLARE_PROJ_ADMIN)
        if not global_packages.get(SC.IS_DOCKER_RUN):
            if package_dir == admin_package:
                cmd = get_cmd_path(prod_dir, package_dir, "fl_admin.sh")
            else:
                cmd = get_cmd_path(prod_dir, package_dir, "start.sh")
        else:
            if package_dir == admin_package:
                cmd = get_cmd_path(prod_dir, package_dir, "docker.sh")
            else:
                cmd = get_cmd_path(prod_dir, package_dir, "docker.sh -d")

    elif cmd_type == SC.CMD_STOP:

        if not global_packages.get(SC.IS_DOCKER_RUN):
            cmd = get_stop_cmd(prod_dir, package_dir)
        else:
            cmd = f"docker stop {package_dir}"

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


def get_upload_dir(startup_dir) -> str:
    console_config_path = os.path.join(startup_dir, "fed_admin.json")
    try:
        with open(console_config_path, "r") as f:
            console_config = json.load(f)
            upload_dir = console_config["admin"]["upload_dir"]
    except IOError as e:
        raise CLIException(f"failed to load {console_config_path} {e}")
    except json.decoder.JSONDecodeError as e:
        raise CLIException(f"failed to load {console_config_path}, please double check the configuration {e}")

    return upload_dir


def prepare_examples(example_dir: str, workspace: str):
    if example_dir is None or example_dir == "":
        raise ValueError("example_dir is required")
    src = os.path.abspath(example_dir)
    if not os.path.isdir(src):
        raise ValueError(f"example_dir '{example_dir}' is not valid directory")

    prod_dir = get_prod_dir(workspace)
    if not os.path.exists(prod_dir):
        print("please use nvflare local --prepare to create workspace first")
        exit(0)

    startup_dir = os.path.join(prod_dir, f"{global_packages[SC.FLARE_PROJ_ADMIN]}/{SC.STARTUP}")
    transfer = get_upload_dir(startup_dir)
    dst = os.path.join(startup_dir, transfer)
    print(f"link examples from {src} to {dst}")
    if not os.path.islink(dst):
        os.symlink(src, dst)


def get_prod_dir(workspace, project_name: str = DEFAULT_PROJECT_NAME):
    prod_dir = os.path.join(workspace, project_name, "prod_00")
    return prod_dir


def gen_project_config_file(workspace: str) -> str:
    project_file = os.path.join(workspace, "project.yml")
    if not os.path.isfile(project_file):
        gen_default_project_config("dummy_project.yml", project_file)
    return project_file


def verify_host(host_name: str) -> bool:
    try:
        host_name = socket.gethostbyname(host_name)
        return True
    except:
        return False


def verify_hosts(project_config: OrderedDict):
    hosts: List[str] = get_project_hosts(project_config)
    for h in hosts:
        if not verify_host(h):
            print(f"host name: '{h}' is not defined, considering modify /etc/hosts to add localhost alias")
            exit(0)


def get_project_hosts(project_config) -> List[str]:
    participants: List[dict] = project_config["participants"]
    return [p["name"] for p in participants if p["type"] == "client" or p["type"] == "server"]


def get_fl_server_name(project_config: OrderedDict) -> str:
    participants: List[dict] = project_config["participants"]
    servers = [p["name"] for p in participants if p["type"] == "server"]
    if len(servers) == 1:
        return servers[0]
    else:
        raise ValueError(f"project should only have one server, but {len(servers)} are provided: {servers}")


def get_proj_admin(project_config: OrderedDict):
    participants: List[dict] = project_config["participants"]
    admins = [p["name"] for p in participants if p["type"] == "admin"]

    if len(admins) == 1:
        return admins[0]
    else:
        raise ValueError(f"project should only have only one project admin, but {len(admins)} are provided: {admins}")


def get_fl_client_names(project_config: OrderedDict) -> List[str]:
    participants: List[dict] = project_config["participants"]
    client_names = [p["name"] for p in participants if p["type"] == "client"]
    return client_names


def prepare_builders(project_dict: OrderedDict) -> List:
    builders = list()
    admin_name = [p["name"] for p in project_dict["participants"] if p["type"] == "admin"][0]
    for b in project_dict.get("builders"):
        path = b.get("path")
        args = b.get("args")

        if b.get("path") == "nvflare.lighter.impl.static_file.StaticFileBuilder":
            path = "nvflare.lighter.impl.local_static_file.LocalStaticFileBuilder"
            args["overseer_agent"]["args"]["sp_end_point"] = "localhost:8002:8003"
            args["username"] = admin_name

        elif b.get("path") == "nvflare.lighter.impl.cert.CertBuilder":
            path = "nvflare.lighter.impl.local_cert.LocalCertBuilder"

        builders.append(instantiate_class(path, args))
    return builders


def local_provision(
        clients: List[str],
        number_of_clients: int,
        workspace: str,
        docker_image: str,
        use_he: bool = False,
        project_conf_path: str = "",
) -> dict:
    user_provided_project_config = False
    if project_conf_path:
        src_project_file = project_conf_path
        dst_project_file = os.path.join(workspace, "project.yml")
        user_provided_project_config = True
    else:
        src_project_file = gen_project_config_file(workspace)
        dst_project_file = src_project_file

    print(f"provision at {workspace} for {number_of_clients} clients with {src_project_file}")
    project_config: OrderedDict = load_yaml(src_project_file)
    if not project_config:
        raise ValueError(f"empty or invalid project config from project yaml file: {src_project_file}")

    if not user_provided_project_config:
        project_config = update_server_name(project_config)
        project_config = update_clients(clients, number_of_clients, project_config)
        project_config = add_he_builder(use_he, project_config)
        if docker_image:
            project_config = update_static_file_builder(docker_image, project_config)
    save_project_config(project_config, dst_project_file)
    packages = get_packages(project_config)
    project = prepare_project(project_config)
    builders = prepare_builders(project_config)
    provisioner = Provisioner(workspace, builders)
    provisioner.provision(project)

    return packages


def get_packages(project_config):
    packages = {
        SC.FLARE_SERVER: get_fl_server_name(project_config),
        SC.FLARE_PROJ_ADMIN: get_proj_admin(project_config),
        SC.FLARE_CLIENTS: get_fl_client_names(project_config),
        SC.IS_DOCKER_RUN: is_docker_run(project_config),
    }
    return packages


def save_project_config(project_config, project_file):
    with open(project_file, "w") as file:
        yaml.dump(project_config, file)


def update_server_name(project_config):
    old_server_name = get_fl_server_name(project_config)
    server_name = "server"
    if old_server_name != server_name:
        update_project_server_name_config(project_config, old_server_name, server_name)
    return project_config


def is_docker_run(project_config: OrderedDict):
    static_builder = [
        b
        for b in project_config.get("builders")
        if b.get("path") == "nvflare.lighter.impl.static_file.StaticFileBuilder"
    ][0]
    return "docker_image" in static_builder["args"]


def update_static_file_builder(docker_image: str, project_config: OrderedDict):
    # need to keep the order of the builders
    for b in project_config.get("builders"):
        if b.get("path") == "nvflare.lighter.impl.static_file.StaticFileBuilder":
            b["args"]["docker_image"] = docker_image

    return project_config


def add_docker_builder(use_docker: bool, project_config: OrderedDict):
    if use_docker:
        docker_builder = {
            "path": "nvflare.lighter.impl.docker.DockerBuilder",
            "args": {"base_image": "python:3.8", "requirements_file": "requirements.txt"},
        }
        project_config["builders"].append(docker_builder)

    return project_config


def add_he_builder(use_he: bool, project_config: OrderedDict):
    if use_he:
        he_builder = {
            "path": "nvflare.lighter.impl.he.HEBuilder",
            "args": {},
        }
        project_config["builders"].append(he_builder)

    return project_config


def update_clients(clients: List[str], n_clients: int, project_config: OrderedDict) -> OrderedDict:
    requested_clients = prepare_clients(clients, n_clients)

    participants: List[dict] = project_config["participants"]
    new_participants = [p for p in participants if p["type"] != "client"]

    for client in requested_clients:
        client_dict = {"name": client, "type": "client", "org": "nvidia"}
        new_participants.append(client_dict)

    project_config["participants"] = new_participants

    return project_config


def prepare_clients(clients, number_of_clients):
    if not clients:
        clients = []
        for i in range(number_of_clients):
            clients.append(f"site-{(i + 1)}")

    return clients


def prepare_poc(
        clients: List[str],
        number_of_clients: int,
        workspace: str,
        docker_image: str,
        use_he: bool,
        project_conf_path: str = "",
) -> bool:
    if clients:
        number_of_clients = len(clients)
    if not project_conf_path:
        print(f"prepare poc at {workspace} for {number_of_clients} clients")
    else:
        print(f"prepare poc at {workspace} with {project_conf_path}")

    if os.path.exists(workspace):
        answer = input(
            f"This will delete poc folder in {workspace} directory and create a new one. Is it OK to proceed? (y/N) "
        )
        if answer.strip().upper() == "Y":
            shutil.rmtree(workspace, ignore_errors=True)
            prepare_poc_provision(clients, number_of_clients, workspace, docker_image, use_he, project_conf_path)
            return True
        else:
            return False
    else:
        prepare_poc_provision(clients, number_of_clients, workspace, docker_image, use_he, project_conf_path)
        return True


def prepare_poc_provision(
        clients: List[str],
        number_of_clients: int,
        workspace: str,
        docker_image: str,
        use_he: bool = False,
        project_conf_path: str = "",
):
    os.makedirs(workspace, exist_ok=True)
    os.makedirs(os.path.join(workspace, "data"), exist_ok=True)
    global global_packages
    global_packages = local_provision(clients, number_of_clients, workspace, docker_image, use_he, project_conf_path)
    server_name = global_packages[SC.FLARE_SERVER]
    # update storage
    if workspace != DEFAULT_WORKSPACE:
        update_storage_locations(
            local_dir=f"{workspace}/{server_name}/local", default_resource_name="resources.json", workspace=workspace
        )


def sort_package_cmds(cmd_type, package_cmds: list) -> list:
    def sort_first(val):
        return val[0]

    order_packages = []
    for package_name, cmd_path in package_cmds:
        if package_name == global_packages[SC.FLARE_SERVER]:
            order_packages.append((0, package_name, cmd_path))
        elif package_name == global_packages[SC.FLARE_PROJ_ADMIN]:
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
    prod_dir = get_prod_dir(poc_workspace)
    console_dir = os.path.join(prod_dir, global_packages[SC.FLARE_PROJ_ADMIN])
    server_dir = os.path.join(prod_dir, global_packages[SC.FLARE_SERVER])
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
    project_config = setup_global_packages(poc_workspace)
    if white_list is None:
        white_list = []
    if excluded is None:
        excluded = []
    print(f"start_poc at {poc_workspace}, gpu_ids={gpu_ids}, excluded = {excluded}, white_list={white_list}")
    validate_packages(project_config, white_list, excluded)
    validate_poc_workspace(poc_workspace)
    _run_poc(SC.CMD_START, poc_workspace, gpu_ids, excluded=excluded, white_list=white_list)


def validate_packages(project_config, white_list: List, excluded: List):
    participant_names = [p["name"] for p in project_config["participants"]]
    validate_participants(participant_names, white_list)
    validate_participants(participant_names, excluded)


def validate_participants(participant_names, list_participants):
    for p in list_participants:
        if p not in participant_names:
            print(f"package for participant '{p}' is not defined, expecting one of followings: {participant_names}")
            exit(1)


def setup_global_packages(poc_workspace) -> Optional[Dict]:
    project_file = os.path.join(poc_workspace, "project.yml")
    if os.path.isfile(project_file):
        project_config = load_yaml(project_file)
        global global_packages
        global_packages = get_packages(project_config)
        return project_config
    else:
        raise ValueError(f"{project_file} is missing, make sure you have first run 'nvflare poc --prepare'")


def stop_poc(poc_workspace: str, excluded=None, white_list=None):
    project_config = setup_global_packages(poc_workspace)

    if white_list is None:
        white_list = []
    if excluded is None:
        excluded = [global_packages[SC.FLARE_PROJ_ADMIN]]
    else:
        excluded.append(global_packages[SC.FLARE_PROJ_ADMIN])

    validate_packages(project_config, white_list)

    validate_poc_workspace(poc_workspace)
    gpu_ids: List[int] = []
    prod_dir = get_prod_dir(poc_workspace)

    p_size = len(white_list)
    if p_size == 0 or global_packages[SC.FLARE_SERVER] in white_list:
        print("start shutdown NVFLARE")
        shutdown_system(prod_dir, username=global_packages[SC.FLARE_PROJ_ADMIN])
    else:
        print(f"start shutdown {white_list}")

    _run_poc(SC.CMD_STOP, poc_workspace, gpu_ids, excluded=excluded, white_list=white_list)


def _get_clients(package_commands: list) -> List[str]:
    clients = [
        package_dir_name
        for package_dir_name, _ in package_commands
        if package_dir_name != global_packages[SC.FLARE_PROJ_ADMIN]
           and package_dir_name != global_packages[SC.FLARE_SERVER]
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
        return (
                p_dir_name == global_packages[SC.FLARE_PROJ_ADMIN]
                or p_dir_name == global_packages[SC.FLARE_SERVER]
                or p_dir_name in global_packages[SC.FLARE_CLIENTS]
        )

    prod_dir = get_prod_dir(poc_workspace)

    if white_list is None:
        white_list = []
    package_commands = []
    for root, dirs, files in os.walk(prod_dir):
        if root == prod_dir:
            fl_dirs = [d for d in dirs if is_fl_package_dir(d)]
            for package_dir_name in fl_dirs:
                if package_dir_name not in excluded:
                    if len(white_list) == 0 or package_dir_name in white_list:
                        cmd = get_package_command(cmd_type, prod_dir, package_dir_name)
                        if cmd:
                            package_commands.append((package_dir_name, cmd))
    return sort_package_cmds(cmd_type, package_commands)


def prepare_env(package_name, gpu_ids: Optional[List[int]] = None):
    import os
    my_env = None
    if gpu_ids:
        my_env = os.environ.copy()
        if len(gpu_ids) > 0:
            my_env["CUDA_VISIBLE_DEVICES"] = ",".join([str(gid) for gid in gpu_ids])

    if global_packages.get(SC.IS_DOCKER_RUN):
        my_env = os.environ.copy() if my_env is None else my_env
        if gpu_ids and len(gpu_ids) > 0:
            my_env["GPU2USE"] = f"--gpus={my_env['CUDA_VISIBLE_DEVICES']}"

        my_env["MY_DATA_DIR"] = os.path.join(get_poc_workspace(), "data")
        my_env["SVR_NAME"] = package_name

    return my_env


def async_process(package_name, cmd_path, gpu_ids: Optional[List[int]] = None):
    my_env = prepare_env(package_name, gpu_ids)
    if my_env:
        subprocess.Popen(cmd_path.split(" "), env=my_env)
    else:
        subprocess.Popen(cmd_path.split(" "))


def sync_process(package_name, cmd_path):
    my_env = os.environ.copy()
    subprocess.run(cmd_path.split(" "), env=my_env)


def _run_poc(cmd_type: str, poc_workspace: str, gpu_ids: List[int], excluded: list, white_list=None):
    if white_list is None:
        white_list = []
    package_commands = _build_commands(cmd_type, poc_workspace, excluded, white_list)
    clients = _get_clients(package_commands)
    gpu_assignments: Dict[str, List[int]] = client_gpu_assignments(clients, gpu_ids)
    for package_name, cmd_path in package_commands:
        if package_name == global_packages[SC.FLARE_PROJ_ADMIN]:
            sync_process(package_name, cmd_path)
        elif package_name == global_packages[SC.FLARE_SERVER]:
            async_process(package_name, cmd_path, None)
        else:
            async_process(package_name, cmd_path, gpu_assignments[package_name])


def clean_poc(poc_workspace: str):
    import shutil

    if os.path.isdir(poc_workspace):
        project_config = setup_global_packages(poc_workspace)
        if project_config is not None:
            if is_poc_ready(poc_workspace):
                shutil.rmtree(poc_workspace, ignore_errors=True)
                print(f"{poc_workspace} is removed")
            else:
                raise CLIException(f"{poc_workspace} is not valid poc directory")
    else:
        raise CLIException(f"{poc_workspace} is not valid poc directory")


def def_poc_parser(sub_cmd):
    cmd = "poc"
    poc_parser = sub_cmd.add_parser(cmd)
    poc_parser.add_argument(
        "-n", "--number_of_clients", type=int, nargs="?", default=2, help="number of sites or clients, default to 2"
    )
    poc_parser.add_argument(
        "-c",
        "--clients",
        nargs="*",  # 0 or more values expected => creates a list
        type=str,
        default=[],  # default if nothing is provided
        help="Space separated client names. If specified, number_of_clients argument will be ignored.",
    )
    poc_parser.add_argument(
        "-p",
        "--package",
        type=str,
        nargs="?",
        default="all",
        help="participant, Default to all participants, only used for start/stop poc commands when specified",
    )
    poc_parser.add_argument(
        "-e",
        "--examples",
        type=str,
        nargs="?",
        default="all",
        help="examples directory, only used in '--prepare-examples' command",
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
        default=None,
        help="gpu device ids will be used as CUDA_VISIBLE_DEVICES. used for poc start command",
    )
    poc_parser.add_argument(
        "-he",
        "--he",
        action="store_true",
        help="enable homomorphic encryption. Use with '--prepare' command ",
    )

    poc_parser.add_argument(
        "-i",
        "--project_input",
        type=str,
        nargs="?",
        default="",
        help="project.yaml file path, it should be used with '--prepare' command. If specified, "
             + "'number_of_clients','clients' and 'docker' specific options will be ignored.",
    )
    poc_parser.add_argument(
        "-d",
        "--docker_image",
        nargs="?",
        default=None,
        const="nvflare/nvflare",
        help="generate docker.sh based on the docker_image, used in '--prepare' command. and generate docker.sh "
             + " '--start/stop' commands will start with docker.sh ",
    )
    poc_parser.add_argument(
        "--prepare",
        dest="prepare_poc",
        action="store_const",
        const=prepare_poc,
        help="prepare poc workspace and provision",
    )

    poc_parser.add_argument(
        "--prepare-examples",
        dest="prepare_examples",
        action="store_const",
        const=prepare_examples,
        help="create an symbolic link to the examples directory, requires nvflare_example directory with '-e'",
    )

    poc_parser.add_argument("--start", dest="start_poc", action="store_const", const=start_poc, help="start local")
    poc_parser.add_argument("--stop", dest="stop_poc", action="store_const", const=stop_poc, help="stop local")
    poc_parser.add_argument(
        "--clean", dest="clean_poc", action="store_const", const=clean_poc, help="cleanup local workspace"
    )
    return {cmd: poc_parser}


def is_poc(cmd_args) -> bool:
    return (
            hasattr(cmd_args, "start_poc")
            or hasattr(cmd_args, "prepare_poc")
            or hasattr(cmd_args, "stop_poc")
            or hasattr(cmd_args, "clean_poc")
            or hasattr(cmd_args, "prepare_examples")
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

    if cmd_args.gpu is not None and cmd_args.prepare_poc:
        raise ValueError("-gpu should not be used for 'nvflare poc --prepare' command ")

    poc_workspace = get_poc_workspace()
    if cmd_args.start_poc:
        if cmd_args.gpu is not None and isinstance(cmd_args.gpu, list) and len(cmd_args.gpu) > 0:
            gpu_ids = get_gpu_ids(cmd_args.gpu, get_local_host_gpu_ids())
        else:
            gpu_ids = []
        start_poc(poc_workspace, gpu_ids, excluded, white_list)
    elif cmd_args.prepare_poc:
        project_conf_path = ""
        if cmd_args.project_input:
            project_conf_path = cmd_args.project_input
        prepare_poc(
            cmd_args.clients,
            cmd_args.number_of_clients,
            poc_workspace,
            cmd_args.docker_image,
            cmd_args.he,
            project_conf_path,
        )
    elif cmd_args.prepare_examples:
        prepare_examples(cmd_args.examples, poc_workspace)
    elif cmd_args.stop_poc:
        stop_poc(poc_workspace, excluded, white_list)
    elif cmd_args.clean_poc:
        clean_poc(poc_workspace)
    else:
        raise Exception(f"unable to handle poc command:{cmd_args}")


def get_poc_workspace():
    poc_workspace = os.getenv("NVFLARE_POC_WORKSPACE")
    if poc_workspace is None or len(poc_workspace.strip()) == 0:
        poc_workspace = DEFAULT_WORKSPACE
    return poc_workspace
