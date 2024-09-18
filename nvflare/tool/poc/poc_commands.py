# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import time
from pathlib import Path
from typing import Dict, List, Optional, OrderedDict, Tuple

import yaml
from pyhocon import ConfigFactory as CF

from nvflare.cli_exception import CLIException
from nvflare.cli_unknown_cmd_exception import CLIUnknownCmdException
from nvflare.fuel.utils.class_utils import instantiate_class
from nvflare.fuel.utils.config import ConfigFormat
from nvflare.fuel.utils.gpu_utils import get_host_gpu_ids
from nvflare.lighter.provision import gen_default_project_config, prepare_project
from nvflare.lighter.spec import Provisioner
from nvflare.lighter.utils import load_yaml, update_project_server_name_config, update_storage_locations
from nvflare.tool.api_utils import shutdown_system
from nvflare.tool.poc.service_constants import FlareServiceConstants as SC
from nvflare.utils.cli_utils import get_hidden_nvflare_config_path, get_or_create_hidden_nvflare_dir, hocon_to_string

DEFAULT_WORKSPACE = "/tmp/nvflare/poc"
DEFAULT_PROJECT_NAME = "example_project"

CMD_PREPARE_POC = "prepare"
CMD_PREPARE_JOBS_DIR = "prepare-jobs-dir"
CMD_START_POC = "start"
CMD_STOP_POC = "stop"
CMD_CLEAN_POC = "clean"


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


def get_service_command(cmd_type: str, prod_dir: str, service_dir, service_config: Dict) -> str:
    cmd = ""
    proj_admin_dir_name = service_config.get(SC.FLARE_PROJ_ADMIN, SC.FLARE_PROJ_ADMIN)
    admin_dirs = service_config.get(SC.FLARE_OTHER_ADMINS, [])
    admin_dirs.append(proj_admin_dir_name)

    if cmd_type == SC.CMD_START:
        if not service_config.get(SC.IS_DOCKER_RUN):
            if service_dir in admin_dirs:
                cmd = get_cmd_path(prod_dir, service_dir, "fl_admin.sh")
            else:
                cmd = get_cmd_path(prod_dir, service_dir, "start.sh")
        else:
            if service_dir in admin_dirs:
                cmd = get_cmd_path(prod_dir, service_dir, "fl_admin.sh")
            else:
                cmd = get_cmd_path(prod_dir, service_dir, "docker.sh -d")

    elif cmd_type == SC.CMD_STOP:
        if not service_config.get(SC.IS_DOCKER_RUN):
            cmd = get_stop_cmd(prod_dir, service_dir)
        else:
            if service_dir in admin_dirs:
                cmd = get_stop_cmd(prod_dir, service_dir)
            else:
                cmd = f"docker stop {service_dir}"

    else:
        raise CLIException("unknown cmd_type :", cmd_type)
    return cmd


def get_stop_cmd(poc_workspace: str, service_dir_name: str):
    service_dir = os.path.join(poc_workspace, service_dir_name)
    stop_file = os.path.join(service_dir, "shutdown.fl")
    return f"touch {stop_file}"


def get_nvflare_home() -> Optional[str]:
    nvflare_home = None
    if "NVFLARE_HOME" in os.environ:
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


def is_dir_empty(path: str):
    targe_dir = os.listdir(path)
    return len(targe_dir) == 0


def prepare_jobs_dir(cmd_args):
    poc_workspace = get_poc_workspace()
    _prepare_jobs_dir(cmd_args.jobs_dir, poc_workspace)


def _prepare_jobs_dir(jobs_dir: str, workspace: str, config_packages: Optional[Tuple] = None):
    project_config, service_config = config_packages if config_packages else setup_service_config(workspace)
    project_name = project_config.get("name")
    if jobs_dir is None or jobs_dir == "":
        raise CLIException("jobs_dir is required")
    src = os.path.abspath(jobs_dir)
    if not os.path.isdir(src):
        raise CLIException(f"jobs_dir '{jobs_dir}' is not valid directory")

    prod_dir = get_prod_dir(workspace, project_name)
    if not os.path.exists(prod_dir):
        raise CLIException("please use nvflare poc prepare to create workspace first")

    console_dir = os.path.join(prod_dir, f"{service_config[SC.FLARE_PROJ_ADMIN]}")
    startup_dir = os.path.join(console_dir, SC.STARTUP)
    transfer = get_upload_dir(startup_dir)
    dst = os.path.join(console_dir, transfer)
    if not is_dir_empty(dst):
        print(" ")
        answer = input(f"job directory at {dst} is already exists, replace with new one ? (y/N) ")
        if answer.strip().upper() == "Y":
            if os.path.islink(dst):
                os.unlink(dst)
            if os.path.isdir(dst):
                shutil.rmtree(dst, ignore_errors=True)

            print(f"link job directory from {src} to {dst}")
            os.symlink(src, dst)
    else:
        if os.path.isdir(dst):
            shutil.rmtree(dst, ignore_errors=True)
        print(f"link job directory from {src} to {dst}")
        os.symlink(src, dst)


def get_prod_dir(workspace, project_name: str = DEFAULT_PROJECT_NAME):
    project_name = project_name if project_name else DEFAULT_PROJECT_NAME
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
        raise CLIException(f"project should only have one server, but {len(servers)} are provided: {servers}")


def get_fl_admins(project_config: OrderedDict, is_project_admin: bool):
    participants: List[dict] = project_config["participants"]
    return [
        p["name"]
        for p in participants
        if p["type"] == "admin" and (p["role"] == "project_admin" if is_project_admin else p["role"] != "project_admin")
    ]


def get_other_admins(project_config: OrderedDict):
    return get_fl_admins(project_config, is_project_admin=False)


def get_proj_admin(project_config: OrderedDict):
    admins = get_fl_admins(project_config, is_project_admin=True)
    if len(admins) == 1:
        return admins[0]
    else:
        raise CLIException(f"project should have only one project admin, but {len(admins)} are provided: {admins}")


def get_fl_client_names(project_config: OrderedDict) -> List[str]:
    participants: List[dict] = project_config["participants"]
    client_names = [p["name"] for p in participants if p["type"] == "client"]
    return client_names


def replace_server_with_localhost(sp_end_point: str) -> str:
    """
    :param sp_end_point:(str) example: server1:8002:8003
    :return: localhost:<port1>:<port2>
    """
    parts = sp_end_point.split(":")
    if len(parts) != 3:
        raise ValueError("Input must be in the format 'server:port1:port2'")
    for p in parts:
        if not p:
            raise ValueError("Input must be in the format 'server:port1:port2', each part can not be empty")

    parts[0] = "localhost"
    return ":".join(parts)


def prepare_builders(project_dict: OrderedDict) -> List:
    builders = list()
    for b in project_dict.get("builders"):
        path = b.get("path")
        args = b.get("args")

        if b.get("path") == "nvflare.lighter.impl.static_file.StaticFileBuilder":
            path = "nvflare.lighter.impl.local_static_file.LocalStaticFileBuilder"
            sp_end_point = args["overseer_agent"]["args"]["sp_end_point"]
            args["overseer_agent"]["args"]["sp_end_point"] = replace_server_with_localhost(sp_end_point)

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
) -> Tuple:
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
        raise CLIException(f"empty or invalid project config from project yaml file: {src_project_file}")

    if not user_provided_project_config:
        project_config = update_server_name(project_config)
        project_config = update_clients(clients, number_of_clients, project_config)
        project_config = add_he_builder(use_he, project_config)
        if docker_image:
            project_config = update_static_file_builder(docker_image, project_config)
    save_project_config(project_config, dst_project_file)
    service_config = get_service_config(project_config)
    project = prepare_project(project_config)
    builders = prepare_builders(project_config)
    provisioner = Provisioner(workspace, builders)
    provisioner.provision(project)

    return project_config, service_config


def get_service_config(project_config):
    service_config = {
        SC.FLARE_SERVER: get_fl_server_name(project_config),
        SC.FLARE_PROJ_ADMIN: get_proj_admin(project_config),
        SC.FLARE_OTHER_ADMINS: get_other_admins(project_config),
        SC.FLARE_CLIENTS: get_fl_client_names(project_config),
        SC.IS_DOCKER_RUN: is_docker_run(project_config),
    }
    return service_config


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
    if "builders" not in project_config:
        return False
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
        project_config["builders"].insert(-1, he_builder)

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


def save_startup_kit_dir_config(workspace, project_name):
    dst = get_or_create_hidden_nvflare_config_path()
    config = None
    if os.path.isfile(dst):
        try:
            config = CF.parse_file(dst)
        except Exception as e:
            config = None

    prod_dir = get_prod_dir(workspace, project_name)
    conf = f"""
        startup_kit {{
            path = {prod_dir}
        }}
        poc_workspace {{
            path = {workspace}
        }}
    """
    if config:
        new_config = CF.parse_string(conf)
        config = new_config.with_fallback(config)
        config_str = hocon_to_string(ConfigFormat.PYHOCON, config)
    else:
        config_str = conf

    with open(dst, "w") as file:
        file.write(f"{config_str}\n")


def prepare_poc(cmd_args):
    poc_workspace = get_poc_workspace()
    project_conf_path = ""
    if cmd_args.project_input:
        project_conf_path = cmd_args.project_input

    _prepare_poc(
        cmd_args.clients,
        cmd_args.number_of_clients,
        poc_workspace,
        cmd_args.docker_image,
        cmd_args.he,
        project_conf_path,
    )


def _prepare_poc(
    clients: List[str],
    number_of_clients: int,
    workspace: str,
    docker_image: str = None,
    use_he: bool = False,
    project_conf_path: str = "",
    examples_dir: Optional[str] = None,
) -> bool:
    if clients:
        number_of_clients = len(clients)
    if not project_conf_path:
        print(f"prepare poc at {workspace} for {number_of_clients} clients")
    else:
        print(f"prepare poc at {workspace} with {project_conf_path}")

    project_config = None
    if os.path.exists(workspace):
        answer = input(
            f"This will delete poc workspace directory: '{workspace}' and create a new one. Is it OK to proceed? (y/N) "
        )
        if answer.strip().upper() == "Y":

            workspace_path = Path(workspace)
            project_file = Path(project_conf_path)
            if workspace_path in project_file.parents:
                raise CLIException(
                    f"\nProject file: '{project_conf_path}' is under workspace directory:"
                    f"'{workspace}', which is to be deleted. "
                    f"Please copy {project_conf_path} to different location before running this command."
                )

            shutil.rmtree(workspace, ignore_errors=True)
        else:
            return False

    project_config = prepare_poc_provision(
        clients, number_of_clients, workspace, docker_image, use_he, project_conf_path, examples_dir
    )

    project_name = project_config.get("name") if project_config else None
    save_startup_kit_dir_config(workspace, project_name)
    return True


def get_or_create_hidden_nvflare_config_path() -> str:
    """
    Get the path for the hidden nvflare configuration file.

    Returns:
        str: The path to the hidden nvflare configuration file.
    """
    hidden_nvflare_dir = get_or_create_hidden_nvflare_dir()

    hidden_nvflare_config_file = get_hidden_nvflare_config_path(str(hidden_nvflare_dir))
    return hidden_nvflare_config_file


def prepare_poc_provision(
    clients: List[str],
    number_of_clients: int,
    workspace: str,
    docker_image: str,
    use_he: bool = False,
    project_conf_path: str = "",
    examples_dir: Optional[str] = None,
) -> Dict:
    os.makedirs(workspace, exist_ok=True)
    os.makedirs(os.path.join(workspace, "data"), exist_ok=True)
    project_config, service_config = local_provision(
        clients, number_of_clients, workspace, docker_image, use_he, project_conf_path
    )
    project_name = project_config.get("name")
    server_name = service_config[SC.FLARE_SERVER]
    # update storage
    if workspace != DEFAULT_WORKSPACE:
        prod_dir = get_prod_dir(workspace, project_name)
        update_storage_locations(local_dir=f"{prod_dir}/{server_name}/local", workspace=workspace)
    examples_dir = get_examples_dir(examples_dir)
    if examples_dir is not None:
        _prepare_jobs_dir(examples_dir, workspace, None)

    return project_config


def get_examples_dir(examples_dir):
    if examples_dir:
        return examples_dir
    nvflare_home = get_nvflare_home()
    default_examples_dir = os.path.join(nvflare_home, SC.EXAMPLES) if nvflare_home else None
    return default_examples_dir


def _sort_service_cmds(cmd_type, service_cmds: list, service_config) -> list:
    def sort_first(val):
        return val[0]

    order_services = []
    for service_name, cmd_path in service_cmds:
        if service_name == service_config[SC.FLARE_SERVER]:
            order_services.append((0, service_name, cmd_path))
        elif service_name == service_config[SC.FLARE_PROJ_ADMIN]:
            order_services.append((sys.maxsize, service_name, cmd_path))
        else:
            if len(service_cmds) == 1:
                order_services.append((0, service_name, cmd_path))
            else:
                order_services.append((random.randint(2, len(service_cmds)), service_name, cmd_path))

    order_services.sort(key=sort_first)
    if cmd_type == SC.CMD_STOP:
        order_services.reverse()
    return [(service_name, cmd_path) for n, service_name, cmd_path in order_services]


def get_cmd_path(poc_workspace, service_name, cmd):
    service_dir = os.path.join(poc_workspace, service_name)
    bin_dir = os.path.join(service_dir, SC.STARTUP)
    cmd_path = os.path.join(bin_dir, cmd)
    return cmd_path


def is_poc_ready(poc_workspace: str, service_config, project_config):
    # check server and admin directories exist
    project_name = project_config.get("name") if project_config else DEFAULT_PROJECT_NAME
    prod_dir = get_prod_dir(poc_workspace, project_name)
    console_dir = os.path.join(prod_dir, service_config[SC.FLARE_PROJ_ADMIN])
    server_dir = os.path.join(prod_dir, service_config[SC.FLARE_SERVER])
    return os.path.isdir(server_dir) and os.path.isdir(console_dir)


def validate_poc_workspace(poc_workspace: str, service_config, project_config=None):
    if not is_poc_ready(poc_workspace, service_config, project_config):
        raise CLIException(f"workspace {poc_workspace} is not ready, please use poc prepare to prepare poc workspace")


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


def start_poc(cmd_args):
    poc_workspace = get_poc_workspace()

    services_list = get_service_list(cmd_args)
    excluded = get_excluded(cmd_args)
    gpu_ids = get_gpis(cmd_args)

    _start_poc(poc_workspace, gpu_ids, excluded, services_list)


def get_gpis(cmd_args):
    if cmd_args.gpu is not None and isinstance(cmd_args.gpu, list) and len(cmd_args.gpu) > 0:
        gpu_ids = get_gpu_ids(cmd_args.gpu, get_local_host_gpu_ids())
    else:
        gpu_ids = []
    return gpu_ids


def get_excluded(cmd_args):
    excluded = None
    if cmd_args.exclude != "":
        excluded = [cmd_args.exclude]
    return excluded


def get_service_list(cmd_args):
    if cmd_args.service != "all":
        services_list = [cmd_args.service]
    else:
        services_list = []
    return services_list


def _start_poc(poc_workspace: str, gpu_ids: List[int], excluded=None, services_list=None):
    project_config, service_config = setup_service_config(poc_workspace)
    if services_list is None:
        services_list = []
    if excluded is None:
        excluded = []
    other_admins = service_config.get(SC.FLARE_OTHER_ADMINS, [])
    for admin_dir in other_admins:
        if admin_dir not in services_list:
            excluded.append(admin_dir)

    validate_services(project_config, services_list, excluded)
    validate_poc_workspace(poc_workspace, service_config, project_config)
    _run_poc(
        SC.CMD_START,
        poc_workspace,
        gpu_ids,
        service_config,
        project_config,
        excluded=excluded,
        services_list=services_list,
    )


def validate_services(project_config, services_list: List, excluded: List):
    participant_names = [p["name"] for p in project_config["participants"]]
    validate_participants(participant_names, services_list)
    validate_participants(participant_names, excluded)


def validate_participants(participant_names, list_participants):
    for p in list_participants:
        if p not in participant_names:
            print(f"participant '{p}' is not defined, expecting one of followings: {participant_names}")
            exit(1)


def setup_service_config(poc_workspace) -> Tuple:
    project_file = os.path.join(poc_workspace, "project.yml")
    if os.path.isfile(project_file):
        project_config = load_yaml(project_file)
        service_config = get_service_config(project_config) if project_config else None
        return project_config, service_config
    else:
        raise CLIException(f"{project_file} is missing, make sure you have first run 'nvflare poc prepare'")


def stop_poc(cmd_args):
    poc_workspace = get_poc_workspace()
    excluded = get_excluded(cmd_args)
    services_list = get_service_list(cmd_args)
    _stop_poc(poc_workspace, excluded, services_list)


def _stop_poc(poc_workspace: str, excluded=None, services_list=None):
    project_config, service_config = setup_service_config(poc_workspace)

    if services_list is None:
        services_list = []
    if excluded is None:
        excluded = [service_config[SC.FLARE_PROJ_ADMIN]]
    else:
        excluded.append(service_config[SC.FLARE_PROJ_ADMIN])

    validate_services(project_config, services_list, excluded)

    validate_poc_workspace(poc_workspace, service_config, project_config)
    gpu_ids: List[int] = []
    project_name = project_config.get("name")
    prod_dir = get_prod_dir(poc_workspace, project_name)

    p_size = len(services_list)
    if p_size == 0 or service_config[SC.FLARE_SERVER] in services_list:
        print("start shutdown NVFLARE")
        shutdown_system(prod_dir, username=service_config[SC.FLARE_PROJ_ADMIN])
    else:
        print(f"start shutdown {services_list}")

    _run_poc(
        SC.CMD_STOP,
        poc_workspace,
        gpu_ids,
        service_config,
        project_config,
        excluded=excluded,
        services_list=services_list,
    )


def _get_clients(service_commands: list, service_config) -> List[str]:
    clients = [
        service_dir_name
        for service_dir_name, _ in service_commands
        if service_dir_name != service_config[SC.FLARE_PROJ_ADMIN]
        and service_dir_name not in service_config.get(SC.FLARE_OTHER_ADMINS, [])
        and service_dir_name != service_config[SC.FLARE_SERVER]
    ]
    return clients


def _build_commands(
    cmd_type: str, poc_workspace: str, service_config, project_config, excluded: list, services_list=None
) -> list:
    """Builds commands.

    Args:
        cmd_type (str): start/stop
        poc_workspace (str): poc workspace directory path
        service_config (_type_): service_config
        excluded (list): excluded service/participants name
        services_list (_type_, optional): Service names. If empty, include every service/participants

    Returns:
        list: built commands
    """

    def is_fl_service_dir(p_dir_name: str) -> bool:
        fl_service = (
            p_dir_name == service_config[SC.FLARE_PROJ_ADMIN]
            or p_dir_name in service_config[SC.FLARE_OTHER_ADMINS]
            or p_dir_name == service_config[SC.FLARE_SERVER]
            or p_dir_name in service_config[SC.FLARE_CLIENTS]
        )
        return fl_service

    project_name = project_config.get("name")
    prod_dir = get_prod_dir(poc_workspace, project_name)

    if services_list is None:
        services_list = []
    service_commands = []
    for root, dirs, files in os.walk(prod_dir):
        if root == prod_dir:
            fl_dirs = [d for d in dirs if is_fl_service_dir(d)]
            for service_dir_name in fl_dirs:
                if service_dir_name not in excluded:
                    if len(services_list) == 0 or service_dir_name in services_list:
                        cmd = get_service_command(cmd_type, prod_dir, service_dir_name, service_config)
                        if cmd:
                            service_commands.append((service_dir_name, cmd))
    return _sort_service_cmds(cmd_type, service_commands, service_config)


def prepare_env(service_name, gpu_ids: Optional[List[int]], service_config: Dict):
    import os

    my_env = None
    if gpu_ids:
        my_env = os.environ.copy()
        if len(gpu_ids) > 0:
            my_env["CUDA_VISIBLE_DEVICES"] = ",".join([str(gid) for gid in gpu_ids])

    if service_config.get(SC.IS_DOCKER_RUN):
        my_env = os.environ.copy() if my_env is None else my_env
        if gpu_ids and len(gpu_ids) > 0:
            my_env["GPU2USE"] = f"--gpus={my_env['CUDA_VISIBLE_DEVICES']}"

        my_env["MY_DATA_DIR"] = os.path.join(get_poc_workspace(), "data")
        my_env["SVR_NAME"] = service_name

    return my_env


def async_process(service_name, cmd_path, gpu_ids: Optional[List[int]], service_config: Dict):
    my_env = prepare_env(service_name, gpu_ids, service_config)
    if my_env:
        subprocess.Popen(cmd_path.split(" "), env=my_env)
    else:
        subprocess.Popen(cmd_path.split(" "))


def sync_process(service_name, cmd_path):
    my_env = os.environ.copy()
    subprocess.run(cmd_path.split(" "), env=my_env)


def _run_poc(
    cmd_type: str,
    poc_workspace: str,
    gpu_ids: List[int],
    service_config: Dict,
    project_config: Dict,
    excluded: list,
    services_list=None,
):
    if services_list is None:
        services_list = []
    service_commands = _build_commands(cmd_type, poc_workspace, service_config, project_config, excluded, services_list)
    clients = _get_clients(service_commands, service_config)
    gpu_assignments: Dict[str, List[int]] = client_gpu_assignments(clients, gpu_ids)
    for service_name, cmd_path in service_commands:
        if service_name == service_config[SC.FLARE_PROJ_ADMIN]:
            # give other commands a chance to start first
            if len(service_commands) > 1:
                time.sleep(2)
            sync_process(service_name, cmd_path)
        elif service_name == service_config[SC.FLARE_SERVER]:
            async_process(service_name, cmd_path, None, service_config)
        else:
            time.sleep(1)
            gpu_ids = gpu_assignments[service_name] if service_name in clients else None
            async_process(service_name, cmd_path, gpu_ids, service_config)


def clean_poc(cmd_args):
    poc_workspace = get_poc_workspace()
    _clean_poc(poc_workspace)


def is_poc_running(poc_workspace, service_config, project_config):
    project_name = project_config.get("name") if project_config else DEFAULT_PROJECT_NAME
    prod_dir = get_prod_dir(poc_workspace, project_name)
    server_dir = os.path.join(prod_dir, service_config[SC.FLARE_SERVER])
    pid_file = os.path.join(server_dir, "pid.fl")
    return os.path.exists(pid_file)


def _clean_poc(poc_workspace: str):
    import shutil

    if os.path.isdir(poc_workspace):
        project_config, service_config = setup_service_config(poc_workspace)
        if project_config is not None:
            if is_poc_ready(poc_workspace, service_config, project_config):
                if not is_poc_running(poc_workspace, service_config, project_config):
                    shutil.rmtree(poc_workspace, ignore_errors=True)
                    print(f"{poc_workspace} is removed")
                else:
                    print("system is still running, please stop the system first.")
            else:
                raise CLIException(f"{poc_workspace} is not valid poc directory")
    else:
        raise CLIException(f"{poc_workspace} is not valid poc directory")


poc_sub_cmd_handlers = {
    CMD_PREPARE_POC: prepare_poc,
    CMD_PREPARE_JOBS_DIR: prepare_jobs_dir,
    CMD_START_POC: start_poc,
    CMD_STOP_POC: stop_poc,
    CMD_CLEAN_POC: clean_poc,
}


def def_poc_parser(sub_cmd):
    cmd = "poc"
    parser = sub_cmd.add_parser(cmd)
    add_legacy_options(parser)

    poc_parser = parser.add_subparsers(title=cmd, dest="poc_sub_cmd", help="poc subcommand")
    define_prepare_parser(poc_parser)
    define_prepare_jobs_parser(poc_parser)
    define_start_parser(poc_parser)
    define_stop_parser(poc_parser)
    define_clean_parser(poc_parser)
    return {cmd: parser}


def add_legacy_options(parser):
    parser.add_argument(
        "--prepare",
        dest="old_prepare_poc",
        action="store_const",
        const=old_prepare_poc,
        help="deprecated, suggest use 'nvflare poc prepare'",
    )
    parser.add_argument(
        "--start",
        dest="old_start_poc",
        action="store_const",
        const=old_start_poc,
        help="deprecated, suggest use 'nvflare poc start'",
    )
    parser.add_argument(
        "--stop",
        dest="old_stop_poc",
        action="store_const",
        const=old_stop_poc,
        help="deprecated, suggest use 'nvflare poc stop'",
    )
    parser.add_argument(
        "--clean",
        dest="old_clean_poc",
        action="store_const",
        const=old_clean_poc,
        help="deprecated, suggest use 'nvflare poc clean'",
    )


def old_start_poc():
    print(f"'nvflare poc --{CMD_START_POC}' is deprecated, please use 'nvflare poc {CMD_START_POC}' ")


def old_stop_poc():
    print(f"'nvflare poc --{CMD_STOP_POC}' is deprecated, please use 'nvflare poc {CMD_STOP_POC}' ")


def old_clean_poc():
    print(f"'nvflare poc --{CMD_CLEAN_POC}' is deprecated, please use 'nvflare poc {CMD_CLEAN_POC}' ")


def old_prepare_poc():
    print(f"'nvflare poc --{CMD_PREPARE_POC}' is deprecated, please use 'nvflare poc {CMD_PREPARE_POC}' ")


def define_prepare_parser(poc_parser, cmd: Optional[str] = None, help_str: Optional[str] = None):
    cmd = CMD_PREPARE_POC if cmd is None else cmd
    help_str = "prepare poc environment by provisioning local project" if help_str is None else help_str
    prepare_parser = poc_parser.add_parser(cmd, help=help_str)

    prepare_parser.add_argument(
        "-n", "--number_of_clients", type=int, nargs="?", default=2, help="number of sites or clients, default to 2"
    )
    prepare_parser.add_argument(
        "-c",
        "--clients",
        nargs="*",  # 0 or more values expected => creates a list
        type=str,
        default=[],  # default if nothing is provided
        help="Space separated client names. If specified, number_of_clients argument will be ignored.",
    )
    prepare_parser.add_argument(
        "-he",
        "--he",
        action="store_true",
        help="enable homomorphic encryption. ",
    )

    prepare_parser.add_argument(
        "-i",
        "--project_input",
        type=str,
        nargs="?",
        default="",
        help="project.yaml file path, If specified, "
        + "'number_of_clients','clients' and 'docker' specific options will be ignored.",
    )
    prepare_parser.add_argument(
        "-d",
        "--docker_image",
        nargs="?",
        default=None,
        const="nvflare/nvflare",
        help="generate docker.sh based on the docker_image, used in '--prepare' command. and generate docker.sh "
        + " 'start/stop' commands will start with docker.sh ",
    )

    prepare_parser.add_argument("-debug", "--debug", action="store_true", help="debug is on")


def define_prepare_jobs_parser(poc_parser):
    prepare_jobs_dir_parser = poc_parser.add_parser(CMD_PREPARE_JOBS_DIR, help="prepare jobs directory")
    prepare_jobs_dir_parser.add_argument("-j", "--jobs_dir", type=str, nargs="?", default=None, help="jobs directory")
    prepare_jobs_dir_parser.add_argument("-debug", "--debug", action="store_true", help="debug is on")


def define_clean_parser(poc_parser):
    clean_parser = poc_parser.add_parser(CMD_CLEAN_POC, help="clean up poc workspace")
    clean_parser.add_argument("-debug", "--debug", action="store_true", help="debug is on")


def define_start_parser(poc_parser):
    start_parser = poc_parser.add_parser(CMD_START_POC, help="start services in poc mode")

    start_parser.add_argument(
        "-p",
        "--service",
        type=str,
        nargs="?",
        default="all",
        help="participant, Default to all participants",
    )

    start_parser.add_argument(
        "-ex",
        "--exclude",
        type=str,
        nargs="?",
        default="",
        help="exclude service directory during 'start', default to " ", i.e. nothing to exclude",
    )
    start_parser.add_argument(
        "-gpu",
        "--gpu",
        type=int,
        nargs="*",
        default=None,
        help="gpu device ids will be used as CUDA_VISIBLE_DEVICES. used for poc start command",
    )
    start_parser.add_argument("-debug", "--debug", action="store_true", help="debug is on")


def define_stop_parser(poc_parser):
    stop_parser = poc_parser.add_parser(CMD_STOP_POC, help="stop services in poc mode")

    stop_parser.add_argument(
        "-p",
        "--service",
        type=str,
        nargs="?",
        default="all",
        help="participant, Default to all participants",
    )
    stop_parser.add_argument(
        "-ex",
        "--exclude",
        type=str,
        nargs="?",
        default="",
        help="exclude service directory during 'stop', default to " ", i.e. nothing to exclude",
    )
    stop_parser.add_argument("-debug", "--debug", action="store_true", help="debug is on")


def get_local_host_gpu_ids():
    try:
        return get_host_gpu_ids()
    except Exception as e:
        raise CLIException(f"Failed to get host gpu ids:{e}")


def handle_poc_cmd(cmd_args):
    if cmd_args.poc_sub_cmd:
        poc_cmd_handler = poc_sub_cmd_handlers.get(cmd_args.poc_sub_cmd, None)
        poc_cmd_handler(cmd_args)
    elif cmd_args.old_start_poc:
        old_start_poc()
    elif cmd_args.old_stop_poc:
        old_stop_poc()
    elif cmd_args.old_clean_poc:
        old_clean_poc()
    elif cmd_args.old_prepare_poc:
        old_prepare_poc()
    else:
        raise CLIUnknownCmdException("unknown command")


def get_poc_workspace():
    poc_workspace = os.getenv("NVFLARE_POC_WORKSPACE")

    if not poc_workspace:
        src_path = get_or_create_hidden_nvflare_config_path()
        if os.path.isfile(src_path):
            from pyhocon import ConfigFactory as CF

            config = CF.parse_file(src_path)
            poc_workspace = config.get("poc_workspace.path", None)

    if poc_workspace is None or len(poc_workspace.strip()) == 0:
        poc_workspace = DEFAULT_WORKSPACE

    return poc_workspace
