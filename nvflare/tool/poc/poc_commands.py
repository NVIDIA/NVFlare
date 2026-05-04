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
import contextlib
import copy
import errno
import io
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, OrderedDict, Tuple

import yaml
from pyhocon import ConfigFactory as CF

from nvflare.cli_exception import CLIException
from nvflare.cli_unknown_cmd_exception import CLIUnknownCmdException
from nvflare.fuel.utils.gpu_utils import get_host_gpu_ids
from nvflare.lighter.constants import CtxKey, PropKey, ProvisionMode
from nvflare.lighter.prov_utils import prepare_builders, prepare_packager
from nvflare.lighter.provision import gen_default_project_config, prepare_project
from nvflare.lighter.provisioner import Provisioner
from nvflare.lighter.utils import (
    load_yaml,
    update_project_server_name_config,
    update_server_default_host,
    update_storage_locations,
)
from nvflare.tool.api_utils import SystemStartTimeout, shutdown_system, wait_for_system_start
from nvflare.tool.kit.kit_config import (
    STARTUP_KIT_KIND_ADMIN,
    STARTUP_KIT_KIND_SITE,
    StartupKitConfigError,
    add_startup_kit_entry,
    classify_startup_kit,
    get_active_startup_kit_id,
    get_startup_kit_entries,
    load_cli_config,
    remove_entries_under_workspace,
    remove_startup_kit_entry,
    save_cli_config,
)
from nvflare.tool.poc.service_constants import FlareServiceConstants as SC
from nvflare.utils.cli_utils import (
    CONFIG_VERSION,
    CURRENT_CONFIG_VERSION,
    create_poc_workspace_config,
    get_hidden_nvflare_config_path,
    get_or_create_hidden_nvflare_dir,
    load_hidden_config_state,
    save_config,
)

DEFAULT_WORKSPACE = "/tmp/nvflare/poc"
DEFAULT_PROJECT_NAME = "example_project"

CMD_PREPARE_POC = "prepare"
CMD_START_POC = "start"
CMD_STOP_POC = "stop"
CMD_CLEAN_POC = "clean"
CMD_CONFIG_POC = "config"
CMD_ADD_USER = "add-user"
CMD_ADD_SITE = "add-site"

# POC prepare creates the single Project Admin. Dynamic POC add can only add secondary users.
POC_USER_CERT_ROLES = ("org_admin", "lead", "member")
POC_KEY = "poc"
STARTUP_KIT_KEY = "startup_kit"
WORKSPACE_KEY = "workspace"
POC_START_READY_TIMEOUT = 30
POC_DEFAULT_FED_LEARN_PORT = 8002
POC_DEFAULT_ADMIN_PORT = 8003
POC_LOCAL_HOST = "localhost"
POC_PORT_PREFLIGHT_HOST = "127.0.0.1"
POC_PORT_PREFLIGHT_SCOPE = "loopback"
POC_PORT_PREFLIGHT_NOTE = "Preflight checks loopback port availability only; poc start may still fail if another local bind address conflicts."
POC_SERVICE_CONSOLE_LOG = "poc_console.log"
POC_RUNTIME_KEY = "poc_runtime"
POC_RUNTIME_DOCKER = "docker"
POC_DOCKER_IMAGE_KEY = "docker_image"
POC_DOCKER_NETWORK_KEY = "network"
POC_DOCKER_DEFAULT_NETWORK = "nvflare-network"
POC_DOCKER_SERVER_ALIAS = "server"
POC_DOCKER_DATA_STUDY = "default"
POC_DOCKER_DATA_DATASET = "poc"
POC_DOCKER_DATA_MOUNT = f"/data/{POC_DOCKER_DATA_STUDY}/{POC_DOCKER_DATA_DATASET}"


def _quiet_cli_streams(enabled: bool):
    if not enabled:
        return contextlib.nullcontext()

    stack = contextlib.ExitStack()
    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
    stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
    return stack


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
            gpu_assignments[client].append(gpu_id)
    return gpu_assignments


def get_service_command(
    cmd_type: str, prod_dir: str, service_dir, service_config: Dict, study: Optional[str] = None
) -> str:
    cmd = ""
    proj_admin_dir_name = service_config.get(SC.FLARE_PROJ_ADMIN, SC.FLARE_PROJ_ADMIN)
    admin_dirs = list(service_config.get(SC.FLARE_OTHER_ADMINS, []))
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
                cmd = get_cmd_path(prod_dir, service_dir, "start_docker.sh")

    elif cmd_type == SC.CMD_STOP:
        if not service_config.get(SC.IS_DOCKER_RUN):
            cmd = get_stop_cmd(prod_dir, service_dir)
        else:
            if service_dir in admin_dirs:
                cmd = get_stop_cmd(prod_dir, service_dir)
            else:
                cmd = f"docker stop {service_dir}"

    else:
        raise CLIException(f"unknown cmd_type: {cmd_type}")

    if cmd_type == SC.CMD_START and study and service_dir in admin_dirs and cmd.endswith("fl_admin.sh"):
        cmd = f"{cmd} --study {study}"
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
    return not os.listdir(path)


def _link_jobs_dir_to_admin_transfer(
    jobs_dir: str, workspace: str, config_packages: Optional[Tuple] = None, force: bool = False
) -> bool:
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
    from nvflare.tool.cli_output import print_human, prompt_yn

    if not is_dir_empty(dst):
        if not force:
            if not sys.stdin.isatty():
                raise CLIException(
                    f"jobs directory {dst} already exists; use --force to overwrite in non-interactive mode"
                )
            if not prompt_yn(f"Jobs directory already exists: {dst}. Replace it?"):
                return False
        if os.path.islink(dst):
            os.unlink(dst)
        if os.path.isdir(dst):
            shutil.rmtree(dst, ignore_errors=True)
        print_human(f"link job directory from {src} to {dst}")
        os.symlink(src, dst)
    else:
        if os.path.islink(dst):
            os.unlink(dst)
        if os.path.isdir(dst):
            shutil.rmtree(dst, ignore_errors=True)
        print_human(f"link job directory from {src} to {dst}")
        os.symlink(src, dst)
    return True


def _get_prod_dirs(workspace, project_name: str = DEFAULT_PROJECT_NAME) -> List[str]:
    project_name = project_name if project_name else DEFAULT_PROJECT_NAME
    project_dir = os.path.join(workspace, project_name)
    prod_dirs = []
    if os.path.isdir(project_dir):
        for name in os.listdir(project_dir):
            if not name.startswith("prod_"):
                continue
            try:
                stage = int(name.split("_")[-1])
            except ValueError:
                continue
            prod_dirs.append((stage, os.path.join(project_dir, name)))

    return [path for _, path in sorted(prod_dirs)]


def get_prod_dir(workspace, project_name: str = DEFAULT_PROJECT_NAME):
    project_name = project_name if project_name else DEFAULT_PROJECT_NAME
    prod_dirs = _get_prod_dirs(workspace, project_name)
    if prod_dirs:
        return prod_dirs[-1]
    return os.path.join(workspace, project_name, "prod_00")


def gen_project_config_file(workspace: str) -> str:
    project_file = os.path.join(workspace, "project.yml")
    if not os.path.isfile(project_file):
        gen_default_project_config("dummy_project.yml", project_file)
    return project_file


def verify_host(host_name: str) -> bool:
    try:
        host_name = socket.gethostbyname(host_name)
        return True
    except Exception:
        return False


def verify_hosts(project_config: OrderedDict):
    hosts: List[str] = get_project_hosts(project_config)
    for h in hosts:
        if not verify_host(h):
            from nvflare.tool.cli_output import print_human

            print_human(f"host name: '{h}' is not defined, considering modify /etc/hosts to add localhost alias")


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

    from nvflare.tool.cli_output import print_human

    print_human(f"provision at {workspace} for {number_of_clients} clients with {src_project_file}")
    project_config: OrderedDict = load_yaml(src_project_file)
    if not project_config:
        raise CLIException(f"empty or invalid project config from project yaml file: {src_project_file}")

    if not user_provided_project_config:
        project_config = update_server_name(project_config)
        project_config = update_clients(clients, number_of_clients, project_config)
        project_config = add_he_builder(use_he, project_config)
        if docker_image:
            project_config = add_poc_docker_runtime(docker_image, project_config)
    project_config = update_server_default_host(project_config, "localhost")
    save_project_config(project_config, dst_project_file)
    service_config = get_service_config(project_config)
    provision_config = copy.deepcopy(project_config)
    project = prepare_project(provision_config)
    builders = prepare_builders(provision_config)
    packager = prepare_packager(provision_config)
    provisioner = Provisioner(workspace, builders, packager)
    provisioner.provision(project, mode=ProvisionMode.POC)

    return project_config, service_config


def get_service_config(project_config):
    docker_run_mode = get_docker_run_mode(project_config)
    service_config = {
        SC.FLARE_SERVER: get_fl_server_name(project_config),
        SC.FLARE_PROJ_ADMIN: get_proj_admin(project_config),
        SC.FLARE_OTHER_ADMINS: get_other_admins(project_config),
        SC.FLARE_CLIENTS: get_fl_client_names(project_config),
        SC.IS_DOCKER_RUN: bool(docker_run_mode),
        SC.DOCKER_RUN_MODE: docker_run_mode,
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


def get_docker_run_mode(project_config: OrderedDict):
    runtime_config = _get_poc_runtime_config(project_config)
    if runtime_config.get("runtime") == POC_RUNTIME_DOCKER:
        return SC.DOCKER_RUN_MODE_DEPLOY
    return ""


def is_docker_run(project_config: OrderedDict):
    return bool(get_docker_run_mode(project_config))


def add_poc_docker_runtime(docker_image: str, project_config: OrderedDict):
    project_config[POC_RUNTIME_KEY] = {
        "runtime": POC_RUNTIME_DOCKER,
        POC_DOCKER_IMAGE_KEY: docker_image,
        POC_DOCKER_NETWORK_KEY: POC_DOCKER_DEFAULT_NETWORK,
    }
    return project_config


def _get_poc_runtime_config(project_config: Dict) -> Mapping:
    runtime_config = project_config.get(POC_RUNTIME_KEY)
    if runtime_config is None:
        return {}
    if not isinstance(runtime_config, Mapping):
        raise CLIException(f"{POC_RUNTIME_KEY} must be a mapping")

    runtime = runtime_config.get("runtime")
    if runtime is not None and not isinstance(runtime, str):
        raise CLIException(f"{POC_RUNTIME_KEY}.runtime must be a string")
    return runtime_config


def get_poc_docker_runtime_config(project_config: Dict) -> Optional[Dict]:
    runtime_config = _get_poc_runtime_config(project_config)
    if runtime_config.get("runtime") != POC_RUNTIME_DOCKER:
        return None

    parent_config = runtime_config.get("parent")
    if parent_config is None or parent_config == {}:
        parent = runtime_config
    else:
        if not isinstance(parent_config, Mapping):
            raise CLIException(f"{POC_RUNTIME_KEY}.parent must be a mapping")
        parent = parent_config

    job_launcher_config = runtime_config.get("job_launcher")
    if job_launcher_config is None or job_launcher_config == {}:
        job_launcher = {}
    else:
        if not isinstance(job_launcher_config, Mapping):
            raise CLIException(f"{POC_RUNTIME_KEY}.job_launcher must be a mapping")
        job_launcher = job_launcher_config

    docker_image = parent.get(POC_DOCKER_IMAGE_KEY)
    if not docker_image:
        raise CLIException(f"{POC_RUNTIME_KEY}.{POC_DOCKER_IMAGE_KEY} is required for Docker POC runtime")

    return {
        "runtime": POC_RUNTIME_DOCKER,
        "parent": {
            "docker_image": docker_image,
            "network": parent.get(POC_DOCKER_NETWORK_KEY, POC_DOCKER_DEFAULT_NETWORK),
        },
        "job_launcher": job_launcher,
    }


def _prepare_poc_docker_deployments(poc_workspace: str, project_config: Dict):
    docker_config = get_poc_docker_runtime_config(project_config)
    if not docker_config:
        return False

    project_name = project_config.get("name")
    prod_dir = get_prod_dir(poc_workspace, project_name)
    participants = project_config.get("participants", [])
    for participant in participants:
        p_type = participant.get("type")
        if p_type not in ("server", "client"):
            continue
        kit_dir = os.path.join(prod_dir, participant["name"])
        _prepare_poc_docker_kit(kit_dir, docker_config, poc_workspace)
    return True


def _write_poc_docker_study_data(kit_dir: Path, poc_workspace: str) -> None:
    study_data_file = kit_dir / "local" / "study_data.yaml"
    try:
        with study_data_file.open("rt", encoding="utf-8") as f:
            study_data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        study_data = {}
    except yaml.YAMLError as e:
        raise CLIException(f"invalid study data file for POC Docker kit '{kit_dir}': {e}") from e

    if not isinstance(study_data, dict):
        raise CLIException(f"study data file for POC Docker kit '{kit_dir}' must contain a dictionary")

    study_entry = study_data.setdefault(POC_DOCKER_DATA_STUDY, {})
    if not isinstance(study_entry, dict):
        raise CLIException(
            f"study data entry '{POC_DOCKER_DATA_STUDY}' for POC Docker kit '{kit_dir}' must contain a dictionary"
        )
    study_entry.setdefault(
        POC_DOCKER_DATA_DATASET,
        {"source": os.path.realpath(os.path.join(poc_workspace, "data")), "mode": "rw"},
    )

    with study_data_file.open("wt", encoding="utf-8") as f:
        yaml.safe_dump(study_data, f, default_flow_style=False, sort_keys=False)


def _prepare_poc_docker_kit(kit_dir: str, docker_config: Dict, poc_workspace: str):
    from nvflare.tool.deploy.deploy_commands import (
        ROLE_CLIENT,
        RUNTIME_DOCKER,
        _prepare_docker,
        _validate_kit,
        _validate_runtime_config,
    )

    kit_path = Path(kit_dir)
    try:
        _validate_runtime_config(RUNTIME_DOCKER, docker_config)
        kit_info = _validate_kit(kit_path)
        _prepare_docker(kit_info, kit_path, docker_config)
        _write_poc_docker_study_data(kit_path, poc_workspace)
        if kit_info.role == ROLE_CLIENT:
            _patch_poc_docker_client_target(kit_path)
    except SystemExit as e:
        raise CLIException(f"failed to prepare Docker deployment for startup kit '{kit_dir}'") from e
    except Exception as e:
        raise CLIException(f"failed to prepare Docker deployment for startup kit '{kit_dir}': {e}") from e


def _patch_poc_docker_client_target(kit_dir: Path):
    fed_client_file = kit_dir / SC.STARTUP / "fed_client.json"
    with fed_client_file.open("rt", encoding="utf-8") as f:
        fed_client = json.load(f)

    changed = False
    for server in fed_client.get("servers") or []:
        service = server.get("service") or {}
        target = service.get("target")
        if not isinstance(target, str) or ":" not in target:
            continue
        host, port = target.rsplit(":", 1)
        if host not in {POC_LOCAL_HOST, "127.0.0.1", "0.0.0.0"}:
            continue
        service["target"] = f"{POC_DOCKER_SERVER_ALIAS}:{port}"
        changed = True

    if changed:
        with fed_client_file.open("wt", encoding="utf-8") as f:
            json.dump(fed_client, f, indent=2)
            f.write("\n")


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


def _is_startup_kit_kind(kit_dir: str, expected_kind: str) -> bool:
    try:
        kind, _ = classify_startup_kit(kit_dir)
    except StartupKitConfigError:
        return False
    return kind == expected_kind


def _get_generated_poc_startup_kits(project_config: Dict, prod_dir: str) -> Tuple[Dict[str, str], Optional[str]]:
    participants = project_config.get("participants", [])
    if not isinstance(participants, list):
        raise CLIException("project.yml participants must be a list")

    entries = {}
    active_id = None
    for participant in participants:
        if not isinstance(participant, dict):
            raise CLIException("participant entry must be a mapping")
        identity = participant.get("name")
        if not identity:
            raise CLIException("participant missing name")

        kit_dir = os.path.abspath(os.path.join(prod_dir, identity))
        participant_type = participant.get("type")
        if participant_type == "admin" and _is_startup_kit_kind(kit_dir, STARTUP_KIT_KIND_ADMIN):
            entries[identity] = kit_dir
            if active_id is None and participant.get("role") == "project_admin":
                active_id = identity

    return entries, active_id


def _get_configured_poc_workspace(config: Dict[str, Any]) -> Optional[str]:
    try:
        workspace = config.get(f"{POC_KEY}.{WORKSPACE_KEY}", None)
    except Exception:
        return None
    return workspace.strip() if isinstance(workspace, str) and workspace.strip() else None


def _register_poc_startup_kits(
    config: Dict[str, Any], workspace: str, kit_entries: Dict[str, str]
) -> Tuple[Dict[str, Any], set]:
    removed_ids = set()
    previous_workspace = _get_configured_poc_workspace(config)
    if previous_workspace:
        config, removed_ids = remove_entries_under_workspace(config, previous_workspace)
    config, current_removed_ids = remove_entries_under_workspace(config, workspace)
    removed_ids.update(current_removed_ids)
    entries = get_startup_kit_entries(config)

    for kit_id, kit_path in kit_entries.items():
        existing_path = entries.get(kit_id)
        if existing_path:
            if os.path.exists(os.path.expanduser(existing_path)):
                raise CLIException(
                    f"startup kit id '{kit_id}' already exists outside POC workspace; "
                    f"run 'nvflare config remove {kit_id}' or replace it explicitly"
                )
            config = remove_startup_kit_entry(config, kit_id)
            removed_ids.add(kit_id)
        config = add_startup_kit_entry(config, kit_id, kit_path, force=True)

    return config, removed_ids


def _write_poc_startup_kit_registry(workspace: str, project_name: str, project_config: Dict):
    config = load_cli_config()
    prod_dir = get_prod_dir(workspace, project_name)
    kit_entries, active_id = _get_generated_poc_startup_kits(project_config, prod_dir)
    config, _removed_ids = _register_poc_startup_kits(config, workspace, kit_entries)

    if active_id:
        config.put("startup_kits.active", active_id)
    else:
        from nvflare.tool.cli_output import print_human

        print_human(
            "No generated Project Admin startup kit was found; "
            "run 'nvflare config use <id>' after registering an admin startup kit."
        )

    config.put(f"{POC_KEY}.{WORKSPACE_KEY}", workspace)
    try:
        config.pop(f"{POC_KEY}.{STARTUP_KIT_KEY}", None)
        config.pop("prod.startup_kit", None)
    except Exception:
        pass
    save_cli_config(config)


def save_startup_kit_dir_config(workspace, project_name):
    project_file = os.path.join(workspace, "project.yml")
    try:
        project_config = load_yaml(project_file) if os.path.isfile(project_file) else None
    except Exception:
        project_config = None
    if not project_config or not isinstance(project_config, dict):
        raise CLIException(f"invalid or unreadable project config: {project_file}")
    _write_poc_startup_kit_registry(workspace, project_name, project_config)


def _load_poc_project_config(poc_workspace: str) -> Tuple[str, Dict]:
    project_file = os.path.join(poc_workspace, "project.yml")
    if not os.path.isfile(project_file):
        raise CLIException("please use nvflare poc prepare to create workspace first")

    project_config = load_yaml(project_file)
    if not isinstance(project_config, dict):
        raise CLIException(f"invalid or unreadable project config: {project_file}")

    participants = project_config.get("participants")
    if not isinstance(participants, list):
        raise CLIException("project.yml participants must be a list")

    return project_file, project_config


def _find_participant_index(participants: List[Dict], name: str) -> Optional[int]:
    for index, participant in enumerate(participants):
        if isinstance(participant, dict) and participant.get("name") == name:
            return index
    return None


def _upsert_poc_participant(project_config: Dict, participant: Dict, force: bool) -> str:
    participants = project_config.get("participants")
    index = _find_participant_index(participants, participant["name"])
    if index is None:
        participants.append(participant)
        return "added"

    existing = participants[index]
    if existing.get("type") != participant.get("type"):
        raise CLIException(
            f"participant '{participant['name']}' already exists as type '{existing.get('type')}', "
            f"not '{participant.get('type')}'"
        )

    if not force:
        raise CLIException(f"participant '{participant['name']}' already exists; use --force to replace it")

    if existing.get("type") == "admin" and existing.get("role") != participant.get("role"):
        raise CLIException(
            f"participant '{participant['name']}' already exists with role '{existing.get('role')}'; "
            "changing a POC user's certificate role requires nvflare poc clean and a fresh prepare"
        )

    updated = dict(existing)
    updated.update(participant)
    participants[index] = updated
    return "updated"


def _restore_poc_active_kit(previous_active: Optional[str], preferred_active: Optional[str] = None):
    config = load_cli_config()
    entries = get_startup_kit_entries(config)
    active_id = None
    if previous_active and previous_active in entries:
        active_id = previous_active
    elif preferred_active and preferred_active in entries:
        active_id = preferred_active

    if active_id:
        config.put("startup_kits.active", active_id)
        save_cli_config(config)


def _get_active_startup_kit_id_safely() -> Optional[str]:
    try:
        return get_active_startup_kit_id(load_cli_config())
    except Exception:
        return None


def _is_local_port_available(port: int, host: str = POC_PORT_PREFLIGHT_HOST) -> Tuple[bool, Optional[str]]:
    try:
        port = int(port)
    except (TypeError, ValueError):
        return False, "invalid_port"

    if port < 1 or port > 65535:
        return False, "invalid_port"

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, port))
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            return False, "in_use"
        return False, e.strerror or str(e)

    return True, None


def _get_poc_server_port_specs(project_config: Dict) -> List[Dict]:
    if not isinstance(project_config, dict):
        return []

    try:
        server_name = get_fl_server_name(project_config)
    except Exception:
        server_name = None

    participants = project_config.get("participants", [])
    if not isinstance(participants, list):
        return []

    for participant in participants:
        if not isinstance(participant, dict) or participant.get("type") != "server":
            continue
        if server_name and participant.get("name") != server_name:
            continue

        fed_learn_port = participant.get(PropKey.FED_LEARN_PORT, POC_DEFAULT_FED_LEARN_PORT)
        admin_port = participant.get(PropKey.ADMIN_PORT, POC_DEFAULT_ADMIN_PORT)
        specs = [
            {"name": PropKey.FED_LEARN_PORT, "port": fed_learn_port},
            {"name": PropKey.ADMIN_PORT, "port": admin_port},
        ]
        deduped_specs = []
        seen_ports = set()
        for spec in specs:
            port = spec.get("port")
            if port in seen_ports:
                continue
            seen_ports.add(port)
            deduped_specs.append(spec)
        return deduped_specs

    return []


def _build_poc_port_preflight(project_config: Dict, host: str = POC_PORT_PREFLIGHT_HOST) -> Dict:
    port_specs = _get_poc_server_port_specs(project_config)
    if not port_specs:
        return {
            "checked": False,
            "host": host,
            "scope": POC_PORT_PREFLIGHT_SCOPE,
            "ports": [],
            "conflicts": [],
            "message": "server port configuration is not available",
            "note": POC_PORT_PREFLIGHT_NOTE,
        }

    checked_ports = []
    conflicts = []
    for spec in port_specs:
        port = spec.get("port")
        available, reason = _is_local_port_available(port, host=host)
        checked = {
            "name": spec.get("name"),
            "port": port,
            "available": available,
            "conflict": not available,
        }
        if reason:
            checked["reason"] = reason
        checked_ports.append(checked)

        if not available:
            conflict = dict(checked)
            conflict["message"] = f"Port {port} is not available on {host}: {reason}"
            conflicts.append(conflict)

    return {
        "checked": True,
        "host": host,
        "scope": POC_PORT_PREFLIGHT_SCOPE,
        "ports": checked_ports,
        "conflicts": conflicts,
        "note": POC_PORT_PREFLIGHT_NOTE,
    }


def _get_poc_server_participant(project_config: Dict, service_config: Dict = None) -> Optional[Dict]:
    if not isinstance(project_config, dict):
        return None

    server_name = service_config.get(SC.FLARE_SERVER) if service_config else None
    if not server_name:
        try:
            server_name = get_fl_server_name(project_config)
        except Exception:
            server_name = None

    participants = project_config.get("participants", [])
    if not isinstance(participants, list):
        return None

    for participant in participants:
        if not isinstance(participant, dict) or participant.get("type") != "server":
            continue
        if server_name and participant.get("name") != server_name:
            continue
        return participant
    return None


def _get_poc_server_ports(project_config: Dict, service_config: Dict = None) -> Tuple[int, int]:
    server_participant = _get_poc_server_participant(project_config, service_config)
    if not server_participant:
        return POC_DEFAULT_FED_LEARN_PORT, POC_DEFAULT_ADMIN_PORT

    fed_learn_port = server_participant.get(PropKey.FED_LEARN_PORT, POC_DEFAULT_FED_LEARN_PORT)
    try:
        fed_learn_port = int(fed_learn_port)
    except (TypeError, ValueError):
        fed_learn_port = POC_DEFAULT_FED_LEARN_PORT

    admin_port = server_participant.get(PropKey.ADMIN_PORT, POC_DEFAULT_ADMIN_PORT)
    try:
        admin_port = int(admin_port)
    except (TypeError, ValueError):
        admin_port = POC_DEFAULT_ADMIN_PORT

    return fed_learn_port, admin_port


def _build_poc_endpoint_info(project_config: Dict, service_config: Dict = None) -> Dict:
    fed_learn_port, admin_port = _get_poc_server_ports(project_config, service_config)
    server_address = f"{POC_LOCAL_HOST}:{fed_learn_port}"
    admin_address = f"{POC_LOCAL_HOST}:{admin_port}"
    return {
        "server_url": f"grpc://{server_address}",
        "server_address": server_address,
        "admin_address": admin_address,
        "default_port": POC_DEFAULT_FED_LEARN_PORT,
        "default_server_port": POC_DEFAULT_FED_LEARN_PORT,
        "default_admin_port": POC_DEFAULT_ADMIN_PORT,
    }


def _build_poc_start_port_preflight(
    project_config: Dict, service_config: Dict, services_list: List[str], excluded: List[str]
) -> Dict:
    if not project_config or not service_config:
        return {
            "checked": False,
            "host": POC_PORT_PREFLIGHT_HOST,
            "scope": POC_PORT_PREFLIGHT_SCOPE,
            "ports": [],
            "conflicts": [],
            "message": "server port configuration is not available",
            "note": POC_PORT_PREFLIGHT_NOTE,
        }

    starts_server, _ = _get_started_readiness_participants(service_config, services_list, excluded)
    if not starts_server:
        return {
            "checked": False,
            "host": POC_PORT_PREFLIGHT_HOST,
            "scope": POC_PORT_PREFLIGHT_SCOPE,
            "ports": [],
            "conflicts": [],
            "message": "server was not selected for startup",
            "note": POC_PORT_PREFLIGHT_NOTE,
        }

    return _build_poc_port_preflight(project_config)


def _poc_port_warnings(port_preflight: Dict) -> List[str]:
    return [conflict.get("message") for conflict in port_preflight.get("conflicts", []) if conflict.get("message")]


def _build_poc_port_diagnostics(port_preflight: Dict) -> Dict:
    return {
        "port_conflict": bool(port_preflight.get("conflicts")),
        "port_preflight": port_preflight,
        "warnings": _poc_port_warnings(port_preflight),
    }


def _get_existing_poc_prod_dir(poc_workspace: str, project_name: str) -> str:
    prod_dirs = _get_prod_dirs(poc_workspace, project_name)
    if not prod_dirs:
        raise CLIException("POC workspace has no existing provisioned output; run 'nvflare poc prepare' first")
    return prod_dirs[-1]


def _remove_path(path: str):
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


class _PocDynamicProvisionLogger:
    def debug(self, msg: str):
        pass

    def info(self, msg: str):
        pass

    def warning(self, msg: str):
        pass

    def error(self, msg: str):
        pass


def _dynamic_poc_project_config(project_config: Dict, participant: Dict) -> Dict:
    dynamic_config = copy.deepcopy(project_config)
    participants = project_config.get("participants", [])
    servers = [p for p in participants if isinstance(p, dict) and p.get("type") == "server"]
    if len(servers) != 1:
        raise CLIException(f"project should only have one server, but {len(servers)} are provided")

    dynamic_config["participants"] = [copy.deepcopy(servers[0]), copy.deepcopy(participant)]
    # Dynamic POC provisioning only builds the new participant kit. Study registry updates
    # belong to the running server metadata, not to this short-lived reduced project.
    dynamic_config.pop("studies", None)
    return dynamic_config


def _ensure_dynamic_poc_ca_available(poc_workspace: str, project_name: str, prod_dir: str, project_config: Dict) -> str:
    state_file = os.path.join(poc_workspace, project_name, "state", "cert.json")
    if not os.path.isfile(state_file):
        raise CLIException(
            "existing POC CA state was not found; run 'nvflare poc prepare' before using POC add commands"
        )

    server_name = get_fl_server_name(project_config)
    root_ca_path = os.path.join(prod_dir, server_name, SC.STARTUP, "rootCA.pem")
    if not os.path.isfile(root_ca_path):
        raise CLIException(
            "existing POC rootCA.pem was not found; run 'nvflare poc prepare' before using POC add commands"
        )
    return root_ca_path


def _provision_poc_participant_only(
    poc_workspace: str,
    project_config: Dict,
    participant: Dict,
    target_prod_dir: str,
    force: bool = False,
) -> str:
    project_name = project_config.get("name") or DEFAULT_PROJECT_NAME
    _ensure_dynamic_poc_ca_available(poc_workspace, project_name, target_prod_dir, project_config)
    dynamic_config = _dynamic_poc_project_config(project_config, participant)

    project = prepare_project(dynamic_config)
    builders = prepare_builders(dynamic_config)
    packager = prepare_packager(dynamic_config)
    provisioner = Provisioner(poc_workspace, builders, packager)

    temp_prod_dir = None
    try:
        ctx = provisioner.provision(project, mode=ProvisionMode.POC, logger=_PocDynamicProvisionLogger())
        if ctx.get(CtxKey.BUILD_ERROR):
            raise CLIException("dynamic POC provisioning failed; see provisioning output for details")

        temp_prod_dir = ctx.get(CtxKey.CURRENT_PROD_DIR)
        if not temp_prod_dir:
            raise CLIException("dynamic POC provisioning did not produce an output directory")

        participant_name = participant["name"]
        generated_kit = os.path.join(temp_prod_dir, participant_name)
        if not os.path.isdir(generated_kit):
            raise CLIException(f"startup kit was not generated for participant '{participant_name}'")

        target_kit = os.path.join(target_prod_dir, participant_name)
        if os.path.exists(target_kit):
            if not force:
                raise CLIException(f"startup kit already exists for participant '{participant_name}'")
            _remove_path(target_kit)

        shutil.move(generated_kit, target_kit)
        if get_docker_run_mode(project_config) == SC.DOCKER_RUN_MODE_DEPLOY and participant.get("type") in (
            "server",
            "client",
        ):
            _prepare_poc_docker_kit(target_kit, get_poc_docker_runtime_config(project_config), poc_workspace)
        return target_kit
    finally:
        if temp_prod_dir and os.path.isdir(temp_prod_dir):
            shutil.rmtree(temp_prod_dir, ignore_errors=True)


def _dynamic_poc_provision(
    poc_workspace: str,
    project_file: str,
    project_config: Dict,
    participant: Dict,
    expected_kit_kind: str,
    previous_active: Optional[str],
    preferred_active: Optional[str] = None,
    force: bool = False,
) -> Tuple[Dict, str]:
    project_name = project_config.get("name") if project_config else DEFAULT_PROJECT_NAME
    prod_dir = _get_existing_poc_prod_dir(poc_workspace, project_name)
    startup_kit = _provision_poc_participant_only(
        poc_workspace,
        project_config,
        participant,
        prod_dir,
        force=force,
    )
    if not _is_startup_kit_kind(startup_kit, expected_kit_kind):
        raise CLIException(f"startup kit was not generated for participant '{participant['name']}'")

    save_project_config(project_config, project_file)
    save_startup_kit_dir_config(poc_workspace, project_name)
    _restore_poc_active_kit(previous_active, preferred_active)
    return project_config, prod_dir


def _add_poc_user(poc_workspace: str, cert_role: str, email: str, org: str, force: bool = False) -> Dict:
    if cert_role not in POC_USER_CERT_ROLES:
        raise CLIException(
            f"unsupported POC user certificate role '{cert_role}'; "
            f"valid roles are: {', '.join(POC_USER_CERT_ROLES)}"
        )
    project_file, project_config = _load_poc_project_config(poc_workspace)
    previous_active = get_active_startup_kit_id(load_cli_config())
    participant = {"name": email, "type": "admin", "org": org, "role": cert_role}
    action = _upsert_poc_participant(project_config, participant, force)
    preferred_active = email if not previous_active else None
    project_config, prod_dir = _dynamic_poc_provision(
        poc_workspace,
        project_file,
        project_config,
        participant,
        STARTUP_KIT_KIND_ADMIN,
        previous_active,
        preferred_active=preferred_active,
        force=force,
    )
    startup_kit = os.path.join(prod_dir, email)
    if not _is_startup_kit_kind(startup_kit, STARTUP_KIT_KIND_ADMIN):
        raise CLIException(f"startup kit was not generated for user '{email}'")

    result = {
        "status": action,
        "type": "user",
        "id": email,
        "identity": email,
        "org": org,
        "cert_role": cert_role,
        "startup_kit": startup_kit,
        "prod_dir": prod_dir,
    }
    if preferred_active:
        result["active"] = email
    else:
        result["next_step"] = f"nvflare config use {email}"
    return result


def _add_poc_site(poc_workspace: str, name: str, org: str, force: bool = False) -> Dict:
    project_file, project_config = _load_poc_project_config(poc_workspace)
    previous_active = get_active_startup_kit_id(load_cli_config())
    participant = {"name": name, "type": "client", "org": org}
    action = _upsert_poc_participant(project_config, participant, force)
    project_config, prod_dir = _dynamic_poc_provision(
        poc_workspace,
        project_file,
        project_config,
        participant,
        STARTUP_KIT_KIND_SITE,
        previous_active,
        force=force,
    )
    startup_kit = os.path.join(prod_dir, name)
    if not _is_startup_kit_kind(startup_kit, STARTUP_KIT_KIND_SITE):
        raise CLIException(f"startup kit was not generated for site '{name}'")

    return {
        "status": action,
        "type": "site",
        "id": name,
        "name": name,
        "org": org,
        "startup_kit": startup_kit,
        "prod_dir": prod_dir,
    }


def add_poc_user(cmd_args):
    from nvflare.tool.cli_output import is_json_mode, output_error, output_ok, print_human
    from nvflare.tool.cli_schema import handle_schema_flag

    json_mode = is_json_mode()
    handle_schema_flag(
        _poc_sub_cmd_parsers.get(CMD_ADD_USER),
        "nvflare poc add-user",
        ["nvflare poc add-user lead bob@nvidia.com --org nvidia"],
        sys.argv[1:],
    )
    poc_workspace = get_poc_workspace()
    try:
        with _quiet_cli_streams(json_mode):
            result = _add_poc_user(
                poc_workspace,
                cmd_args.cert_role,
                cmd_args.email,
                cmd_args.org,
                force=getattr(cmd_args, "force", False),
            )
    except StartupKitConfigError as e:
        output_error("STARTUP_KIT_MISSING", exit_code=4, detail=str(e), hint=getattr(e, "hint", ""))
        raise SystemExit(4)
    except ValueError as e:
        output_error("STARTUP_KIT_MISSING", exit_code=4, detail=str(e))
        raise SystemExit(4)
    except CLIException as e:
        output_error("INVALID_ARGS", exit_code=4, detail=str(e))
        raise SystemExit(4)
    except Exception as e:
        output_error("INTERNAL_ERROR", exit_code=5, detail=str(e))
        raise SystemExit(5)

    if json_mode:
        output_ok(result)
        return
    print_human(f"\nPOC user {result['status']}: {result['identity']}")
    print_human(f"  Startup kit: {result['startup_kit']}")
    if result.get("active"):
        print_human(f"  Active startup kit: {result['active']}")
    else:
        print_human(f"  Activate with: {result['next_step']}")


def add_poc_site(cmd_args):
    from nvflare.tool.cli_output import is_json_mode, output_error, output_ok, print_human
    from nvflare.tool.cli_schema import handle_schema_flag

    json_mode = is_json_mode()
    handle_schema_flag(
        _poc_sub_cmd_parsers.get(CMD_ADD_SITE),
        "nvflare poc add-site",
        ["nvflare poc add-site site-3 --org nvidia"],
        sys.argv[1:],
    )
    poc_workspace = get_poc_workspace()
    try:
        with _quiet_cli_streams(json_mode):
            result = _add_poc_site(
                poc_workspace,
                cmd_args.name,
                cmd_args.org,
                force=getattr(cmd_args, "force", False),
            )
    except StartupKitConfigError as e:
        output_error("STARTUP_KIT_MISSING", exit_code=4, detail=str(e), hint=getattr(e, "hint", ""))
        raise SystemExit(4)
    except ValueError as e:
        output_error("STARTUP_KIT_MISSING", exit_code=4, detail=str(e))
        raise SystemExit(4)
    except CLIException as e:
        output_error("INVALID_ARGS", exit_code=4, detail=str(e))
        raise SystemExit(4)
    except Exception as e:
        output_error("INTERNAL_ERROR", exit_code=5, detail=str(e))
        raise SystemExit(5)

    if json_mode:
        output_ok(result)
        return
    print_human(f"\nPOC site {result['status']}: {result['name']}")
    print_human(f"  Startup kit: {result['startup_kit']}")
    print_human(f"  Start with: nvflare poc start -p {result['name']}")


def prepare_poc(cmd_args):
    from nvflare.tool.cli_output import is_json_mode, output_error, output_ok
    from nvflare.tool.cli_schema import handle_schema_flag
    from nvflare.tool.install_skills import install_skills

    json_mode = is_json_mode()
    handle_schema_flag(
        _poc_sub_cmd_parsers.get(CMD_PREPARE_POC),
        "nvflare poc prepare",
        ["nvflare poc prepare -n 2", "nvflare poc prepare -n 3 --force"],
        sys.argv[1:],
    )
    poc_workspace = get_poc_workspace()
    project_conf_path = ""
    if cmd_args.project_input:
        project_conf_path = cmd_args.project_input

    force = getattr(cmd_args, "force", False)

    if os.path.exists(poc_workspace) and not force:
        if not sys.stdin.isatty():
            output_error(
                "INVALID_ARGS",
                exit_code=4,
                detail="workspace exists; use --force to overwrite in non-interactive mode",
            )
            raise SystemExit(4)
        # Interactive: let _prepare_poc handle the prompt
    prior_active_startup_kit = _get_active_startup_kit_id_safely()
    try:
        with _quiet_cli_streams(json_mode):
            result = _prepare_poc(
                cmd_args.clients,
                cmd_args.number_of_clients,
                poc_workspace,
                cmd_args.docker_image,
                cmd_args.he,
                project_conf_path,
                force=force,
            )
    except CLIException as e:
        output_error("INVALID_ARGS", exit_code=4, detail=str(e))
        raise SystemExit(4)
    except Exception as e:
        output_error("INTERNAL_ERROR", exit_code=5, detail=str(e))
        raise SystemExit(5)

    if result is False:
        return  # user said no at prompt

    # Gather client names
    project_file = os.path.join(poc_workspace, "project.yml")
    project_config = None
    clients = (
        list(cmd_args.clients) if cmd_args.clients else [f"site-{i + 1}" for i in range(cmd_args.number_of_clients)]
    )
    try:
        pc = load_yaml(project_file)
        if pc:
            project_config = pc
            participants = pc.get("participants", [])
            if not isinstance(participants, list):
                raise CLIException("project.yml participants must be a list")
            clients = []
            for p in participants:
                if not isinstance(p, dict):
                    raise CLIException("participant entry must be a mapping")
                if p.get("type") == "client":
                    name = p.get("name")
                    if not name:
                        raise CLIException("client participant missing name")
                    clients.append(name)
    except (OSError, IOError, yaml.YAMLError):
        # If the post-provision readback fails, preserve the best-known client list instead of
        # silently reporting an empty set in the success payload.
        pass
    except CLIException as e:
        output_error("INVALID_ARGS", exit_code=4, detail=str(e))
        raise SystemExit(4)

    active_startup_kit = _get_active_startup_kit_id_safely()
    result = {
        "workspace": poc_workspace,
        "clients": clients,
        "startup_kit": {
            "prior_active": prior_active_startup_kit,
            "active": active_startup_kit,
            "changed": prior_active_startup_kit != active_startup_kit,
        },
        "port_preflight": _build_poc_port_preflight(project_config),
    }
    if json_mode:
        try:
            with _quiet_cli_streams(True):
                install_skills()
        except Exception:
            pass
        output_ok(result)
        return
    from nvflare.tool.cli_output import print_human

    print_human(f"\nPOC workspace ready at: {poc_workspace}")
    print_human(f"  Clients: {', '.join(clients) if clients else '(none)'}")
    print_human("  Next: place your jobs under the admin transfer folder, then run 'nvflare poc start'")
    print_human("  That starts the server and clients only; admin consoles must be started explicitly.")
    try:
        install_skills()
    except Exception:
        pass


def _prepare_poc(
    clients: List[str],
    number_of_clients: int,
    workspace: str,
    docker_image: Optional[str] = None,
    use_he: bool = False,
    project_conf_path: str = "",
    examples_dir: Optional[str] = None,
    force: bool = False,
) -> bool:
    if clients:
        number_of_clients = len(clients)
    from nvflare.tool.cli_output import print_human, prompt_yn

    if not project_conf_path:
        print_human(f"Preparing POC workspace at {workspace} for {number_of_clients} clients...")
    else:
        print_human(f"Preparing POC workspace at {workspace} using {project_conf_path}...")

    if os.path.exists(workspace):
        running_poc = _get_running_poc_context(workspace)
        if running_poc:
            if force:
                _ensure_poc_stopped(workspace, project_config=running_poc[0], service_config=running_poc[1])
            else:
                raise CLIException("system is still running, please stop the system first.")

        if not force:
            if not prompt_yn(
                f"This will delete poc workspace directory: '{workspace}' and create a new one. Is it OK to proceed?"
            ):
                return False

        workspace_path = Path(workspace)
        project_file = Path(project_conf_path)
        if project_conf_path and workspace_path in project_file.parents:
            raise CLIException(
                f"\nProject file: '{project_conf_path}' is under workspace directory:"
                f"'{workspace}', which is to be deleted. "
                f"Please copy {project_conf_path} to different location before running this command."
            )

        shutil.rmtree(workspace, ignore_errors=True)

    project_config = prepare_poc_provision(
        clients, number_of_clients, workspace, docker_image, use_he, project_conf_path, examples_dir
    )

    project_name = project_config.get("name") if project_config else None
    save_startup_kit_dir_config(workspace, project_name)
    return True


def _get_running_poc_context(workspace: str):
    try:
        project_config, service_config = setup_service_config(workspace)
    except Exception:
        # Best-effort detection only: unreadable workspace metadata should not block force recreation.
        return None

    if not project_config or not service_config:
        return None

    if not is_poc_ready(workspace, service_config, project_config):
        return None

    if not is_poc_running(workspace, service_config, project_config):
        return None

    return project_config, service_config


def _ensure_poc_stopped(
    workspace: str,
    timeout_in_sec: int = 30,
    poll_interval: float = 1.0,
    project_config=None,
    service_config=None,
    reason: str = "recreating the workspace",
):
    if project_config is None or service_config is None:
        running_poc = _get_running_poc_context(workspace)
        if not running_poc:
            return
        project_config, service_config = running_poc

    from nvflare.tool.cli_output import print_human

    print_human(f"Existing POC system is still running; stopping it before {reason}.")
    _stop_poc(workspace, project_config=project_config, service_config=service_config)

    deadline = time.time() + timeout_in_sec
    while time.time() < deadline:
        if not is_poc_running(workspace, service_config, project_config):
            return
        time.sleep(poll_interval)

    raise CLIException("system is still running after shutdown was requested; please run 'nvflare poc stop' first.")


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
    _prepare_poc_docker_deployments(workspace, project_config)
    # update storage
    if workspace != DEFAULT_WORKSPACE:
        prod_dir = get_prod_dir(workspace, project_name)
        update_storage_locations(local_dir=f"{prod_dir}/{server_name}/local", workspace=workspace)
    examples_dir = get_examples_dir(examples_dir)
    if examples_dir is not None:
        _link_jobs_dir_to_admin_transfer(examples_dir, workspace, None)

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
    if isinstance(user_input_gpu_ids, int) and user_input_gpu_ids == -1:
        gpu_ids = host_gpu_ids
    else:
        gpu_ids = user_input_gpu_ids
        validate_gpu_ids(gpu_ids, host_gpu_ids)
    return gpu_ids


def _get_poc_start_ready_timeout(cmd_args) -> int:
    ready_timeout = getattr(cmd_args, "timeout", POC_START_READY_TIMEOUT)
    if isinstance(ready_timeout, bool) or not isinstance(ready_timeout, int):
        ready_timeout = POC_START_READY_TIMEOUT
    if ready_timeout <= 0:
        raise CLIException("--timeout must be greater than 0 seconds")
    return ready_timeout


def start_poc(cmd_args):
    from nvflare.tool.cli_output import is_json_mode, output_error, output_error_message, output_ok
    from nvflare.tool.cli_schema import handle_schema_flag

    json_mode = is_json_mode()
    handle_schema_flag(
        _poc_sub_cmd_parsers.get(CMD_START_POC),
        "nvflare poc start",
        ["nvflare poc start", "nvflare poc start -p server"],
        sys.argv[1:],
        output_modes=["json"],
        streaming=False,
        mutating=True,
    )
    poc_workspace = get_poc_workspace()

    services_list = get_service_list(cmd_args)
    excluded = get_excluded(cmd_args)
    gpu_ids = get_gpis(cmd_args)
    study = getattr(cmd_args, "study", None)
    no_wait = getattr(cmd_args, "no_wait", False)
    no_wait = no_wait if isinstance(no_wait, bool) else False
    try:
        ready_timeout = _get_poc_start_ready_timeout(cmd_args)
    except CLIException as e:
        output_error("INVALID_ARGS", exit_code=4, detail=str(e))
        raise SystemExit(4)

    port_preflight = {
        "checked": False,
        "host": POC_PORT_PREFLIGHT_HOST,
        "scope": POC_PORT_PREFLIGHT_SCOPE,
        "ports": [],
        "conflicts": [],
        "message": "server port configuration is not available",
        "note": POC_PORT_PREFLIGHT_NOTE,
    }
    try:
        pre_project_config, pre_service_config = setup_service_config(poc_workspace)
        port_preflight = _build_poc_start_port_preflight(
            pre_project_config, pre_service_config, services_list, excluded
        )
    except Exception:
        pass

    try:
        with _quiet_cli_streams(json_mode):
            _start_poc(poc_workspace, gpu_ids, excluded, services_list, study=study)
    except CLIException as e:
        error_data = _build_poc_port_diagnostics(port_preflight) if json_mode else None
        output_error("INVALID_ARGS", exit_code=4, detail=str(e), data=error_data)
        raise SystemExit(4)
    except Exception as e:
        error_data = _build_poc_port_diagnostics(port_preflight) if json_mode else None
        output_error("INTERNAL_ERROR", exit_code=5, detail=str(e), data=error_data)
        raise SystemExit(5)

    # Get client names from project config
    clients = []
    project_config = None
    service_config = None
    endpoint_info = None
    try:
        project_config, service_config = setup_service_config(poc_workspace)
        if project_config:
            participants = project_config.get("participants", [])
            if not isinstance(participants, list):
                raise CLIException("project.yml participants must be a list")
            clients = []
            for p in participants:
                if not isinstance(p, dict):
                    raise CLIException("participant entry must be a mapping")
                if p.get("type") == "client":
                    name = p.get("name")
                    if name:
                        clients.append(name)
            endpoint_info = _build_poc_endpoint_info(project_config, service_config)
    except (OSError, IOError, yaml.YAMLError):
        pass
    except CLIException as e:
        output_error("INVALID_ARGS", exit_code=4, detail=str(e))
        raise SystemExit(4)
    if endpoint_info is None:
        endpoint_info = _build_poc_endpoint_info(project_config, service_config)

    ready = False
    wait_performed = False
    if not no_wait and service_config:
        try:
            with _quiet_cli_streams(json_mode):
                wait_performed = _wait_for_poc_system_ready(
                    poc_workspace,
                    project_config,
                    service_config,
                    services_list,
                    excluded,
                    timeout_in_sec=ready_timeout,
                )
            ready = wait_performed
        except SystemStartTimeout as e:
            output_error_message(
                "CONNECTION_FAILED",
                message="POC system did not become ready before the startup timeout.",
                hint=(
                    "Check the POC server/client logs, or use 'nvflare poc start --no-wait' "
                    "for fire-and-forget startup."
                ),
                exit_code=2,
                detail=str(e),
            )
            raise SystemExit(2)

    port_diagnostics = _build_poc_port_diagnostics(port_preflight)
    server_url = endpoint_info["server_url"]
    result = {
        "status": "starting" if no_wait else "running",
        "server_url": server_url,
        "server_address": endpoint_info["server_address"],
        "admin_address": endpoint_info["admin_address"],
        "default_port": endpoint_info["default_port"],
        "default_server_port": endpoint_info["default_server_port"],
        "default_admin_port": endpoint_info["default_admin_port"],
        "wait": not no_wait,
        "no_wait": no_wait,
        "ready_timeout": ready_timeout,
        "ready": ready,
        "clients": clients,
        "console_logs": _build_poc_console_logs(poc_workspace, project_config, service_config, services_list, excluded),
    }
    result.update(port_diagnostics)
    if json_mode:
        output_ok(result)
        return
    from nvflare.tool.cli_output import print_human

    if no_wait:
        print_human(f"\nPOC system start requested. Server: {server_url}")
    else:
        print_human(f"\nPOC system started. Server: {server_url}")
    print_human(f"  Server address: {endpoint_info['server_address']}")
    print_human(f"  Admin address: {endpoint_info['admin_address']}")
    if clients:
        print_human(f"  Clients: {', '.join(clients)}")
    console_logs = result.get("console_logs") or {}
    if console_logs:
        project_name = project_config.get("name") if project_config else DEFAULT_PROJECT_NAME
        print_human(
            f"  Service console logs: {get_prod_dir(poc_workspace, project_name)}/<participant>/{POC_SERVICE_CONSOLE_LOG}"
        )
    for warning in port_diagnostics["warnings"]:
        print_human(f"  Warning: {warning}")
    if service_config:
        proj_admin = service_config.get(SC.FLARE_PROJ_ADMIN, SC.FLARE_PROJ_ADMIN)
        print_human("  Admin console is optional. You can submit and manage jobs with nvflare commands.")
        print_human(f"  To open the console, run in another terminal: nvflare poc start -p {proj_admin}")
    if no_wait:
        print_human("  Readiness was not checked. Before submitting jobs, check status with: nvflare system status")
        print_human("  After ready, submit jobs with: nvflare job submit -j <job_folder>")
    else:
        print_human("  Submit jobs with: nvflare job submit -j <job_folder>")


def _get_started_readiness_participants(service_config: Dict, services_list: List[str], excluded: List[str]):
    excluded_set = set(excluded or [])
    server_name = service_config.get(SC.FLARE_SERVER)
    clients = list(service_config.get(SC.FLARE_CLIENTS, []))
    if services_list:
        started = [service for service in services_list if service not in excluded_set]
    else:
        started = [service for service in [server_name] + clients if service and service not in excluded_set]
    return server_name in started, [client for client in clients if client in started]


def _wait_for_poc_system_ready(
    poc_workspace: str,
    project_config: Dict,
    service_config: Dict,
    services_list: List[str],
    excluded: List[str],
    timeout_in_sec: int = POC_START_READY_TIMEOUT,
) -> bool:
    starts_server, expected_clients = _get_started_readiness_participants(service_config, services_list, excluded)
    if not starts_server and not expected_clients:
        return False

    project_name = project_config.get("name") if project_config else DEFAULT_PROJECT_NAME
    prod_dir = get_prod_dir(poc_workspace, project_name)
    try:
        wait_for_system_start(
            len(expected_clients),
            prod_dir,
            username=service_config[SC.FLARE_PROJ_ADMIN],
            secure_mode=True,
            second_to_wait=0,
            timeout_in_sec=timeout_in_sec,
            poll_interval=1.0,
            conn_timeout=1.0,
            expected_clients=expected_clients,
        )
    except SystemStartTimeout:
        raise
    except Exception as e:
        raise SystemStartTimeout(str(e)) from e
    return True


def get_gpis(cmd_args):
    if isinstance(cmd_args.gpu, list) and cmd_args.gpu:
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


def _get_server_url(project_config, service_config) -> str:
    return _build_poc_endpoint_info(project_config, service_config)["server_url"]


def _start_poc(poc_workspace: str, gpu_ids: List[int], excluded=None, services_list=None, study: Optional[str] = None):
    project_config, service_config = setup_service_config(poc_workspace)
    if services_list is None:
        services_list = []
    if excluded is None:
        excluded = []
    proj_admin_dir_name = service_config.get(SC.FLARE_PROJ_ADMIN, SC.FLARE_PROJ_ADMIN)
    admin_dirs = list(service_config.get(SC.FLARE_OTHER_ADMINS, []))
    admin_dirs.append(proj_admin_dir_name)

    # By default, do not start admin console services unless explicitly requested.
    if not services_list:
        for admin_dir in admin_dirs:
            if admin_dir not in excluded:
                excluded.append(admin_dir)
    else:
        for admin_dir in admin_dirs:
            if admin_dir not in services_list and admin_dir not in excluded:
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
        study=study,
    )


def validate_services(project_config, services_list: List, excluded: List):
    participant_names = [p["name"] for p in project_config["participants"]]
    validate_participants(participant_names, services_list)
    validate_participants(participant_names, excluded)


def validate_participants(participant_names, list_participants):
    for p in list_participants:
        if p not in participant_names:
            raise CLIException(f"participant '{p}' is not defined, expecting one of: {participant_names}")


def setup_service_config(poc_workspace) -> Tuple:
    project_file = os.path.join(poc_workspace, "project.yml")
    if os.path.isfile(project_file):
        project_config = load_yaml(project_file)
        service_config = get_service_config(project_config) if project_config else None
        return project_config, service_config
    else:
        raise CLIException(f"{project_file} is missing, make sure you have first run 'nvflare poc prepare'")


def stop_poc(cmd_args):
    from nvflare.tool.cli_output import is_json_mode, output_error, output_error_message, output_ok, print_human
    from nvflare.tool.cli_schema import handle_schema_flag

    json_mode = is_json_mode()
    handle_schema_flag(
        _poc_sub_cmd_parsers.get(CMD_STOP_POC),
        "nvflare poc stop",
        ["nvflare poc stop", "nvflare poc stop -p server"],
        sys.argv[1:],
    )
    poc_workspace = get_poc_workspace()
    excluded = get_excluded(cmd_args)
    services_list = get_service_list(cmd_args)
    no_wait = getattr(cmd_args, "no_wait", False)
    no_wait = no_wait if isinstance(no_wait, bool) else False

    try:
        if not json_mode:
            if no_wait:
                print_human("Requesting POC shutdown without waiting...")
            else:
                print_human("Stopping POC system; waiting for shutdown to complete...")
        with _quiet_cli_streams(json_mode):
            stop_result = _stop_poc(poc_workspace, excluded, services_list, wait=not no_wait)
    except CLIException as e:
        output_error("INVALID_ARGS", exit_code=4, detail=str(e))
        raise SystemExit(4)
    except TimeoutError as e:
        output_error_message(
            "CONNECTION_FAILED",
            message="POC system shutdown did not complete before the timeout.",
            hint="Check the POC server/client logs, or use 'nvflare poc stop --no-wait' for fire-and-forget shutdown.",
            exit_code=2,
            detail=str(e),
        )
        raise SystemExit(2)
    except Exception as e:
        output_error("INTERNAL_ERROR", exit_code=5, detail=str(e))
        raise SystemExit(5)

    result = {"status": "shutdown_initiated" if no_wait else "stopped", "wait": not no_wait, "no_wait": no_wait}
    if stop_result:
        result.update(stop_result)
    if json_mode:
        output_ok(result)
        return

    active_job_ids = (stop_result or {}).get("active_job_ids") or []
    if active_job_ids:
        print_human(f"Active jobs aborted before shutdown: {', '.join(active_job_ids)}")
    if (stop_result or {}).get("already_stopped"):
        print_human("POC system was already stopped.")
        return
    if no_wait:
        print_human("POC system shutdown requested.")
    else:
        print_human("POC system stopped.")


def _stop_poc(
    poc_workspace: str,
    excluded=None,
    services_list=None,
    project_config=None,
    service_config=None,
    wait: bool = True,
):
    if project_config is None or service_config is None:
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
        from nvflare.tool.cli_output import print_human

        with _quiet_cli_streams(True):
            print_human("Starting shutdown of NVFLARE")
            shutdown_result = (
                shutdown_system(prod_dir, username=service_config[SC.FLARE_PROJ_ADMIN], wait=wait, verbose=False) or {}
            )
        return {"shutdown": shutdown_result, **shutdown_result}
    else:
        from nvflare.tool.cli_output import print_human

        print_human(f"Starting shutdown of {services_list} using the stop_fl.sh script")

        _run_poc(
            SC.CMD_STOP,
            poc_workspace,
            gpu_ids,
            service_config,
            project_config,
            excluded=excluded,
            services_list=services_list,
        )
        return {"services": services_list}


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
    cmd_type: str,
    poc_workspace: str,
    service_config,
    project_config,
    excluded: list,
    services_list=None,
    study: Optional[str] = None,
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
                        cmd = get_service_command(cmd_type, prod_dir, service_dir_name, service_config, study=study)
                        if cmd:
                            service_commands.append((service_dir_name, cmd))
    return _sort_service_cmds(cmd_type, service_commands, service_config)


def prepare_env(service_name, gpu_ids: Optional[List[int]], service_config: Dict):
    my_env = None
    if gpu_ids:
        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = ",".join([str(gid) for gid in gpu_ids])

    if service_config.get(SC.IS_DOCKER_RUN):
        my_env = os.environ.copy() if my_env is None else my_env
        if gpu_ids:
            my_env["GPU2USE"] = f'--gpus="device={my_env["CUDA_VISIBLE_DEVICES"]}"'

        my_env["MY_DATA_DIR"] = os.path.join(get_poc_workspace(), "data")
        my_env["SVR_NAME"] = service_name

    return my_env


def _poc_service_console_log(cmd_path: str) -> str:
    cmd = cmd_path.split(" ")[0]
    startup_dir = os.path.dirname(os.path.abspath(cmd))
    service_dir = os.path.dirname(startup_dir)
    return os.path.join(service_dir, POC_SERVICE_CONSOLE_LOG)


def _build_poc_console_logs(
    poc_workspace: str, project_config: Dict, service_config: Dict, services_list, excluded
) -> Dict:
    if not project_config or not service_config:
        return {}

    logs = {}
    project_name = project_config.get("name")
    prod_dir = get_prod_dir(poc_workspace, project_name)
    started_server, started_clients = _get_started_readiness_participants(service_config, services_list, excluded)
    if started_server:
        server_name = service_config.get(SC.FLARE_SERVER)
        if server_name:
            logs[server_name] = os.path.join(prod_dir, server_name, POC_SERVICE_CONSOLE_LOG)
    for client_name in started_clients:
        logs[client_name] = os.path.join(prod_dir, client_name, POC_SERVICE_CONSOLE_LOG)
    return logs


def async_process(service_name, cmd_path, gpu_ids: Optional[List[int]], service_config: Dict):
    my_env = prepare_env(service_name, gpu_ids, service_config)
    console_log = _poc_service_console_log(cmd_path)
    os.makedirs(os.path.dirname(console_log), exist_ok=True)
    with open(console_log, "a", buffering=1) as output:
        output.write(f"\n--- nvflare poc start {time.asctime()} ---\n")
        if my_env:
            subprocess.Popen(cmd_path.split(" "), env=my_env, stdout=output, stderr=subprocess.STDOUT)
        else:
            subprocess.Popen(cmd_path.split(" "), stdout=output, stderr=subprocess.STDOUT)


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
    study: Optional[str] = None,
):
    if services_list is None:
        services_list = []
    service_commands = _build_commands(
        cmd_type, poc_workspace, service_config, project_config, excluded, services_list, study=study
    )
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
            client_gpu_ids = gpu_assignments[service_name] if service_name in clients else None
            async_process(service_name, cmd_path, client_gpu_ids, service_config)


def clean_poc(cmd_args):
    from nvflare.tool.cli_output import is_json_mode, output_error, output_ok
    from nvflare.tool.cli_schema import handle_schema_flag

    json_mode = is_json_mode()
    handle_schema_flag(
        _poc_sub_cmd_parsers.get(CMD_CLEAN_POC),
        "nvflare poc clean",
        ["nvflare poc clean", "nvflare poc clean --force"],
        sys.argv[1:],
    )
    poc_workspace = get_poc_workspace()
    try:
        with _quiet_cli_streams(json_mode):
            _clean_poc(poc_workspace, force=getattr(cmd_args, "force", False))
    except CLIException as e:
        output_error("INVALID_ARGS", exit_code=4, detail=str(e))
        raise SystemExit(4)
    except Exception as e:
        output_error("INTERNAL_ERROR", exit_code=5, detail=str(e))
        raise SystemExit(5)
    if json_mode:
        output_ok({"workspace": poc_workspace, "status": "removed"})


def _clean_poc_config(poc_workspace: str):
    config = load_cli_config()
    config, _removed_ids = remove_entries_under_workspace(config, poc_workspace)
    for key in (f"{POC_KEY}.{WORKSPACE_KEY}", f"{POC_KEY}.{STARTUP_KIT_KEY}", "prod.startup_kit"):
        try:
            config.pop(key, None)
        except Exception:
            pass
    save_cli_config(config)


def is_poc_running(poc_workspace, service_config, project_config):
    project_name = project_config.get("name") if project_config else DEFAULT_PROJECT_NAME
    prod_dirs = _get_prod_dirs(poc_workspace, project_name)
    if not prod_dirs:
        prod_dirs = [get_prod_dir(poc_workspace, project_name)]
    for prod_dir in prod_dirs:
        server_dir = os.path.join(prod_dir, service_config[SC.FLARE_SERVER])
        pid_file = os.path.join(server_dir, "pid.fl")
        daemon_pid_file = os.path.join(server_dir, "daemon_pid.fl")
        if _is_live_pid_file(pid_file) or _is_live_pid_file(daemon_pid_file):
            return True
    return False


def _is_live_pid_file(pid_file: str) -> bool:
    if not os.path.exists(pid_file):
        return False

    try:
        with open(pid_file, "r") as f:
            pid = int(f.read().strip())
    except (OSError, ValueError):
        return False

    try:
        os.kill(pid, 0)
        return True
    except OSError as e:
        if e.errno == errno.EPERM:
            return True
        return False


def _clean_poc(poc_workspace: str, force: bool = False):
    if os.path.isdir(poc_workspace):
        project_config, service_config = setup_service_config(poc_workspace)
        if project_config is None:
            raise CLIException(f"{poc_workspace} is not valid poc directory")
        if is_poc_ready(poc_workspace, service_config, project_config):
            if is_poc_running(poc_workspace, service_config, project_config):
                if force:
                    _ensure_poc_stopped(
                        poc_workspace,
                        project_config=project_config,
                        service_config=service_config,
                        reason="cleaning the workspace",
                    )
                else:
                    raise CLIException("system is still running, please stop the system first.")

            if is_poc_running(poc_workspace, service_config, project_config):
                raise CLIException("system is still running, please stop the system first.")

            from nvflare.tool.cli_output import print_human

            shutil.rmtree(poc_workspace)
            _clean_poc_config(poc_workspace)

            print_human(f"{poc_workspace} is removed")
            return True
        else:
            raise CLIException(f"{poc_workspace} is not valid poc directory")
    else:
        raise CLIException(f"{poc_workspace} is not valid poc directory")


def config_poc(cmd_args):
    from nvflare.tool.cli_output import is_json_mode, output_error, output_ok, print_human
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        _poc_sub_cmd_parsers.get(CMD_CONFIG_POC),
        "nvflare poc config",
        ["nvflare poc config", "nvflare poc config --pw /path/to/poc_workspace"],
        sys.argv[1:],
        output_modes=["json"],
        streaming=False,
        mutating=True,
        idempotent=True,
        retry_token={"supported": False},
    )

    requested_workspace = getattr(cmd_args, "poc_workspace_dir", None)
    config_file_path, loaded_config, _migration_needed = load_hidden_config_state()
    nvflare_config = loaded_config or CF.parse_string("{}")

    if requested_workspace is None:
        data = {
            "config_file": config_file_path,
            "poc_workspace_dir": get_poc_workspace(),
            "env_override": os.getenv("NVFLARE_POC_WORKSPACE"),
        }
        if is_json_mode():
            output_ok(data)
        else:
            print_human(f"POC workspace: {data['poc_workspace_dir']}")
            if data["env_override"]:
                print_human(f"  Environment override: NVFLARE_POC_WORKSPACE={data['env_override']}")
            print_human(f"  Config file: {config_file_path}")
        return

    try:
        nvflare_config = create_poc_workspace_config(nvflare_config, requested_workspace)
        nvflare_config.put(CONFIG_VERSION, CURRENT_CONFIG_VERSION)
        save_config(nvflare_config, config_file_path)
    except ValueError as e:
        output_error("INVALID_ARGS", exit_code=4, detail=str(e))
        raise SystemExit(4)

    poc_workspace_dir = nvflare_config.get(f"{POC_KEY}.{WORKSPACE_KEY}", None)
    data = {
        "config_file": config_file_path,
        "poc_workspace_dir": poc_workspace_dir,
    }
    if is_json_mode():
        output_ok(data)
        return
    print_human(f"POC workspace configured: {poc_workspace_dir}")


poc_sub_cmd_handlers = {
    CMD_PREPARE_POC: prepare_poc,
    CMD_ADD_USER: add_poc_user,
    CMD_ADD_SITE: add_poc_site,
    CMD_START_POC: start_poc,
    CMD_STOP_POC: stop_poc,
    CMD_CLEAN_POC: clean_poc,
    CMD_CONFIG_POC: config_poc,
}

# Populated by define_*_parser functions; used by handlers for --schema support
_poc_sub_cmd_parsers = {}
_poc_root_parser = None


def def_poc_parser(sub_cmd):
    global _poc_root_parser
    cmd = "poc"
    parser = sub_cmd.add_parser(cmd, help="manage a local proof-of-concept FL system")
    _poc_root_parser = parser

    poc_parser = parser.add_subparsers(title=cmd, dest="poc_sub_cmd", help="poc subcommand")
    define_config_parser(poc_parser)
    define_prepare_parser(poc_parser)
    define_add_user_parser(poc_parser)
    define_add_site_parser(poc_parser)
    define_start_parser(poc_parser)
    define_stop_parser(poc_parser)
    define_clean_parser(poc_parser)
    return {cmd: parser}


def define_config_parser(poc_parser):
    config_parser = poc_parser.add_parser(CMD_CONFIG_POC, help="configure local POC settings")
    _poc_sub_cmd_parsers[CMD_CONFIG_POC] = config_parser
    config_parser.add_argument(
        "-pw",
        "--pw",
        "--poc_workspace_dir",
        "--poc-workspace-dir",
        dest="poc_workspace_dir",
        type=str,
        nargs="?",
        default=None,
        help="POC workspace location",
    )
    config_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")


def define_prepare_parser(poc_parser, cmd: Optional[str] = None, help_str: Optional[str] = None):
    cmd = CMD_PREPARE_POC if cmd is None else cmd
    help_str = "prepare poc environment by provisioning local project" if help_str is None else help_str
    prepare_parser = poc_parser.add_parser(cmd, help=help_str)
    _poc_sub_cmd_parsers[CMD_PREPARE_POC] = prepare_parser

    prepare_parser.add_argument(
        "-n",
        "--number-of-clients",
        "--number_of_clients",  # backward compat
        dest="number_of_clients",
        type=int,
        nargs="?",
        default=2,
        help="number of sites or clients, default to 2",
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
        "--project-input",
        "--project_input",  # backward compat
        dest="project_input",
        type=str,
        nargs="?",
        default="",
        help="project.yaml file path, If specified, "
        + "'number_of_clients','clients' and 'docker' specific options will be ignored.",
    )
    prepare_parser.add_argument(
        "-d",
        "--docker-image",
        "--docker_image",  # backward compat
        dest="docker_image",
        nargs="?",
        default=None,
        const="nvflare/nvflare",
        help="prepare Docker runtime start_docker.sh using this SP/CP Docker image. "
        + "Jobs must specify a Docker job image in launcher_spec.",
    )

    prepare_parser.add_argument("-debug", "--debug", action="store_true", help="debug is on")
    prepare_parser.add_argument("--force", action="store_true", help="overwrite existing workspace without prompting")
    prepare_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")


def define_add_user_parser(poc_parser):
    user_parser = poc_parser.add_parser(CMD_ADD_USER, help="add a POC user startup kit")
    _poc_sub_cmd_parsers[CMD_ADD_USER] = user_parser
    user_parser.add_argument("cert_role", choices=POC_USER_CERT_ROLES, help="certificate role for the user")
    user_parser.add_argument("email", help="user identity, usually an email address")
    user_parser.add_argument("--org", default="nvidia", help="organization name, default nvidia")
    user_parser.add_argument("--force", action="store_true", help="replace an existing user participant")
    user_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")


def define_add_site_parser(poc_parser):
    site_parser = poc_parser.add_parser(CMD_ADD_SITE, help="add a POC site startup kit")
    _poc_sub_cmd_parsers[CMD_ADD_SITE] = site_parser
    site_parser.add_argument("name", help="site name")
    site_parser.add_argument("--org", default="nvidia", help="organization name, default nvidia")
    site_parser.add_argument("--force", action="store_true", help="replace an existing site participant")
    site_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")


def define_clean_parser(poc_parser):
    clean_parser = poc_parser.add_parser(CMD_CLEAN_POC, help="clean up poc workspace")
    _poc_sub_cmd_parsers[CMD_CLEAN_POC] = clean_parser
    clean_parser.add_argument("-debug", "--debug", action="store_true", help="debug is on")
    clean_parser.add_argument("--force", action="store_true", help="stop a running POC system before cleanup")
    clean_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")


def define_start_parser(poc_parser):
    start_parser = poc_parser.add_parser(CMD_START_POC, help="start services in poc mode")
    _poc_sub_cmd_parsers[CMD_START_POC] = start_parser

    start_parser.add_argument(
        "-p",
        "--service",
        type=str,
        nargs="?",
        default="all",
        help="participant to start. Default starts server and client services only; admin consoles are excluded unless explicitly selected",
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
    start_parser.add_argument(
        "--study",
        type=str,
        default=None,
        help="study for admin console launches only; ignored for server and client services",
    )
    start_parser.add_argument(
        "--no-wait",
        dest="no_wait",
        action="store_true",
        help="return after starting processes without waiting for admin server/client readiness",
    )
    start_parser.add_argument(
        "--timeout",
        type=int,
        default=POC_START_READY_TIMEOUT,
        help=f"seconds to wait for admin server/client readiness, default {POC_START_READY_TIMEOUT}",
    )
    start_parser.add_argument("-debug", "--debug", action="store_true", help="debug is on")
    start_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")


def define_stop_parser(poc_parser):
    stop_parser = poc_parser.add_parser(CMD_STOP_POC, help="stop services in poc mode")
    _poc_sub_cmd_parsers[CMD_STOP_POC] = stop_parser

    stop_parser.add_argument(
        "-p",
        "--service",
        type=str,
        nargs="?",
        default="all",
        help="participant to stop. Default stops the running POC system; project admin console is not a default managed service",
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
    stop_parser.add_argument(
        "--no-wait",
        dest="no_wait",
        action="store_true",
        help="return after requesting shutdown without waiting for completion",
    )
    stop_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")


def get_local_host_gpu_ids():
    try:
        return get_host_gpu_ids()
    except Exception as e:
        raise CLIException(f"Failed to get host gpu ids:{e}")


def handle_poc_cmd(cmd_args):
    poc_sub_cmd = getattr(cmd_args, "poc_sub_cmd", None)
    if poc_sub_cmd:
        poc_cmd_handler = poc_sub_cmd_handlers.get(poc_sub_cmd, None)
        if poc_cmd_handler is None:
            from nvflare.tool.cli_output import output_usage_error

            output_usage_error(_poc_root_parser, "unknown poc command", exit_code=4)
            raise SystemExit(4)
        poc_cmd_handler(cmd_args)
        return

    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        _poc_root_parser,
        "nvflare poc",
        [
            "nvflare poc config --schema",
            "nvflare poc prepare --schema",
            "nvflare poc start --schema",
            "nvflare poc stop --schema",
        ],
        getattr(cmd_args, "_argv", sys.argv[1:]),
    )
    raise CLIUnknownCmdException("unknown command")


def get_poc_workspace():
    poc_workspace = os.getenv("NVFLARE_POC_WORKSPACE")

    if not poc_workspace:
        src_path = get_hidden_nvflare_config_path(str(get_or_create_hidden_nvflare_dir()))
        if os.path.isfile(src_path):
            from pyhocon import ConfigFactory as CF

            config = CF.parse_file(src_path)
            poc_workspace = config.get("poc.workspace", None)
            if not poc_workspace:
                poc_workspace = config.get("poc_workspace.path", None)

    if poc_workspace is None or len(poc_workspace.strip()) == 0:
        poc_workspace = DEFAULT_WORKSPACE

    return poc_workspace
