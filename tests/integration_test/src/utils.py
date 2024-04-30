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
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from typing import List

import yaml

from nvflare.apis.job_def import RunStatus
from nvflare.fuel.hci.client.api_status import APIStatus
from nvflare.fuel.hci.client.fl_admin_api import FLAdminAPI
from nvflare.fuel.hci.client.fl_admin_api_constants import FLDetailKey
from nvflare.fuel.hci.client.fl_admin_api_spec import TargetType
from nvflare.fuel.utils.class_utils import instantiate_class

from .constants import DEFAULT_RESOURCE_CONFIG, FILE_STORAGE, PROVISION_SCRIPT, RESOURCE_CONFIG
from .example import Example

OUTPUT_YAML_DIR = os.path.join("data", "test_configs", "generated")
PROJECT_YAML = os.path.join("data", "projects", "ha_1_servers_2_clients.yml")
POSTFIX = "_copy"
REQUIREMENTS_TO_EXCLUDE = ["nvflare", "jupyter", "notebook"]


def read_yaml(yaml_file_path):
    if not os.path.exists(yaml_file_path):
        raise RuntimeError(f"Yaml file doesnt' exist at {yaml_file_path}")

    with open(yaml_file_path, "rb") as f:
        data = yaml.safe_load(f)

    return data


def cleanup_path(path: str):
    if os.path.exists(path):
        print(f"Clean up directory: {path}")
        shutil.rmtree(path)


def run_provision_command(project_yaml: str, workspace: str):
    command = f"{sys.executable} -m {PROVISION_SCRIPT} -p {project_yaml} -w {workspace}"
    process = run_command_in_subprocess(command)
    process.wait()


def run_command_in_subprocess(command):
    new_env = os.environ.copy()
    python_path = ":".join(sys.path)[1:]  # strip leading colon
    new_env["PYTHONPATH"] = python_path
    process = subprocess.Popen(
        shlex.split(command),
        preexec_fn=os.setsid,
        env=new_env,
    )
    return process


def _get_resource_json_file(workspace_path: str, site_name: str) -> str:
    resource_json_path = os.path.join(workspace_path, site_name, "local", RESOURCE_CONFIG)
    if not os.path.exists(resource_json_path):
        default_json_path = os.path.join(workspace_path, site_name, "local", DEFAULT_RESOURCE_CONFIG)
        if not os.path.exists(default_json_path):
            raise RuntimeError(f"Missing {RESOURCE_CONFIG} at: {resource_json_path}")
        resource_json_path = default_json_path
    return resource_json_path


def _check_snapshot_persistor_in_resource(resource_json: dict):
    if "snapshot_persistor" not in resource_json:
        raise RuntimeError(f"Missing snapshot_persistor in {RESOURCE_CONFIG}")
    if "args" not in resource_json["snapshot_persistor"]:
        raise RuntimeError("Missing args in snapshot_persistor")
    if "storage" not in resource_json["snapshot_persistor"]["args"]:
        raise RuntimeError("Missing storage in snapshot_persistor's args")
    if "args" not in resource_json["snapshot_persistor"]["args"]["storage"]:
        raise RuntimeError("Missing args in snapshot_persistor's storage")
    if "path" not in resource_json["snapshot_persistor"]["args"]["storage"]:
        raise RuntimeError("Missing path in snapshot_persistor's storage")
    if resource_json["snapshot_persistor"]["args"]["storage"]["path"] != FILE_STORAGE:
        raise RuntimeError(f"Only support {FILE_STORAGE} storage in snapshot_persistor's args")
    if "root_dir" not in resource_json["snapshot_persistor"]["args"]["storage"]["args"]:
        raise RuntimeError("Missing root_dir in snapshot_persistor's storage's args")
    return True


def _get_snapshot_path_from_workspace(path: str, server_name: str) -> str:
    resource_json_path = _get_resource_json_file(path, server_name)
    with open(resource_json_path, "r") as f:
        resource_json = json.load(f)
    _check_snapshot_persistor_in_resource(resource_json)
    return resource_json["snapshot_persistor"]["args"]["storage"]["args"]["root_dir"]


def update_snapshot_path_in_workspace(path: str, server_name: str, snapshot_path: str = None):
    new_snapshot_path = snapshot_path if snapshot_path else tempfile.mkdtemp()
    resource_json_path = _get_resource_json_file(workspace_path=path, site_name=server_name)
    with open(resource_json_path, "r") as f:
        resource_json = json.load(f)
    _check_snapshot_persistor_in_resource(resource_json)
    resource_json["snapshot_persistor"]["args"]["storage"]["args"]["root_dir"] = new_snapshot_path
    with open(resource_json_path, "w") as f:
        json.dump(resource_json, f)
    return new_snapshot_path


def _check_job_store_in_resource(resource_json: dict):
    if "components" not in resource_json:
        raise RuntimeError(f"Missing components in {RESOURCE_CONFIG}")
    job_manager_config = None
    for c in resource_json["components"]:
        if "id" in c and c["id"] == "job_manager":
            job_manager_config = c
    if not job_manager_config:
        raise RuntimeError(f"Missing job_manager in {RESOURCE_CONFIG}")
    if "args" not in job_manager_config:
        raise RuntimeError("Missing args in job_manager.")
    if "uri_root" not in job_manager_config["args"]:
        raise RuntimeError("Missing uri_root in job_manager's args.")


def _get_job_store_path_from_workspace(path: str, server_name: str) -> str:
    resource_json_path = _get_resource_json_file(path, server_name)
    with open(resource_json_path, "r") as f:
        resource_json = json.load(f)
    _check_job_store_in_resource(resource_json)
    for c in resource_json["components"]:
        if "id" in c and c["id"] == "job_manager":
            return c["args"]["uri_root"]


def update_job_store_path_in_workspace(path: str, server_name: str, job_store_path: str = None):
    new_job_store_path = job_store_path if job_store_path else tempfile.mkdtemp()
    resource_json_path = _get_resource_json_file(workspace_path=path, site_name=server_name)
    with open(resource_json_path, "r") as f:
        resource_json = json.load(f)

    if "components" not in resource_json:
        raise RuntimeError(f"Missing components in {RESOURCE_CONFIG}")
    _check_job_store_in_resource(resource_json)
    for c in resource_json["components"]:
        if "id" in c and c["id"] == "job_manager":
            c["args"]["uri_root"] = new_job_store_path

    with open(resource_json_path, "w") as f:
        json.dump(resource_json, f)

    return new_job_store_path


def cleanup_job_and_snapshot(workspace: str, server_name: str):
    job_store_path = _get_job_store_path_from_workspace(workspace, server_name)
    snapshot_path = _get_snapshot_path_from_workspace(workspace, server_name)
    cleanup_path(job_store_path)
    cleanup_path(snapshot_path)


def get_job_meta(admin_api: FLAdminAPI, job_id: str) -> dict:
    response = admin_api.do_command(f"get_job_meta {job_id}")
    return response.get("meta", {}).get("job_meta", {})


def check_client_status_ready(response: dict) -> bool:
    if response["status"] != APIStatus.SUCCESS:
        return False

    if "details" not in response:
        return False

    data = response.get("raw", {}).get("data", [])
    if data:
        for d in data:
            if d.get("type") == "error":
                return False

    # check fuel/hci/client/fl_admin_api.py for parsing
    if "client_statuses" not in response["details"]:
        return False

    for row in response["details"]["client_statuses"][1:]:
        if row[3] == "No Reply":
            return False

    return True


def check_job_done(job_id: str, admin_api: FLAdminAPI):
    response = admin_api.check_status(target_type=TargetType.SERVER)
    if response and "status" in response:
        if response["status"] != APIStatus.SUCCESS:
            print(f"Check server status failed: {response}.")
            return False
        else:
            if "details" not in response:
                print(f"Check server status missing details: {response}.")
                return False
            else:
                # check if run is stopped
                if (
                    FLDetailKey.SERVER_ENGINE_STATUS in response["details"]
                    and response["details"][FLDetailKey.SERVER_ENGINE_STATUS] == "stopped"
                ):
                    response = admin_api.check_status(target_type=TargetType.CLIENT)
                    if not check_client_status_ready(response):
                        print(f"Check client status failed: {response}")
                        return False
                    else:
                        job_meta = get_job_meta(admin_api, job_id=job_id)
                        job_run_status = job_meta.get("status")

                        for row in response["details"]["client_statuses"]:
                            if row[3] != "stopped":
                                continue
                        # check if the current job is completed
                        if job_run_status in (
                            RunStatus.FINISHED_COMPLETED.value,
                            RunStatus.FINISHED_ABORTED.value,
                        ):
                            return True
    return False


def run_admin_api_tests(admin_api: FLAdminAPI):
    print(("\n" + "*" * 120) * 20)
    print("\n" + "=" * 40)
    print("\nRunning through tests of admin commands:")
    print("\n" + "=" * 40)
    print("\nActive SP:")
    print(admin_api.get_active_sp().get("details"))
    print("\nList SP:")
    print(admin_api.list_sp().get("details"))
    print("\nCommand: get_available_apps_to_upload")
    print(admin_api.get_available_apps_to_upload())
    print("\nList Jobs:")
    list_jobs_return_rows = admin_api.list_jobs().get("details")
    print(list_jobs_return_rows)
    first_job = str(list_jobs_return_rows[0].get("job_id"))
    print("\nCommand: ls server -a .")
    ls_return_message = admin_api.ls_target("server", "-a", ".").get("details").get("message")
    print(ls_return_message)
    print("\nAssert Job {} is in the server root dir...".format(first_job))
    assert first_job in ls_return_message

    print("\nAborting Job {}:".format(first_job))
    print("\n" + "=" * 50)
    print(admin_api.abort_job(first_job).get("details").get("message"))
    print("\n" + "=" * 50)
    print("\nCommand: pwd")
    print(admin_api.get_working_directory("server").get("details").get("message"))

    print("\n" + "=" * 50)
    print("Finished with admin commands testing through FLAdminAPI.")


def _replace_meta_json(meta_json_path: str):
    with open(meta_json_path, "r+") as f:
        job_meta = json.load(f)
        if "resource_spec" in job_meta:
            job_meta.pop("resource_spec")
        job_meta["min_clients"] = 2
        f.seek(0)
        json.dump(job_meta, f, indent=4)
        f.truncate()


def _replace_config_fed_server(server_json_path: str):
    with open(server_json_path, "r+") as f:
        config_fed_server = json.load(f)
        config_fed_server["num_rounds"] = 2
        config_fed_server["min_clients"] = 2
        config_fed_server["TRAIN_SPLIT_ROOT"] = "/tmp/nvflare/test_data"
        f.seek(0)
        json.dump(config_fed_server, f, indent=4)
        f.truncate()


def _replace_config_fed_client(client_json_path: str):
    with open(client_json_path, "r+") as f:
        config_fed_client = json.load(f)
        config_fed_client["TRAIN_SPLIT_ROOT"] = "/tmp/nvflare/test_data"
        config_fed_client["num_rounds"] = 2
        config_fed_client["AGGREGATION_EPOCHS"] = 1
        f.seek(0)
        json.dump(config_fed_client, f, indent=4)
        f.truncate()


def simplify_job(job_folder_path: str, postfix: str = POSTFIX):
    new_job_folder_path = job_folder_path + postfix
    shutil.copytree(job_folder_path, new_job_folder_path, dirs_exist_ok=True)

    # update meta.json
    _replace_meta_json(meta_json_path=os.path.join(new_job_folder_path, "meta.json"))

    for root, dirs, files in os.walk(new_job_folder_path):
        for file in files:
            if file == "config_fed_server.json":
                # set the num_rounds and TRAIN_SPLIT_ROOT in config_fed_server.json
                _replace_config_fed_server(server_json_path=os.path.join(root, file))
            elif file == "config_fed_client.json":
                # set TRAIN_SPLIT_ROOT in config_fed_client.json
                _replace_config_fed_client(client_json_path=os.path.join(root, file))


def generate_test_config_yaml_for_example(
    example: Example,
    project_yaml: str = PROJECT_YAML,
    job_postfix: str = POSTFIX,
) -> List[str]:
    """Generates test configurations for an NVFlare example folder.

    Args:
        example (Example): A well-formatted NVFlare example folder.
        project_yaml (str): Project yaml file for the testing of this example.
        job_postfix (str): Postfix for the newly generated job.
    """
    output_yamls = []
    os.makedirs(OUTPUT_YAML_DIR, exist_ok=True)
    for job in os.listdir(example.jobs_root_dir):
        output_yaml = _generate_test_config_for_one_job(example, job, project_yaml, job_postfix)
        output_yamls.append(output_yaml)
    return output_yamls


def _generate_test_config_for_one_job(
    example: Example,
    job: str,
    project_yaml: str = PROJECT_YAML,
    postfix: str = POSTFIX,
) -> str:
    """Generates test configuration yaml for an NVFlare example.

    Args:
        example (Example): A well-formatted NVFlare example.
        job (str): name of the job.
        project_yaml (str): Project yaml file for the testing of this example.
        postfix (str): Postfix for the newly generated job.
    """
    output_yaml = os.path.join(OUTPUT_YAML_DIR, f"{example.name}_{job}.yml")
    job_dir = os.path.join(example.jobs_root_dir, job)
    requirements_file = os.path.join(example.root, example.requirements_file)
    new_requirements_file = os.path.join(example.root, "temp_requirements.txt")
    exclude_requirements = "\\|".join(REQUIREMENTS_TO_EXCLUDE)

    setup = [
        f"cp {requirements_file} {new_requirements_file}",
        f"sed -i '/{exclude_requirements}/d' {new_requirements_file}",
        f"pip install -r {new_requirements_file}",
    ]
    if example.prepare_data_script is not None:
        setup.append(f"bash {example.prepare_data_script}")
    setup.append(f"python convert_to_test_job.py --job {job_dir} --post {postfix}")
    setup.append(f"rm -f {new_requirements_file}")

    config = {
        "ha": True,
        "jobs_root_dir": example.jobs_root_dir,
        "cleanup": True,
        "project_yaml": project_yaml,
        "additional_python_paths": example.additional_python_paths,
        "tests": [
            {
                "test_name": f"Test a simplified copy of job {job} for example {example.name}.",
                "event_sequence": [
                    {
                        "trigger": {"type": "server_log", "data": "Server started"},
                        "actions": [f"submit_job {job}{postfix}"],
                        "result": {"type": "job_submit_success"},
                    },
                    {
                        "trigger": {"type": "run_state", "data": {"run_finished": True}},
                        "actions": ["ensure_current_job_done"],
                        "result": {"type": "run_state", "data": {"run_finished": True}},
                    },
                ],
                "setup": setup,
                "teardown": [f"rm -rf {job_dir}{postfix}"],
            }
        ],
    }
    with open(output_yaml, "w") as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)
    return output_yaml


def _read_admin_json_file(admin_json_file) -> dict:
    if not os.path.exists(admin_json_file):
        raise RuntimeError("Missing admin json file.")
    with open(admin_json_file, "r") as f:
        admin_json = json.load(f)
    return admin_json


def create_admin_api(workspace_root_dir, upload_root_dir, download_root_dir, admin_user_name, poc):
    admin_startup_folder = os.path.join(workspace_root_dir, admin_user_name, "startup")
    admin_json_file = os.path.join(admin_startup_folder, "fed_admin.json")
    admin_json = _read_admin_json_file(admin_json_file)
    overseer_agent = instantiate_class(
        class_path=admin_json["admin"]["overseer_agent"]["path"],
        init_params=admin_json["admin"]["overseer_agent"]["args"],
    )

    ca_cert = os.path.join(admin_startup_folder, admin_json["admin"]["ca_cert"])
    client_key = os.path.join(admin_startup_folder, admin_json["admin"]["client_key"])
    client_cert = os.path.join(admin_startup_folder, admin_json["admin"]["client_cert"])

    admin_api = FLAdminAPI(
        upload_dir=upload_root_dir,
        download_dir=download_root_dir,
        overseer_agent=overseer_agent,
        insecure=poc,
        user_name=admin_user_name,
        ca_cert=ca_cert,
        client_key=client_key,
        client_cert=client_cert,
        auto_login_max_tries=20,
    )
    return admin_api


def ensure_admin_api_logged_in(admin_api: FLAdminAPI, timeout: int = 60):
    login_success = False
    try:
        start_time = time.time()
        while time.time() - start_time <= timeout:
            if admin_api.is_ready():
                login_success = True
                break
            time.sleep(0.2)

        if not login_success:
            print(f"Admin api failed to log in within {timeout} seconds: {admin_api.fsm.current_state}.")
        else:
            print("Admin successfully logged into server.")
    except Exception as e:
        print(f"Exception in logging in to admin: {e.__str__()}")
    return login_success
