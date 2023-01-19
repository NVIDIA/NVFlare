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
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Dict

import yaml

from nvflare.apis.job_def import RunStatus
from nvflare.fuel.hci.client.api_status import APIStatus
from nvflare.fuel.hci.client.fl_admin_api import FLAdminAPI
from nvflare.fuel.hci.client.fl_admin_api_constants import FLDetailKey
from nvflare.fuel.hci.client.fl_admin_api_spec import TargetType

from .constants import DEFAULT_RESOURCE_CONFIG, FILE_STORAGE, PROVISION_SCRIPT, RESOURCE_CONFIG


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


def _parse_job_run_statuses(list_jobs_string: str) -> Dict[str, RunStatus]:
    """Parse the list_jobs string to return job run status."""
    job_statuses = {}
    # first three is table start and header row, last one is table end
    rows = list_jobs_string.splitlines()[3:-1]
    for r in rows:
        segments = [s.strip() for s in r.split("|")]
        job_statuses[segments[1]] = RunStatus(segments[3])
    return job_statuses


def get_job_run_statuses(admin_api: FLAdminAPI):
    list_jobs_result = admin_api.list_jobs("-a")
    if list_jobs_result["status"] == APIStatus.SUCCESS:
        list_jobs_string = list_jobs_result["details"]["message"]
        job_run_statuses = _parse_job_run_statuses(list_jobs_string)
        return job_run_statuses
    return {}


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
                    if response["status"] != APIStatus.SUCCESS:
                        print(f"CHECK client status failed: {response}")
                        return False
                    if "details" not in response:
                        print(f"Check client status missing details: {response}.")
                        return False
                    else:
                        job_run_statuses = get_job_run_statuses(admin_api)
                        print(f"response is {response}")
                        print(f"job_run_statuses is {job_run_statuses}")
                        for row in response["details"]["client_statuses"]:
                            if row[3] != "stopped":
                                continue
                        # check if the current job is completed
                        if job_id in job_run_statuses and job_run_statuses[job_id] in (
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
    print("\nCommand: set_timeout")
    print(admin_api.set_timeout(11).get("details").get("message"))
    print("\nActive SP:")
    print(admin_api.get_active_sp().get("details"))
    print("\nList SP:")
    print(admin_api.list_sp().get("details"))
    print("\nCommand: get_available_apps_to_upload")
    print(admin_api.get_available_apps_to_upload())
    print("\nList Jobs:")
    list_jobs_return_message = admin_api.list_jobs("-a").get("details").get("message")
    print(list_jobs_return_message)
    first_job = list_jobs_return_message.split()[17]
    print("\nCommand: ls server -a .")
    ls_return_message = admin_api.ls_target("server", "-a", "..").get("details").get("message")
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
    print(("\n" + "*" * 120) * 20)
