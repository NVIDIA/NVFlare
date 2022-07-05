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
import shutil
from typing import List

import yaml

from nvflare.apis.job_def import JobMetaKey
from nvflare.fuel.hci.client.fl_admin_api import FLAdminAPI

FED_SERVER_CONFIG = "fed_server.json"
FILE_STORAGE = "nvflare.app_common.storages.filesystem_storage.FilesystemStorage"


def generate_meta(job_name: str, clients: List[str]):
    resource_spec = {c: {"gpu": 1} for c in clients}
    deploy_map = {job_name: ["@all"]}
    meta = {
        "name": job_name,
        JobMetaKey.RESOURCE_SPEC: resource_spec,
        JobMetaKey.DEPLOY_MAP: deploy_map,
    }
    return meta


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


def _read_server_json(poc_path: str) -> dict:
    fed_server_json_path = os.path.join(poc_path, "server", "startup", FED_SERVER_CONFIG)
    if not os.path.exists(os.path.join(poc_path, "server", "startup", FED_SERVER_CONFIG)):
        raise RuntimeError(f"Missing {FED_SERVER_CONFIG} at: {fed_server_json_path}")
    with open(fed_server_json_path, "r") as f:
        fed_server_json = json.load(f)
    return fed_server_json


def get_snapshot_path_from_poc(path: str) -> str:
    fed_server_json = _read_server_json(path)
    if "snapshot_persistor" not in fed_server_json:
        raise RuntimeError(f"Missing snapshot_persistor in {FED_SERVER_CONFIG}")
    if "args" not in fed_server_json["snapshot_persistor"]:
        raise RuntimeError("Missing args in snapshot_persistor")
    if "storage" not in fed_server_json["snapshot_persistor"]["args"]:
        raise RuntimeError("Missing storage in snapshot_persistor's args")
    if "args" not in fed_server_json["snapshot_persistor"]["args"]["storage"]:
        raise RuntimeError("Missing args in snapshot_persistor's storage")
    if "path" not in fed_server_json["snapshot_persistor"]["args"]["storage"]:
        raise RuntimeError("Missing path in snapshot_persistor's storage")
    if fed_server_json["snapshot_persistor"]["args"]["storage"]["path"] != FILE_STORAGE:
        raise RuntimeError(f"Only support {FILE_STORAGE} storage in snapshot_persistor's args")
    if "root_dir" not in fed_server_json["snapshot_persistor"]["args"]["storage"]["args"]:
        raise RuntimeError("Missing root_dir in snapshot_persistor's storage's args")
    return fed_server_json["snapshot_persistor"]["args"]["storage"]["args"]["root_dir"]


def get_job_store_path_from_poc(path: str) -> str:
    fed_server_json = _read_server_json(path)
    if "components" not in fed_server_json:
        raise RuntimeError(f"Missing components in {FED_SERVER_CONFIG}")
    job_manager_config = None
    for c in fed_server_json["components"]:
        if "id" in c and c["id"] == "job_manager":
            job_manager_config = c
    if not job_manager_config:
        raise RuntimeError(f"Missing job_manager in {FED_SERVER_CONFIG}")
    if "args" not in job_manager_config:
        raise RuntimeError("Missing args in job_manager.")
    if "uri_root" not in job_manager_config["args"]:
        raise RuntimeError("Missing uri_root in job_manager's args.")
    return job_manager_config["args"]["uri_root"]


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
    list_jobs_return_message = admin_api.list_jobs().get("details").get("message")
    print(list_jobs_return_message)
    first_job = list_jobs_return_message.split()[17]
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
    print(("\n" + "*" * 120) * 20)
