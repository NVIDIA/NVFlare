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
import shutil
import sys

from .check_rule import CheckAddressBinding, CheckOverseerRunning, CheckWriting
from .package_checker import PackageChecker
from .utils import NVFlareConfig, NVFlareRole

SERVER_SCRIPT = "nvflare.private.fed.app.server.server_train"


def _get_server_fed_config(package_path: str):
    startup = os.path.join(package_path, "startup")
    fed_config_file = os.path.join(startup, NVFlareConfig.SERVER)
    with open(fed_config_file, "r") as f:
        fed_config = json.load(f)
    return fed_config


def _get_snapshot_storage_root(package_path: str) -> str:
    fed_config = _get_server_fed_config(package_path)
    snapshot_storage_root = ""
    if (
        fed_config.get("snapshot_persistor", {}).get("path")
        == "nvflare.app_common.state_persistors.storage_state_persistor.StorageStatePersistor"
    ):
        storage = fed_config["snapshot_persistor"].get("args", {}).get("storage")
        if storage["path"] == "nvflare.app_common.storages.filesystem_storage.FilesystemStorage":
            snapshot_storage_root = storage["args"]["root_dir"]
    return snapshot_storage_root


def _get_job_storage_root(package_path: str) -> str:
    fed_config = _get_server_fed_config(package_path)
    job_storage_root = ""
    for c in fed_config.get("components", []):
        if c.get("path") == "nvflare.apis.impl.job_def_manager.SimpleJobDefManager":
            job_storage_root = c["args"]["uri_root"]
    return job_storage_root


def _get_grpc_host_and_port(package_path: str) -> (str, int):
    fed_config = _get_server_fed_config(package_path)
    server_conf = fed_config["servers"][0]
    grpc_service_config = server_conf["service"]
    grpc_target_address = grpc_service_config["target"]
    _, port = grpc_target_address.split(":")
    return "localhost", int(port)


def _get_admin_host_and_port(package_path: str) -> (str, int):
    fed_config = _get_server_fed_config(package_path)
    server_conf = fed_config["servers"][0]
    return "localhost", int(server_conf["admin_port"])


class ServerPackageChecker(PackageChecker):
    def __init__(self):
        super().__init__()
        self.snapshot_storage_root = None
        self.job_storage_root = None

    def init_rules(self, package_path):
        self.dry_run_timeout = 3
        self.rules = [
            CheckOverseerRunning(name="Check overseer running", role=NVFlareRole.SERVER),
            CheckAddressBinding(name="Check grpc port binding", get_host_and_port_from_package=_get_grpc_host_and_port),
            CheckAddressBinding(
                name="Check admin port binding", get_host_and_port_from_package=_get_admin_host_and_port
            ),
            CheckWriting(name="Check snapshot storage writable", get_filename_from_package=_get_snapshot_storage_root),
            CheckWriting(name="Check job storage writable", get_filename_from_package=_get_job_storage_root),
        ]

    def should_be_checked(self) -> bool:
        startup = os.path.join(self.package_path, "startup")
        if os.path.exists(os.path.join(startup, NVFlareConfig.SERVER)):
            return True
        return False

    def get_dry_run_command(self) -> str:
        command = (
            f"{sys.executable} -m {SERVER_SCRIPT}"
            f" -m {self.package_path} -s {NVFlareConfig.SERVER}"
            " --set secure_train=true config_folder=config"
        )
        self.snapshot_storage_root = _get_snapshot_storage_root(self.package_path)
        self.job_storage_root = _get_job_storage_root(self.package_path)
        return command

    def stop_dry_run(self, force=True):
        super().stop_dry_run(force=force)
        if os.path.exists(self.snapshot_storage_root):
            shutil.rmtree(self.snapshot_storage_root)
        if os.path.exists(self.job_storage_root):
            shutil.rmtree(self.job_storage_root)
