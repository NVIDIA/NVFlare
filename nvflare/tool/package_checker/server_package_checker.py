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
import signal
import sys
from subprocess import TimeoutExpired

from .package_checker import PackageChecker
from .utils import check_overseer_running, check_response, run_command_in_subprocess, try_bind_address, try_write

SERVER_SCRIPT = "nvflare.private.fed.app.server.server_train"
SERVER_NVF_CONFIG = "fed_server.json"


def _get_snapshot_storage_root(fed_config: dict) -> str:
    snapshot_storage_root = ""
    if (
        fed_config.get("snapshot_persistor", {}).get("path")
        == "nvflare.app_common.state_persistors.storage_state_persistor.StorageStatePersistor"
    ):
        storage = fed_config["snapshot_persistor"].get("args", {}).get("storage")
        if storage["path"] == "nvflare.app_common.storages.filesystem_storage.FilesystemStorage":
            snapshot_storage_root = storage["args"]["root_dir"]
    return snapshot_storage_root


def _get_job_storage_root(fed_config: dict) -> str:
    job_storage_root = ""
    for c in fed_config.get("components", []):
        if c.get("path") == "nvflare.apis.impl.job_def_manager.SimpleJobDefManager":
            job_storage_root = c["args"]["uri_root"]
    return job_storage_root


class ServerPackageChecker(PackageChecker):
    def __init__(self):
        super().__init__()
        self.snapshot_storage_root = None
        self.job_storage_root = None

    def should_be_checked(self, package_path) -> bool:
        startup = os.path.join(package_path, "startup")
        if os.path.exists(os.path.join(startup, SERVER_NVF_CONFIG)):
            return True
        return False

    def check(self, package_path):
        try:
            startup = os.path.join(package_path, "startup")
            fed_config_file = os.path.join(startup, SERVER_NVF_CONFIG)
            with open(fed_config_file, "r") as f:
                fed_config = json.load(f)

            # check overseer
            resp = check_overseer_running(
                startup=startup, overseer_agent_conf=fed_config["overseer_agent"], role="server"
            )
            if not check_response(resp):
                self.add_report(package_path, "Can't connect to overseer", "Please check if overseer is up.")

            # check if server grpc port is open
            server_conf = fed_config["servers"][0]
            grpc_service_config = server_conf["service"]
            grpc_target_address = grpc_service_config["target"]
            host, port = grpc_target_address.split(":")
            e = try_bind_address(host, int(port))
            if e:
                self.add_report(
                    package_path,
                    f"Can't bind to address ({grpc_target_address}) for grpc service: {e}",
                    "Please check the DNS and port.",
                )

            # check if server admin command port is open
            admin_host, admin_port = server_conf["admin_host"], int(server_conf["admin_port"])
            e = try_bind_address(admin_host, admin_port)
            if e:
                self.add_report(
                    package_path,
                    f"Can't bind to address ({admin_host}:{admin_port}) for admin service: {e}",
                    "Please check the DNS and port.",
                )

            # check if user can write to snapshot storage
            self.snapshot_storage_root = _get_snapshot_storage_root(fed_config=fed_config)
            if self.snapshot_storage_root:
                e = try_write(self.snapshot_storage_root)
                if e:
                    self.add_report(
                        package_path,
                        f"Can't write to {self.snapshot_storage_root}: {e}.",
                        "Please check the user permission.",
                    )

            # check if user can write to job storage
            self.job_storage_root = _get_job_storage_root(fed_config=fed_config)
            if self.job_storage_root:
                e = try_write(self.job_storage_root)
                if e:
                    self.add_report(
                        package_path,
                        f"Can't write to {self.job_storage_root}: {e}.",
                        "Please check the user permission.",
                    )

            # check if server can run
            if len(self.report[package_path]) == 0:
                command = (
                    f"{sys.executable} -m {SERVER_SCRIPT}"
                    f" -m {package_path} -s {SERVER_NVF_CONFIG}"
                    " --set secure_train=false config_folder=config"
                )
                process = run_command_in_subprocess(command)
                try:
                    out, _ = process.communicate(timeout=3)
                    self.add_report(
                        package_path,
                        f"Can't start server successfully: \n{out}",
                        "Please check the error message of dry run.",
                    )
                except TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)

        except Exception as e:
            self.add_report(
                package_path,
                f"Exception happens in checking {e}, this package is not in correct format.",
                "Please download a new package.",
            )

    def dry_run(self, package_path):
        command = (
            f"{sys.executable} -m {SERVER_SCRIPT}"
            f" -m {package_path} -s {SERVER_NVF_CONFIG}"
            " --set secure_train=false config_folder=config"
        )
        self.dry_run_process = run_command_in_subprocess(command)

    def stop_dry_run(self, package_path):
        super().stop_dry_run(package_path)
        if os.path.exists(self.snapshot_storage_root):
            shutil.rmtree(self.snapshot_storage_root)
        if os.path.exists(self.job_storage_root):
            shutil.rmtree(self.job_storage_root)
