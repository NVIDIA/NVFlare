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
import collections
import json
from io import BytesIO
from typing import Set, Tuple
from zipfile import ZipFile

from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey

SERVER_CONFIG = "config_fed_server.json"
CLIENT_CONFIG = "config_fed_client.json"


class JobValidator:
    """Job validator"""

    def __init__(self, fl_ctx: FLContext):
        self.fl_ctx = fl_ctx

    def validate(self, job_name: str, job_data: bytes) -> Tuple[bool, str, dict]:
        """Validate job

        Args:
            job_name (str): Job name
            job_data (bytes): Job ZIP data

        Returns:
            Tuple[bool, str, dict]: (is_valid, error_message, meta)
        """

        meta = {}
        try:
            with ZipFile(BytesIO(job_data), "r") as zf:
                meta_file = job_name + "/meta.json"
                meta_data = zf.read(meta_file)
                meta = json.loads(meta_data)
                if JobMetaKey.JOB_FOLDER_NAME not in meta:
                    meta[JobMetaKey.JOB_FOLDER_NAME.value] = job_name

                self._validate_deployment(meta, zf)

                # Validating resource_spec
                resource_spec = meta.get(JobMetaKey.RESOURCE_SPEC.value)
                if resource_spec and not isinstance(resource_spec, dict):
                    raise ValueError(f"Invalid resource_spec for job {job_name}")

        except ValueError as e:
            return False, str(e), meta

        return True, "", meta

    def _validate_deployment(self, meta: dict, zip_file: ZipFile) -> None:

        job_name = meta.get(JobMetaKey.JOB_FOLDER_NAME)
        deploy_map = meta.get(JobMetaKey.DEPLOY_MAP)
        if not deploy_map:
            raise ValueError(f"deploy_map is missing in meta.json for job {job_name}")

        # Validating mandatory clients are deployed
        mandatory_clients = meta.get(JobMetaKey.MANDATORY_CLIENTS)
        if mandatory_clients:
            mandatory_set = set(mandatory_clients)
            all_clients = self._get_all_clients(meta)
            if all_clients and not mandatory_set.issubset(all_clients):
                diff = mandatory_set - all_clients
                raise ValueError(f"Mandatory clients {diff} are not in the deployment for job {job_name}")

        # Validating all apps exists
        site_list = []
        for app, deployment in deploy_map.items():
            zip_folder = job_name + "/" + app + "/config/"
            if not self._entry_exists(zip_file, zip_folder):
                raise ValueError(f"App {app} in deploy_map doesn't exist for job {job_name}")

            if deployment:
                site_list.extend(deployment)

                if "server" in deployment and not self._entry_exists(zip_file, zip_folder + SERVER_CONFIG):
                    raise ValueError(f"App {app} is deployed to server but server config is missing")

                if [client for client in deployment if client != "server"] and not self._entry_exists(
                    zip_file, zip_folder + CLIENT_CONFIG
                ):
                    raise ValueError(f"App {app} is deployed to client but client config is missing")

        # Make sure an app is not deployed to multiple clients
        if site_list:
            duplicates = [site for site, count in collections.Counter(site_list).items() if count > 1]
            if duplicates:
                raise ValueError(f"Multiple apps to be deployed to following sites {duplicates} for job {job_name}")

            min_clients = meta.get(JobMetaKey.MIN_CLIENTS)
            client_set = set([site for site in site_list if site != "server"])
            if min_clients and len(client_set) < min_clients:
                raise ValueError(f"min_clients {min_clients} not met in the deployment for job {job_name}")

    @staticmethod
    def _get_all_clients(meta: dict) -> Set[str]:

        all_clients = set()
        deploy_map = meta.get(JobMetaKey.DEPLOY_MAP)
        if deploy_map:
            for k, v in deploy_map.items():
                all_clients.update(v)
        all_clients.discard("server")

        return all_clients

    @staticmethod
    def _entry_exists(zip_file: ZipFile, path: str) -> bool:
        try:
            zip_file.getinfo(path)
            return True
        except KeyError:
            return False
