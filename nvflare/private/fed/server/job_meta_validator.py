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

import collections
import logging
from io import BytesIO
from typing import Optional, Set, Tuple
from zipfile import ZipFile

from nvflare.apis.app_validation import AppValidationKey
from nvflare.apis.fl_constant import JobConstants
from nvflare.apis.job_def import ALL_SITES, SERVER_SITE_NAME, JobMetaKey
from nvflare.apis.job_meta_validator_spec import JobMetaValidatorSpec
from nvflare.fuel.utils.config import ConfigFormat
from nvflare.fuel.utils.config_factory import ConfigFactory
from nvflare.security.logging import secure_format_exception

CONFIG_FOLDER = "/config/"
CUSTOM_FOLDER = "/custom/"

MAX_CLIENTS = 1000000

logger = logging.getLogger(__name__)


class JobMetaValidator(JobMetaValidatorSpec):
    """Job validator"""

    def _validate_zf(self, job_name, zf):
        meta = self._validate_meta(job_name, zf)
        site_list = self._validate_deploy_map(job_name, meta)
        self._validate_app(job_name, meta, zf)
        clients = self._get_all_clients(site_list)
        self._validate_min_clients(job_name, meta, clients)
        self._validate_resource(job_name, meta)
        self._validate_mandatory_clients(job_name, meta, clients)
        return meta

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
            if isinstance(job_data, bytes):
                with ZipFile(BytesIO(job_data), "r") as zf:
                    meta = self._validate_zf(job_name, zf)
            elif isinstance(job_data, str):
                # job_data is a file name
                with ZipFile(job_data, "r") as zf:
                    meta = self._validate_zf(job_name, zf)
            else:
                raise TypeError(f"job_data must be bytes or str but got {type(job_data)}")
        except ValueError as e:
            return False, str(e), meta

        return True, "", meta

    @staticmethod
    def _validate_meta(job_name: str, zf: ZipFile) -> Optional[dict]:
        base_meta_file = f"{job_name}/{JobConstants.META}"
        logger.debug(f"validate file {base_meta_file}.[json|conf|yml] exists for job {job_name}")

        meta = None
        for ext, fmt in ConfigFormat.config_ext_formats().items():
            meta_file = f"{base_meta_file}{ext}"
            if meta_file in zf.namelist():
                config_loader = ConfigFactory.get_config_loader(fmt)
                meta_data = zf.read(meta_file)
                meta = config_loader.load_config_from_str(meta_data.decode()).to_dict()
                break
        return meta

    @staticmethod
    def _validate_deploy_map(job_name: str, meta: dict) -> list:

        if not meta:
            raise ValueError(
                f"{JobConstants.META}.[json|conf|yml] not existing for job {job_name}, possible in legacy job format.  Please upgrade the job structure."
            )

        deploy_map = meta.get(JobMetaKey.DEPLOY_MAP.value)
        if not deploy_map:
            raise ValueError(f"deploy_map is empty for job {job_name}")

        site_list = [site for deployments in deploy_map.values() for site in deployments]
        if not site_list:
            raise ValueError(f"No site is specified in deploy_map for job {job_name}")

        if ALL_SITES.casefold() in (site.casefold() for site in site_list):
            # if ALL_SITES is specified, no other site can be in the list
            if len(site_list) > 1:
                raise ValueError(f"No other site can be specified if {ALL_SITES} is used for job {job_name}")
            else:
                site_list = [ALL_SITES]
        elif SERVER_SITE_NAME not in site_list:
            raise ValueError(f"Missing server site in deploy_map for job {job_name}")
        else:
            duplicates = [site for site, count in collections.Counter(site_list).items() if count > 1]
            if duplicates:
                raise ValueError(f"Multiple apps to be deployed to following sites {duplicates} for job {job_name}")

        return site_list

    def _validate_app(self, job_name: str, meta: dict, zip_file: ZipFile) -> None:

        deploy_map = meta.get(JobMetaKey.DEPLOY_MAP.value)

        has_byoc = False
        for app, deployments in deploy_map.items():

            config_folder = job_name + "/" + app + CONFIG_FOLDER
            if not self._entry_exists(zip_file, config_folder):
                logger.debug(f"zip folder {config_folder} missing. Files in the zip:")
                for x in zip_file.namelist():
                    logger.debug(f"    {x}")
                raise ValueError(f"App '{app}' in deploy_map doesn't exist for job {job_name}")

            all_sites = ALL_SITES.casefold() in (site.casefold() for site in deployments)

            if (all_sites or SERVER_SITE_NAME in deployments) and not self._config_exists(
                zip_file, config_folder, JobConstants.SERVER_JOB_CONFIG
            ):
                raise ValueError(f"App '{app}' will be deployed to server but server config is missing")

            if (all_sites or [site for site in deployments if site != SERVER_SITE_NAME]) and not self._config_exists(
                zip_file, config_folder, JobConstants.CLIENT_JOB_CONFIG
            ):
                raise ValueError(f"App '{app}' will be deployed to client but client config is missing")

            custom_folder = job_name + "/" + app + CUSTOM_FOLDER
            if self._entry_exists(zip_file, custom_folder):
                has_byoc = True

        if has_byoc:
            meta[AppValidationKey.BYOC] = True

    @staticmethod
    def _convert_value_to_int(v) -> int:
        if isinstance(v, int):
            return v
        else:
            try:
                v = int(v)
                return v
            except ValueError as e:
                raise ValueError(f"invalid data type for {v},can't not convert to Int", secure_format_exception(e))
            except TypeError as e:
                raise ValueError(f"invalid data type for {v},can't not convert to Int", secure_format_exception(e))

    def _validate_min_clients(self, job_name: str, meta: dict, clients: set) -> None:
        logger.debug(f"validate min_clients for job {job_name}")

        value = meta.get(JobMetaKey.MIN_CLIENTS)
        if value is not None:
            min_clients = self._convert_value_to_int(value)
            if min_clients <= 0:
                raise ValueError(f"min_clients {min_clients} must be positive for job {job_name}")
            elif min_clients > MAX_CLIENTS:
                raise ValueError(f"min_clients {min_clients} must be less than {MAX_CLIENTS}  for job {job_name}")

            if next(iter(clients)) != ALL_SITES and len(clients) < min_clients:
                raise ValueError(f"min {min_clients} clients required for job {job_name}, found {len(clients)}.")

    @staticmethod
    def _validate_mandatory_clients(job_name: str, meta: dict, clients: set) -> None:
        logger.debug(f" validate mandatory_clients for job {job_name}")

        if next(iter(clients)) != ALL_SITES:
            # Validating mandatory clients are deployed
            mandatory_clients = meta.get(JobMetaKey.MANDATORY_CLIENTS)
            if mandatory_clients:
                mandatory_set = set(mandatory_clients)
                if not mandatory_set.issubset(clients):
                    diff = mandatory_set - clients
                    raise ValueError(f"Mandatory clients {diff} are not in the deploy_map for job {job_name}")

    @staticmethod
    def _validate_resource(job_name: str, meta: dict) -> None:
        logger.debug(f"validate resource for job {job_name}")

        resource_spec = meta.get(JobMetaKey.RESOURCE_SPEC.value)
        if resource_spec and not isinstance(resource_spec, dict):
            raise ValueError(f"Invalid resource_spec for job {job_name}")

        if not resource_spec:
            logger.debug("empty resource spec provided")

        if resource_spec:
            for k in resource_spec:
                if resource_spec[k] and not isinstance(resource_spec[k], dict):
                    raise ValueError(f"value for key {k} in resource spec is expecting a dictionary")

    @staticmethod
    def _get_all_clients(site_list: Optional[list]) -> Set[str]:

        if site_list[0] == ALL_SITES:
            return {ALL_SITES}

        return set([site for site in site_list if site != SERVER_SITE_NAME])

    @staticmethod
    def _entry_exists(zip_file: ZipFile, path: str) -> bool:
        try:
            zip_file.getinfo(path)
            return True
        except KeyError:
            return False

    @staticmethod
    def _config_exists(zip_file: ZipFile, zip_folder, init_config_path: str) -> bool:
        def match(parent: ZipFile, config_path: str) -> bool:
            import os

            full_path = os.path.join(zip_folder, config_path)
            return full_path in parent.namelist()

        return ConfigFactory.match_config(zip_file, init_config_path, match)
