# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import glob
import os
from typing import List, Union

from nvflare.apis.fl_constant import WorkspaceConstants


class Workspace:
    def __init__(self, root_dir: str, site_name: str = "", config_folder: str = "config"):
        """Define a workspace.

        NOTE::

            Example of client workspace folder structure:

                Workspace ROOT
                    local
                        authorization.json.default
                        resources.json.default
                        custom/
                            custom python code
                        ...
                    startup (optional)
                        provisioned content
                        fed_client.json
                    run_1
                        app
                            config (required)
                                configurations
                            custom (optional)
                                custom python code
                            other_folder (app defined)
                        log.txt
                        job_meta.json
                        ...

        Args:
            root_dir: root directory of the workspace
            site_name: site name of the workspace
            config_folder: where to find required config inside an app
        """
        self.root_dir = root_dir
        self.site_name = site_name
        self.config_folder = config_folder

        # check to make sure the workspace is valid
        if not os.path.isdir(root_dir):
            raise RuntimeError(f"invalid workspace {root_dir}: it does not exist or not a valid dir")

        startup_dir = self.get_startup_kit_dir()
        if not os.path.isdir(startup_dir):
            raise RuntimeError(
                f"invalid workspace {root_dir}: missing startup folder '{startup_dir}' or not a valid dir"
            )

        site_dir = self.get_site_config_dir()
        if not os.path.isdir(site_dir):
            raise RuntimeError(
                f"invalid workspace {root_dir}: missing site config folder '{site_dir}' or not a valid dir"
            )

    def _fallback_path(self, file_names: [str]):
        for n in file_names:
            f = self.get_file_path_in_site_config(n)
            if os.path.exists(f):
                return f
        return None

    def get_authorization_file_path(self):
        return self._fallback_path(
            [WorkspaceConstants.AUTHORIZATION_CONFIG, WorkspaceConstants.DEFAULT_AUTHORIZATION_CONFIG]
        )

    def get_resources_file_path(self):
        return self._fallback_path([WorkspaceConstants.RESOURCES_CONFIG, WorkspaceConstants.DEFAULT_RESOURCES_CONFIG])

    def get_job_resources_file_path(self):
        return self.get_file_path_in_site_config(WorkspaceConstants.JOB_RESOURCES_CONFIG)

    def get_log_config_file_path(self):
        return self._fallback_path([WorkspaceConstants.LOGGING_CONFIG, WorkspaceConstants.DEFAULT_LOGGING_CONFIG])

    def get_file_path_in_site_config(self, file_basename: Union[str, List[str]]):
        if isinstance(file_basename, str):
            return os.path.join(self.get_site_config_dir(), file_basename)
        elif isinstance(file_basename, list):
            return self._fallback_path(file_basename)
        else:
            raise ValueError(f"invalid file_basename '{file_basename}': must be str or List[str]")

    def get_file_path_in_startup(self, file_basename: str):
        return os.path.join(self.get_startup_kit_dir(), file_basename)

    def get_file_path_in_root(self, file_basename: str):
        return os.path.join(self.root_dir, file_basename)

    def get_server_startup_file_path(self):
        # this is to get the full path to "fed_server.json"
        return self.get_file_path_in_startup(WorkspaceConstants.SERVER_STARTUP_CONFIG)

    def get_server_app_config_file_path(self, job_id):
        return os.path.join(self.get_app_config_dir(job_id), WorkspaceConstants.SERVER_APP_CONFIG)

    def get_client_app_config_file_path(self, job_id):
        return os.path.join(self.get_app_config_dir(job_id), WorkspaceConstants.CLIENT_APP_CONFIG)

    def get_client_startup_file_path(self):
        # this is to get the full path to "fed_client.json"
        return self.get_file_path_in_startup(WorkspaceConstants.CLIENT_STARTUP_CONFIG)

    def get_admin_startup_file_path(self):
        # this is to get the full path to "fed_admin.json"
        return self.get_file_path_in_startup(WorkspaceConstants.ADMIN_STARTUP_CONFIG)

    def get_site_config_dir(self) -> str:
        return os.path.join(self.root_dir, WorkspaceConstants.SITE_FOLDER_NAME)

    def get_site_custom_dir(self) -> str:
        return os.path.join(self.get_site_config_dir(), WorkspaceConstants.CUSTOM_FOLDER_NAME)

    def get_startup_kit_dir(self) -> str:
        return os.path.join(self.root_dir, WorkspaceConstants.STARTUP_FOLDER_NAME)

    def get_audit_file_path(self) -> str:
        return os.path.join(self.root_dir, WorkspaceConstants.AUDIT_LOG)

    def get_log_file_path(self) -> str:
        return os.path.join(self.root_dir, WorkspaceConstants.LOG_FILE_NAME)

    def get_root_dir(self) -> str:
        return self.root_dir

    def get_run_dir(self, job_id: str) -> str:
        return os.path.join(self.root_dir, WorkspaceConstants.WORKSPACE_PREFIX + str(job_id))

    def get_app_dir(self, job_id: str) -> str:
        return os.path.join(self.get_run_dir(job_id), WorkspaceConstants.APP_PREFIX + self.site_name)

    def get_app_log_file_path(self, job_id: str) -> str:
        return os.path.join(self.get_run_dir(job_id), WorkspaceConstants.LOG_FILE_NAME)

    def get_app_error_log_file_path(self, job_id: str) -> str:
        return os.path.join(self.get_run_dir(job_id), WorkspaceConstants.ERROR_LOG_FILE_NAME)

    def get_app_config_dir(self, job_id: str) -> str:
        return os.path.join(self.get_app_dir(job_id), self.config_folder)

    def get_app_custom_dir(self, job_id: str) -> str:
        return os.path.join(self.get_app_dir(job_id), WorkspaceConstants.CUSTOM_FOLDER_NAME)

    def get_job_meta_path(self, job_id: str) -> str:
        return os.path.join(self.get_run_dir(job_id), WorkspaceConstants.JOB_META_FILE)

    def get_site_privacy_file_path(self):
        return self.get_file_path_in_site_config(WorkspaceConstants.PRIVACY_CONFIG)

    def get_client_custom_dir(self) -> str:
        return os.path.join(self.get_site_config_dir(), WorkspaceConstants.CUSTOM_FOLDER_NAME)

    def get_stats_pool_summary_path(self, job_id: str, prefix=None) -> str:
        file_name = WorkspaceConstants.STATS_POOL_SUMMARY_FILE_NAME
        if prefix:
            file_name = f"{prefix}.{file_name}"
        return os.path.join(self.get_run_dir(job_id), file_name)

    def get_stats_pool_records_path(self, job_id: str, prefix=None) -> str:
        file_name = WorkspaceConstants.STATS_POOL_RECORDS_FILE_NAME
        if prefix:
            file_name = f"{prefix}.{file_name}"
        return os.path.join(self.get_run_dir(job_id), file_name)

    def get_config_files_for_startup(self, is_server: bool, for_job: bool) -> list:
        """Get all config files to be used for startup of the process (SP, SJ, CP, CJ).

        We first get required config files:
            - the startup file (fed_server.json or fed_client.json) in "startup" folder
            - resource file (resources.json.default or resources.json) in "local" folder

        We then try to get resources files (usually generated by different builders of the Provision system):
            - resources files from the "startup" folder take precedence
            - resources files from the "local" folder are next

        These extra resource config files must be json and follow the following patterns:
        - *__resources.json: these files are for both parent process and job processes
        - *__p_resources.json: these files are for parent process only
        - *__j_resources.json: these files are for job process only

        Args:
            is_server: whether this is for server site or client site
            for_job: whether this is for job process or parent process

        Returns: a list of config file names

        """
        if is_server:
            startup_file_path = self.get_server_startup_file_path()
        else:
            startup_file_path = self.get_client_startup_file_path()

        resource_config_path = self.get_resources_file_path()
        config_files = [startup_file_path, resource_config_path]
        if for_job:
            # this is for job process
            job_resources_file_path = self.get_job_resources_file_path()
            if os.path.exists(job_resources_file_path):
                config_files.append(job_resources_file_path)

        # add other resource config files
        patterns = [WorkspaceConstants.RESOURCE_FILE_NAME_PATTERN]
        if for_job:
            patterns.append(WorkspaceConstants.JOB_RESOURCE_FILE_NAME_PATTERN)
        else:
            patterns.append(WorkspaceConstants.PARENT_RESOURCE_FILE_NAME_PATTERN)

        # add startup files first, then local files
        self._add_resource_files(self.get_startup_kit_dir(), config_files, patterns)
        self._add_resource_files(self.get_site_config_dir(), config_files, patterns)
        return config_files

    @staticmethod
    def _add_resource_files(from_dir: str, to_list: list, patterns: [str]):
        for p in patterns:
            files = glob.glob(os.path.join(from_dir, p))
            if files:
                to_list.extend(files)
