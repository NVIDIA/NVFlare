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

from nvflare.lighter.impl.static_file import StaticFileBuilder


class LocalStaticFileBuilder(StaticFileBuilder):
    def __init__(
        self,
        enable_byoc=False,
        config_folder="",
        scheme="grpc",
        app_validator="",
        download_job_url="",
        docker_image="",
        snapshot_persistor="",
        overseer_agent="",
        components="",
    ):
        """Build all static files from template.

        Uses the information from project.yml through project to go through the participants and write the contents of
        each file with the template, and replacing with the appropriate values from project.yml.

        Usually, two main categories of files are created in all FL participants, static and dynamic. Static files
        have similar contents among different participants, with small differences.  For example, the differences in
        sub_start.sh are client name and python module.  Those are basically static files.  This builder uses template
        file and string replacement to generate those static files for each participant.

        Args:
            enable_byoc: for each participant, true to enable loading of code in the custom folder of applications
            config_folder: usually "config"
            app_validator: optional path to an app validator to verify that uploaded app has the expected structure
            docker_image: when docker_image is set to a docker image name, docker.sh will be generated on server/client/admin
        """
        super().__init__(
            enable_byoc,
            config_folder,
            scheme,
            app_validator,
            download_job_url,
            docker_image,
            snapshot_persistor,
            overseer_agent,
            components,
        )

    def get_server_name(self, server):
        return "localhost"

    def get_overseer_name(self, overseer):
        return "localhost"

    def prepare_admin_config(self, admin, ctx):
        config = super().prepare_admin_config(admin, ctx)
        config["admin"]["username"] = admin.name
        config["admin"]["cred_type"] = "local_cert"
        return config
