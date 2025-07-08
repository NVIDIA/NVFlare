# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os
import shutil

from nvflare.lighter.constants import ProvFileName
from nvflare.lighter.spec import Builder, Project, ProvisionContext


class DockerImageBuilder(Builder):
    def __init__(
        self,
        base_dockerfile="Dockerfile",
        nvflare_url="2.6.0",
        image_name="nvflare/nvflare",
    ):
        """DockerImageBuilder generates a separate Dockefile and build script for each site.

        Args:
            base_dockerfile (str): Path to the base Dockerfile to use as a starting point.
            nvflare_url (str): URL or version string compatible with pip (e.g., a version or
                "git+https://github.com/NVIDIA/NVFlare.git@main").
            image_name (str): Name of the Docker image to be built.
        """
        self.base_dockerfile = base_dockerfile
        self.nvflare_url = nvflare_url
        self.image_name = image_name

    def _build_dockerfile(self, entity, ctx: ProvisionContext):
        dockerfile_path = os.path.join(ctx.get_ws_dir(entity), ProvFileName.DOCKERFILE)
        # Ensure base Dockerfile exists
        if not os.path.exists(self.base_dockerfile):
            raise FileNotFoundError(f"Base Dockerfile not found: {self.base_dockerfile}")

        to_be_copy = [ctx.get_kit_dir(entity), ctx.get_local_dir(entity), ctx.get_transfer_dir(entity)]
        relative_paths = [os.path.relpath(p, ctx.get_ws_dir(entity)) for p in to_be_copy]
        startup_script = self._determine_startup_script(entity, ctx)
        shutil.copy(self.base_dockerfile, dockerfile_path)
        with open(dockerfile_path, "a") as f:
            f.write(f"RUN pip install {self.nvflare_url}\n")
            for p in relative_paths:
                f.write(f"COPY {p} /opt/nvflare/{p}\n")
            f.write(f'ENTRYPOINT ["/opt/nvflare/startup/{startup_script}"]\n')

    def _build_build_docker_sh(self, participant, ctx):
        build_script_path = os.path.join(ctx.get_ws_dir(participant), ProvFileName.DOCKER_BUILD_SH)
        image_tag = f"{self.image_name}:{participant.name}-image"
        with open(build_script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"docker build -t {image_tag} -f {ProvFileName.DOCKERFILE} .\n")

    def _determine_startup_script(self, participant, ctx):
        if participant.type == "admin":
            return "fl_admin.sh"
        else:
            return "sub_start.sh"

    def finalize(self, project: Project, ctx: ProvisionContext):
        for p in project.get_all_participants():
            self._build_dockerfile(p, ctx)
            self._build_build_docker_sh(p, ctx)
