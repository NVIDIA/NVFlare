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
        image_name="nvflare",
        requirement: str = None,
        requirements_txt_path: str = None,
    ):
        """Initializes a DockerImageBuilder.

        This builder generates a Dockefile and build script for each site.
        It does NOT perform the actual Docker image build.

        You can specify Python dependencies using either:

        1. A single pip-style requirement string (e.g., "nvflare==2.7.0")
        2. A full requirements.txt file for more complex environments.

        Only one of `requirement` or `requirements_txt_path` should be provided.

        Args:
            base_dockerfile (str): Path to a base Dockerfile template used for generating site-specific Dockerfiles.
            image_name (str): Name (and optionally tag) to assign to the Docker image (used in the generated build script).
            requirement (str, optional): A single pip-style requirement string (e.g., "nvflare==2.7.0").
            requirements_txt_path (str, optional): Path to a `requirements.txt` file listing dependencies to install.


        Raises:
            ValueError: If both or neither of `requirement` and `requirements_txt_path` are provided.

        Example:
            DockerImageBuilder(requirement="nvflare==2.7.0")

            DockerImageBuilder(requirements_txt_path="env/requirements.txt")
        """
        if requirement and requirements_txt_path:
            raise ValueError("Specify either `requirement` or `requirements_txt_path`, not both.")
        if not requirement and not requirements_txt_path:
            raise ValueError("You must specify either `requirement` or `requirements_txt_path`.")
        if requirements_txt_path and not os.path.exists(requirements_txt_path):
            raise ValueError(f"{requirements_txt_path=} does not exist.")

        self.requirement = requirement
        self.requirements_txt_path = os.path.abspath(requirements_txt_path) if requirements_txt_path else None
        self.base_dockerfile = base_dockerfile
        self.image_name = image_name

    def _build_dockerfile(self, entity, ctx: ProvisionContext):
        dockerfile_path = os.path.join(ctx.get_ws_dir(entity), ProvFileName.DOCKERFILE)
        # Ensure base Dockerfile exists
        base_dockerfile = entity.get_prop("base_dockerfile", self.base_dockerfile)
        if not os.path.exists(base_dockerfile):
            raise FileNotFoundError(f"Base Dockerfile not found: {base_dockerfile}")
        ctx.info(f"Building from {base_dockerfile=}")

        to_be_copy = [ctx.get_kit_dir(entity), ctx.get_local_dir(entity), ctx.get_transfer_dir(entity)]
        relative_paths = [os.path.relpath(p, ctx.get_ws_dir(entity)) for p in to_be_copy]
        startup_script = self._determine_startup_script(entity, ctx)
        shutil.copy(self.base_dockerfile, dockerfile_path)
        with open(dockerfile_path, "a") as f:
            if self.requirement:
                f.write(f"RUN pip install {self.requirement}\n")
            elif self.requirements_txt_path:
                f.write(f"RUN pip install -r {self.requirements_txt_path}\n")
            for p in relative_paths:
                f.write(f"COPY {p} /opt/nvflare/{p}\n")
            f.write(f'ENTRYPOINT ["/opt/nvflare/startup/{startup_script}"]\n')

    def _build_build_docker_sh(self, participant, ctx):
        build_script_path = os.path.join(ctx.get_ws_dir(participant), ProvFileName.DOCKER_BUILD_SH)
        image_tag = f"{self.image_name}_{participant.name}"
        archive_name = f"{image_tag}.tgz"

        with open(build_script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write('SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"\n')
            f.write(f"docker build -t {image_tag} -f $SCRIPT_DIR/{ProvFileName.DOCKERFILE} .\n")
            f.write(f'echo "DOCKER_IMAGE_TAG={image_tag}"\n')
            f.write(f"docker save {image_tag} | gzip > $SCRIPT_DIR/{archive_name}\n")
            f.write(f'echo "DOCKER_ARCHIVE=$SCRIPT_DIR/{archive_name}"\n')
        os.chmod(build_script_path, 0o755)

    def _determine_startup_script(self, participant, ctx):
        if participant.type == "admin":
            return "fl_admin.sh"
        else:
            return "sub_start.sh"

    def finalize(self, project: Project, ctx: ProvisionContext):
        for p in project.get_all_participants():
            self._build_dockerfile(p, ctx)
            self._build_build_docker_sh(p, ctx)
