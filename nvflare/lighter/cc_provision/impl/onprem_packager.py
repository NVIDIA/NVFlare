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
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import yaml

from nvflare.lighter.constants import ProvFileName
from nvflare.lighter.ctx import ProvisionContext
from nvflare.lighter.entity import Participant, Project
from nvflare.lighter.spec import Packager

BUILD_IMAGE_CMD = "build_cvm_image.sh"


def _extract_cvm_tar_path(output):
    for line in output.splitlines():
        match = re.search(r"CVM Bundle\s+([^\s]+)\s+is ready", line)
        if match:
            return match.group(1)
    return None


def _extract_docker_tar_path(output):
    for line in output.splitlines():
        match = re.search(r"DOCKER_ARCHIVE=([^\s]+)", line)
        if match:
            return match.group(1)
    return None


def to_abs_path(yaml_path, file_path):
    """Converts a relative file path to an absolute path based on the directory of the given YAML file.

    Args:
        yaml_path (str): Path to the YAML file. Must be a non-empty string.
        file_path (str): Target file path. If relative, it's resolved against the YAML file's directory.

    Returns:
        str: An absolute file path.

    Raises:
        RuntimeError: If either input is None or empty.
    """
    if not yaml_path or not isinstance(yaml_path, str):
        raise ValueError("Invalid input: 'yaml_path' must be a non-empty string.")
    if not file_path or not isinstance(file_path, str):
        raise ValueError("Invalid input: 'file_path' must be a non-empty string.")

    if os.path.isabs(file_path):
        return os.path.normpath(file_path)

    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
    abs_path = os.path.abspath(os.path.join(yaml_dir, file_path))
    return os.path.normpath(abs_path)


def run_command(command, cwd=None):
    print(f"Running {command=} in {cwd=}")
    process = subprocess.Popen(
        command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True
    )

    stdout_lines = []
    stderr_lines = []

    # Read stdout and stderr asynchronously
    while True:
        stdout_line = process.stdout.readline()
        stderr_line = process.stderr.readline()

        if stdout_line:
            print(stdout_line, end="")
            stdout_lines.append(stdout_line)
        if stderr_line:
            print(stderr_line, end="")
            stderr_lines.append(stderr_line)

        if stdout_line == "" and stderr_line == "" and process.poll() is not None:
            break

    retcode = process.wait()
    if retcode != 0:
        raise subprocess.CalledProcessError(
            retcode, command, output="".join(stdout_lines), stderr="".join(stderr_lines)
        )
    return "".join(stdout_lines)


class OnPremPackager(Packager):
    def __init__(self, cc_config_key="cc_config", build_image_cmd=BUILD_IMAGE_CMD):
        super().__init__()
        self.cc_config_key = cc_config_key
        self.build_image_cmd = build_image_cmd

    def _build_cc_image(self, cc_config_yaml: str):
        """Build CC image for the site."""
        build_image_cmd = to_abs_path(cc_config_yaml, self.build_image_cmd)
        if not os.path.exists(build_image_cmd) or not os.access(build_image_cmd, os.X_OK):
            raise FileNotFoundError(f"Build image command '{build_image_cmd}' not found or is not executable.")
        command = [build_image_cmd, cc_config_yaml]
        output = run_command(command)
        tar_file_path = _extract_cvm_tar_path(output)
        return tar_file_path

    def _add_startup_kit_to_cc_config(self, cc_config_path: str, startup_kit_path: str):
        with open(cc_config_path, "r") as f:
            data = yaml.safe_load(f)

            user_config = data.get("user_config", {})
            user_config.update({"nvflare": startup_kit_path})
            data["user_config"] = user_config

        # Save the updated YAML back to file
        with open(cc_config_path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False)

    def _package_for_participant(self, participant: Participant, ctx: ProvisionContext):
        """Package the startup kit for the participant."""
        if not participant.get_prop(self.cc_config_key):
            return

        dest_dir = Path(ctx.get_result_location())

        cc_config_yaml = os.path.abspath(participant.get_prop(self.cc_config_key))
        if not os.path.exists(cc_config_yaml):
            raise RuntimeError(f"{cc_config_yaml=} does not exist")

        fd, temp_cc_config_yaml = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)

        tar_file_path = None
        try:
            shutil.copyfile(cc_config_yaml, temp_cc_config_yaml)
            # add startup kit to yaml automatically
            self._add_startup_kit_to_cc_config(temp_cc_config_yaml, str(dest_dir / participant.name))
            # Build CC image
            tar_file_path = self._build_cc_image(temp_cc_config_yaml)
        finally:
            os.remove(temp_cc_config_yaml)

        if tar_file_path is None or not os.path.exists(tar_file_path):
            raise RuntimeError("CVM build failed")

        # Copy the package that is generated by the build_image_cmd
        site_dir = dest_dir / participant.name
        shutil.rmtree(site_dir)
        os.mkdir(site_dir)
        shutil.copy(tar_file_path, site_dir / f"{participant.name}.tgz")

    def package(self, project: Project, ctx: ProvisionContext):
        start_all_script = os.path.join(ctx.get_result_location(), ProvFileName.START_ALL_SH)
        if os.path.exists(start_all_script):
            os.remove(start_all_script)

        participants = project.get_all_participants()

        for i, participant in enumerate(participants):
            self._package_for_participant(participant, ctx)
            if i != len(participants) - 1:
                time.sleep(100.0)
