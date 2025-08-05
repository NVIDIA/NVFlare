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

import json
import os
import shutil
import subprocess
from pathlib import Path

import yaml

from nvflare.lighter.constants import ProvFileName
from nvflare.lighter.ctx import ProvisionContext
from nvflare.lighter.entity import Participant, Project
from nvflare.lighter.spec import Packager

BUILD_IMAGE_CMD = "build_cvm_image.sh"


def update_log_filenames(config, new_log_root: str = "/applog"):
    handlers = config.get("handlers", {})
    for handler_name, handler_cfg in handlers.items():
        filename = handler_cfg.get("filename")
        if filename:
            handler_cfg["filename"] = os.path.join(new_log_root, filename)
    return config


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


def update_path_inside_cc_config(yaml_path, config_key: str):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    package_path = config.get(config_key)
    if package_path:
        config[config_key] = to_abs_path(yaml_path, package_path)
    with open(yaml_path, "w") as f:
        yaml.dump(config, f)


def run_command(command, cwd=None):
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,  # Raises CalledProcessError on non-zero return
        )
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        return result.stdout

    except subprocess.CalledProcessError as e:
        print(f"Process failed with return code {e.returncode}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        raise


class OnPremPackager(Packager):
    def __init__(self, cc_config_key="cc_config", build_image_cmd=BUILD_IMAGE_CMD):
        super().__init__()
        self.cc_config_key = cc_config_key
        self.build_image_cmd = build_image_cmd

    def _build_cc_image(self, cc_config_yaml: str, site_name: str, startup_folder_path: str):
        """Build CC image for the site."""
        cc_config_yaml = os.path.abspath(cc_config_yaml)
        update_path_inside_cc_config(cc_config_yaml, config_key="nvflare_package")
        build_image_cmd = to_abs_path(cc_config_yaml, self.build_image_cmd)
        if not os.path.exists(build_image_cmd) or not os.access(build_image_cmd, os.X_OK):
            raise FileNotFoundError(f"Build image command '{build_image_cmd}' not found or is not executable.")
        command = [build_image_cmd, cc_config_yaml, site_name, startup_folder_path]
        output = run_command(command)
        tar_file_path = None
        for line in output.splitlines():
            if line.startswith("CVM_BUNDLE_PATH="):
                tar_file_path = line.split("=", 1)[1]
                break
        return tar_file_path

    def _change_log_dir(self, log_config_path: str):
        with open(log_config_path, "r") as f:
            config = json.load(f)

        updated_config = update_log_filenames(config)

        with open(log_config_path, "w") as f:
            json.dump(updated_config, f, indent=4)

    def _package_for_participant(self, participant: Participant, ctx: ProvisionContext):
        """Package the startup kit for the participant."""
        if not participant.get_prop(self.cc_config_key):
            return

        dest_dir = Path(ctx.get_result_location())

        log_config_path = dest_dir / participant.name / "local" / ProvFileName.LOG_CONFIG_DEFAULT
        self._change_log_dir(log_config_path)

        # Build CC image
        tar_file_path = self._build_cc_image(participant.get_prop(self.cc_config_key), participant.name, str(dest_dir))

        # Copy the package that is generated by the build_image_cmd
        site_dir = dest_dir / participant.name
        shutil.rmtree(site_dir)
        os.mkdir(site_dir)
        shutil.copy(tar_file_path, site_dir)

    def package(self, project: Project, ctx: ProvisionContext):
        for p in project.get_all_participants():
            self._package_for_participant(p, ctx)
