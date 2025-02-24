# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from typing import List, Optional, Union

import yaml

import nvflare.lighter as prov
from nvflare.lighter import utils
from nvflare.lighter.utils import load_yaml

from .constants import CtxKey, PropKey, ProvisionMode
from .entity import Entity, Project


class ProvisionContext(dict):
    def __init__(self, workspace_root_dir: str, project: Project):
        super().__init__()
        self[CtxKey.WORKSPACE] = workspace_root_dir

        wip_dir = os.path.join(workspace_root_dir, "wip")
        state_dir = os.path.join(workspace_root_dir, "state")
        self.update({CtxKey.WIP: wip_dir, CtxKey.STATE: state_dir})
        dirs = [workspace_root_dir, wip_dir, state_dir]
        utils.make_dirs(dirs)

        # set commonly used data into ctx
        self[CtxKey.PROJECT] = project

        server = project.get_server()
        admin_port = server.get_prop(PropKey.ADMIN_PORT, 8003)
        self[CtxKey.ADMIN_PORT] = admin_port
        fed_learn_port = server.get_prop(PropKey.FED_LEARN_PORT, 8002)
        self[CtxKey.FED_LEARN_PORT] = fed_learn_port
        self[CtxKey.SERVER_NAME] = server.name
        self[CtxKey.TEMP_FILES_LOADED] = []
        self[CtxKey.TEMPLATE] = {}

    def get_project(self) -> Project:
        return self.get(CtxKey.PROJECT)

    def load_templates(self, temp_files: Union[str, List[str]]):
        if isinstance(temp_files, str):
            temp_files = [temp_files]
        elif not isinstance(temp_files, list):
            raise ValueError(f"temp_files must be str or List[str] but got {type(temp_files)}")

        prov_folder = os.path.dirname(prov.__file__)
        temp_folder = os.path.join(prov_folder, "templates")

        loaded = self[CtxKey.TEMP_FILES_LOADED]
        template = self[CtxKey.TEMPLATE]
        for f in temp_files:
            if f not in loaded:
                template.update(load_yaml(os.path.join(temp_folder, f)))
                loaded.append(f)

    def get_template_section(self, section_key: str):
        template = self.get(CtxKey.TEMPLATE)
        if not template:
            raise RuntimeError("template is not available")

        section = template.get(section_key)
        if not section:
            raise RuntimeError(f"missing section {section_key} in template")

        return section

    def set_provision_mode(self, mode: str):
        valid_modes = [ProvisionMode.POC, ProvisionMode.NORMAL]
        if mode not in valid_modes:
            raise ValueError(f"invalid provision mode {mode}: must be one of {valid_modes}")
        self[CtxKey.PROVISION_MODE] = mode

    def get_provision_mode(self):
        return self.get(CtxKey.PROVISION_MODE)

    def set_logger(self, logger):
        self[CtxKey.LOGGER] = logger

    def get_logger(self):
        return self.get(CtxKey.LOGGER)

    def get_wip_dir(self):
        return self.get(CtxKey.WIP)

    def get_ws_dir(self, entity: Entity):
        return os.path.join(self.get_wip_dir(), entity.name)

    def get_kit_dir(self, entity: Entity):
        return os.path.join(self.get_ws_dir(entity), "startup")

    def get_transfer_dir(self, entity: Entity):
        return os.path.join(self.get_ws_dir(entity), "transfer")

    def get_local_dir(self, entity: Entity):
        return os.path.join(self.get_ws_dir(entity), "local")

    def get_state_dir(self):
        return self.get(CtxKey.STATE)

    def get_workspace(self):
        return self.get(CtxKey.WORKSPACE)

    def yaml_load_template_section(self, section_key: str):
        return yaml.safe_load(self.get_template_section(section_key))

    def json_load_template_section(self, section_key: str):
        return json.loads(self.get_template_section(section_key))

    def build_from_template(
        self,
        dest_dir: str,
        temp_section: Union[str, List[str]],
        file_name,
        replacement=None,
        mode="t",
        exe=False,
        content_modify_cb=None,
        **cb_kwargs,
    ):
        """Build a file from a template section and writes it to the specified location.

        Args:
            dest_dir: destination directory
            temp_section: template section key
            file_name: file name
            replacement: replacement dict
            mode: file mode
            exe: executable
            content_modify_cb: content modification callback. If specified, it takes the section content as the
                first argument and returns the modified content
            cb_kwargs: additional keyword arguments for the callback

        """
        section = self.build_section_from_template(temp_section, replacement, content_modify_cb, **cb_kwargs)
        utils.write(os.path.join(dest_dir, file_name), section, mode, exe=exe)

    def build_section_from_template(
        self,
        temp_section: Union[str, List[str]],
        replacement=None,
        content_modify_cb=None,
        **cb_kwargs,
    ):
        if isinstance(temp_section, str):
            temp_section = [temp_section]
        elif not isinstance(temp_section, list):
            raise ValueError(f"temp_section must be str or List[str] but got {type(temp_section)}")

        section = ""
        for s in temp_section:
            section += self.get_template_section(s)

        if replacement:
            section = utils.sh_replace(section, replacement)
        if content_modify_cb:
            section = content_modify_cb(section, **cb_kwargs)
        return section

    def info(self, msg: str):
        logger = self.get_logger()
        if logger:
            logger.info(msg)
        else:
            print(f"INFO: {msg}")

    def error(self, msg: str):
        logger = self.get_logger()
        if logger:
            logger.error(msg)
        else:
            print(f"ERROR: {msg}")

    def debug(self, msg: str):
        logger = self.get_logger()
        if logger:
            logger.debug(msg)
        else:
            print(f"DEBUG: {msg}")

    def warning(self, msg: str):
        logger = self.get_logger()
        if logger:
            logger.warning(msg)
        else:
            print(f"WARNING: {msg}")

    def get_result_location(self) -> Optional[str]:
        """Get the directory of the provision result.
        This should be called after the provision is done.

        Returns: the name of the directory that holds the provisioned result.

        """
        return self.get(CtxKey.CURRENT_PROD_DIR)
