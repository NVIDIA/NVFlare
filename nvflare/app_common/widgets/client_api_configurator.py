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

import os
from typing import List

from nvflare.apis.client_api_exportable import ClientAPIExportable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.client.config import ClientConfig, ConfigKey, from_file
from nvflare.client.constants import CLIENT_API_CONFIG
from nvflare.fuel.utils.constants import Mode
from nvflare.fuel.utils.pipe.cell_pipe import CellPipe
from nvflare.fuel.utils.pipe.file_pipe import FilePipe
from nvflare.fuel.utils.pipe.pipe import Pipe
from nvflare.fuel.utils.validation_utils import check_object_type
from nvflare.widgets.widget import Widget

EXTERNAL_CLASS = "external_class"
EXTERNAL_ARGS_MAPPING = "external_args_mapping"


class ClientAPIConfigurator(Widget):
    """Prepares any external configuration files that the client api needs."""

    PIPE_EXTERNAL_MAPPINGS = {
        FilePipe: {
            EXTERNAL_CLASS: "FilePipe",
            EXTERNAL_ARGS_MAPPING: {
                "root_path": lambda pipe: pipe.root_path,
                "mode": lambda pipe: Mode.ACTIVE if pipe.mode == Mode.PASSIVE else Mode.PASSIVE,
            },
        },
        CellPipe: {
            EXTERNAL_CLASS: "CellPipe",
            EXTERNAL_ARGS_MAPPING: {
                "mode": lambda pipe: Mode.ACTIVE if pipe.mode == Mode.PASSIVE else Mode.PASSIVE,
                "site_name": lambda pipe: pipe.site_name,
                "token": lambda pipe: pipe.token,
                "root_url": lambda pipe: pipe.cell.get_root_url_for_child(),
                "secure_mode": lambda pipe: pipe.cell.core_cell.secure,
                "workspace_dir": lambda pipe: pipe.workspace_dir,
            },
        },
    }

    @classmethod
    def get_external_pipe_class(cls, pipe: Pipe) -> str:
        pipe_class = type(pipe)
        for pipe_type, info in cls.PIPE_EXTERNAL_MAPPINGS.items():
            if issubclass(pipe_class, pipe_type):
                return info[EXTERNAL_CLASS]
        raise RuntimeError(f"Pipe of type ({pipe_class}) is not supported")

    @classmethod
    def get_external_pipe_args(cls, pipe: Pipe) -> dict:
        pipe_class = type(pipe)
        for pipe_type, info in cls.PIPE_EXTERNAL_MAPPINGS.items():
            if issubclass(pipe_class, pipe_type):
                args_mapping = info[EXTERNAL_ARGS_MAPPING]
                return {arg: mapper(pipe) for arg, mapper in args_mapping.items()}
        raise RuntimeError(f"Pipe of type ({pipe_class}) is not supported")

    @classmethod
    def write_config_to_file(cls, config_data: dict, fl_ctx: FLContext, filename: str = CLIENT_API_CONFIG):
        """Writes client api config file into app_directory/config folder."""
        engine = fl_ctx.get_engine()
        workspace = engine.get_workspace()
        app_directory = workspace.get_app_dir(fl_ctx.get_job_id())
        config_file_path = os.path.join(app_directory, workspace.config_folder, filename)
        if os.path.exists(config_file_path):
            client_config = from_file(config_file=config_file_path)
        else:
            client_config = ClientConfig()
        configuration = client_config.config
        configuration[ConfigKey.SITE_NAME] = fl_ctx.get_identity_name()
        configuration[ConfigKey.JOB_ID] = fl_ctx.get_job_id()
        configuration.update(config_data)
        client_config.to_json(config_file_path)

    def __init__(self, component_ids: List[str]):
        super().__init__()
        check_object_type("component_ids", component_ids, list)

        # the components that needs to provide information to the client api
        self._component_ids = component_ids

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.ABOUT_TO_START_RUN:
            components_data = self._export_all_components_for_client_api(fl_ctx)
            ClientAPIConfigurator.write_config_to_file(config_data=components_data, fl_ctx=fl_ctx)

    def _export_all_components_for_client_api(self, fl_ctx: FLContext) -> dict:
        """Exports all components for client api."""
        engine = fl_ctx.get_engine()
        components = engine.get_all_components()
        components_data = {}
        for component_id in self._component_ids:
            component_instance: ClientAPIExportable = components.get(component_id)
            if component_instance is None:
                raise RuntimeError(f"missing component: {component_id}")
            component_attributes = component_instance.export_for_client_api()
            for k, v in component_attributes.items():
                if k in components_data:
                    raise RuntimeError(
                        f"key {k} from {component_id} is duplicated, please change export_for_client_api"
                    )
                if k in [ConfigKey.SITE_NAME, ConfigKey.JOB_ID]:
                    raise RuntimeError(
                        f"key {k} from {component_id} is a reserved key, please change export_for_client_api"
                    )
                components_data[k] = v
        return components_data
