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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLMetaKey
from nvflare.apis.fl_context import FLContext
from nvflare.client.config import write_config_to_file
from nvflare.client.constants import CLIENT_API_CONFIG
from nvflare.fuel.utils.attributes_exportable import ExportMode, export_components
from nvflare.fuel.utils.validation_utils import check_object_type
from nvflare.widgets.widget import Widget


class ExternalConfigurator(Widget):
    def __init__(
        self,
        component_ids: List[str],
        config_file_name: str = CLIENT_API_CONFIG,
    ):
        """Prepares any external configuration files.

        Args:
            component_ids: A list of components that are `AttributesExportable`
            config_file_name: The file name of the external config.
        """
        super().__init__()
        check_object_type("component_ids", component_ids, list)

        # the components that needs to export attributes
        self._component_ids = component_ids
        self._config_file_name = config_file_name

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.ABOUT_TO_START_RUN:
            components_data = self._export_all_components(fl_ctx)
            components_data[FLMetaKey.SITE_NAME] = fl_ctx.get_identity_name()
            components_data[FLMetaKey.JOB_ID] = fl_ctx.get_job_id()

            config_file_path = self._get_external_config_file_path(fl_ctx)
            write_config_to_file(config_data=components_data, config_file_path=config_file_path)

    def _get_external_config_file_path(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        workspace = engine.get_workspace()
        app_config_directory = workspace.get_app_config_dir(fl_ctx.get_job_id())
        config_file_path = os.path.join(app_config_directory, self._config_file_name)
        return config_file_path

    def _export_all_components(self, fl_ctx: FLContext) -> dict:
        """Exports all components."""
        engine = fl_ctx.get_engine()
        all_components = engine.get_all_components()
        components = {i: all_components.get(i) for i in self._component_ids}
        reserved_keys = [FLMetaKey.SITE_NAME, FLMetaKey.JOB_ID]
        return export_components(components=components, reserved_keys=reserved_keys, export_mode=ExportMode.PEER)
