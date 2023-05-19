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

import os

from monai.bundle.config_parser import ConfigParser

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.app_constant import StatisticsConstants as StC


class MonaiDataStatsPersistor(FLComponent):
    def __init__(self, fmt="yaml"):
        """Persist pytorch-based from MONAI bundle configuration.

        Args:
            fmt: format used to save the analysis results using MONAI's `ConfigParser`.
                Supported suffixes are "json", "yaml", "yml". Defaults to "yaml".

        Raises:
            ValueError: when source_ckpt_filename does not exist
        """
        super().__init__()

        self.fmt = fmt

    def handle_event(self, event: str, fl_ctx: FLContext):
        if event == EventType.PRE_RUN_RESULT_AVAILABLE:
            result = fl_ctx.get_prop(StC.PRE_RUN_RESULT)
            if result:
                for client_name, _result in result.items():
                    app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
                    output_path = os.path.join(app_root, f"{client_name}_data_stats.{self.fmt}")
                    ConfigParser.export_config_file(_result.data, output_path, fmt=self.fmt, default_flow_style=None)
                    self.log_info(fl_ctx, f"Saved data stats of client {client_name} at {output_path}")
            else:
                self.log_debug(fl_ctx, "Empty pre-task results.")
