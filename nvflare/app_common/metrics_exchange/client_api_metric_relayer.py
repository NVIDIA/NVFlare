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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.data_exchange.piper import Piper
from nvflare.client.config import ClientConfig, ConfigKey
from nvflare.client.constants import CONFIG_METRICS_EXCHANGE

from .metric_relayer import MetricRelayer


class ClientAPIMetricRelayer(MetricRelayer):
    def handle_event(self, event_type: str, fl_ctx: FLContext):
        super().handle_event(event_type, fl_ctx)
        if event_type == EventType.ABOUT_TO_START_RUN:
            self.prepare_external_config(fl_ctx)

    def prepare_external_config(self, fl_ctx: FLContext):
        workspace = fl_ctx.get_engine().get_workspace()
        app_dir = workspace.get_app_dir(fl_ctx.get_job_id())
        config_file = os.path.join(app_dir, workspace.config_folder, CONFIG_METRICS_EXCHANGE)

        # prepare config exchange for data exchanger
        client_config = ClientConfig()
        config_dict = client_config.config
        config_dict[ConfigKey.PIPE_CHANNEL_NAME] = self.pipe_channel_name
        config_dict[ConfigKey.PIPE_CLASS] = Piper.get_external_pipe_class(self.pipe_id, fl_ctx)
        config_dict[ConfigKey.PIPE_ARGS] = Piper.get_external_pipe_args(self.pipe_id, fl_ctx)
        client_config.to_json(config_file)
