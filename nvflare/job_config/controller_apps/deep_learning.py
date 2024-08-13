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
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator
from nvflare.fuel.utils.import_utils import optional_import
from nvflare.job_config.api import ControllerApp

torch, torch_ok = optional_import(module="torch")
tb, tb_ok = optional_import(module="tensorboard")
if torch_ok and tb_ok:
    from nvflare.app_opt.tracking.tb.tb_receiver import TBAnalyticsReceiver


class DeepLearningControllerApp(ControllerApp):
    """Wrapper around `ServerAppConfig`.

    Args:
    """

    def __init__(self, key_metric="accuracy"):
        super().__init__()
        self.key_metric = key_metric
        self._create_server_app()

    def _create_server_app(self):
        component = ValidationJsonGenerator()
        self.app.add_component("json_generator", component)

        if self.key_metric:
            component = IntimeModelSelector(key_metric=self.key_metric)
            self.app.add_component("model_selector", component)

        # TODO: make different tracking receivers configurable
        if torch_ok and tb_ok:
            component = TBAnalyticsReceiver(events=["fed.analytix_log_stats"])
            self.app.add_component("receiver", component)
