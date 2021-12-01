# Copyright (c) 2021, NVIDIA CORPORATION.
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
import os.path

from nvflare.apis.dxo import DataKind, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType


class ValidationJsonGenerator(FLComponent):
    def __init__(self, results_dir=AppConstants.CROSS_VAL_DIR, json_file_name="cross_site_val.json"):
        """A class to generate a json file with cross-site validation results.

        Args:
            results_dir: directory where to save the json with cross-site validation results.
            json_file_name: filename of the json to be saved.

        """
        super(ValidationJsonGenerator, self).__init__()

        self.results_dir = results_dir
        self.val_results = {}
        self.json_file_name = json_file_name

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.val_results.clear()
        elif event_type == AppEventType.VALIDATION_RESULT_RECEIVED:
            model_owner = fl_ctx.get_prop(AppConstants.MODEL_OWNER, None)
            data_client = fl_ctx.get_prop(AppConstants.DATA_CLIENT, None)
            val_results = fl_ctx.get_prop(AppConstants.VALIDATION_RESULT, None)

            if not model_owner:
                self.log_error(fl_ctx, "model_owner unknown. Validation result will not be saved to json")
            if not data_client:
                self.log_error(fl_ctx, "data_client unknown. Validation result will not be saved to json")

            if val_results:
                try:
                    dxo = from_shareable(val_results)
                    dxo.validate()

                    if dxo.data_kind == DataKind.METRICS:
                        if data_client not in self.val_results:
                            self.val_results[data_client] = {}
                        self.val_results[data_client][model_owner] = dxo.data
                    else:
                        self.log_error(fl_ctx, f"Expected dxo of kind METRICS but got {dxo.data_kind} instead.")
                except:
                    self.log_exception(fl_ctx, "Exception in handling validation result.")
            else:
                self.log_error(fl_ctx, "Validation result not found.")
        elif event_type == EventType.END_RUN:
            run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_run_number())
            cross_val_res_dir = os.path.join(run_dir, self.results_dir)
            if not os.path.exists(cross_val_res_dir):
                os.makedirs(cross_val_res_dir)

            res_file_path = os.path.join(cross_val_res_dir, self.json_file_name)
            with open(res_file_path, "w") as f:
                json.dump(self.val_results, f, indent=4)
