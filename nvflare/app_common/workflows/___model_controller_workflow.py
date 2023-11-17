# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any

from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.security.logging import secure_format_exception
from .scatter_and_gather import ScatterAndGather
from .model_controller import ModelController


def _check_non_neg_int(data: Any, name: str):
    if not isinstance(data, int):
        raise ValueError(f"{name} must be int but got {type(data)}")

    if data < 0:
        raise ValueError(f"{name} must be greater than or equal to 0.")


class ___ModelControllerWorkflow(ScatterAndGather):
    def __init__(
        self,
        *args, model_controller_id, **kwargs
    ):
        """The controller for ModelControllerWorkflow Workflow.


        Args:


        Raises:
            TypeError: when any of input arguments does not have correct type
            ValueError: when any of input arguments is out of range
        """
        super().__init__(*args, **kwargs)
        self.model_controller_id = model_controller_id
        self.model_controller = None

    def start_controller(self, fl_ctx: FLContext) -> None:
        self.log_info(fl_ctx, "Initializing model controller workflow.")
        self._phase = AppConstants.PHASE_INIT

        self.model_controller = self._engine.get_component(self.model_controller_id)
        if not isinstance(self.model_controller, ModelController):
            self.system_panic(
                f"model controller {self.model_controller_id} must be an ModelController type object but got {type(self.model_controller)}",
                fl_ctx,
            )
            return

        # initialize the model controller
        try:
            self.model_controller.initialize()
        except Exception as e:
            error_msg = f"Exception when trying to initialize the model controller: {secure_format_exception(e)}"
            self.log_exception(fl_ctx, error_msg)
            self.system_panic(error_msg, fl_ctx)

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        try:

            self.log_info(fl_ctx, "Beginning model controller run.")
            self._phase = AppConstants.PHASE_TRAIN

            self.model_controller.run()
        except Exception as e:
            error_msg = f"Exception in model controller run: {secure_format_exception(e)}"
            self.log_exception(fl_ctx, error_msg)
            self.system_panic(error_msg, fl_ctx)

    def stop_controller(self, fl_ctx: FLContext):
        self._phase = AppConstants.PHASE_FINISHED
        self.model_controller.finalize()
