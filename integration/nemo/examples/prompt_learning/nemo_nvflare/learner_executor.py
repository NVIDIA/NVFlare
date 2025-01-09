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

from nvflare.apis.dxo import from_shareable
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.executors.learner_executor import LearnerExecutor

from .constants import NemoConstants, NemoDataKind


class NemoLearnerExecutor(LearnerExecutor):
    def __init__(
        self,
        learner_id,
        train_task=AppConstants.TASK_TRAIN,
        submit_model_task=AppConstants.TASK_SUBMIT_MODEL,
        validate_task=AppConstants.TASK_VALIDATION,
        share_config_task=NemoConstants.TASK_SHARE_CONFIG,
    ):
        """Key component to run learner on clients.

        Args:
            learner_id (str): id of the learner object
            train_task (str, optional): task name for train. Defaults to AppConstants.TASK_TRAIN.
            submit_model_task (str, optional): task name for submit model. Defaults to AppConstants.TASK_SUBMIT_MODEL.
            validate_task (str, optional): task name for validation. Defaults to AppConstants.TASK_VALIDATION.
            share_config_task (str, optional): share config task name.
        """
        super().__init__(
            learner_id=learner_id,
            train_task=train_task,
            submit_model_task=submit_model_task,
            validate_task=validate_task,
        )
        self.share_config_task = share_config_task
        self.is_initialized = False

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if not self.is_initialized:
            self.is_initialized = True
            self.initialize(fl_ctx)

        if task_name == self.share_config_task:
            self.log_info(fl_ctx, f"Client trainer got task: {task_name}")
            try:
                return self._set_learner_configs(shareable, fl_ctx, abort_signal)
            except Exception as e:
                self.log_error(fl_ctx, f"Setting config failed with exception {e}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        else:
            return super().execute(task_name=task_name, shareable=shareable, fl_ctx=fl_ctx, abort_signal=abort_signal)

    def _set_learner_configs(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        dxo = from_shareable(shareable)

        if dxo.data_kind != NemoDataKind.CONFIGS:
            raise ValueError(f"Expected DXO data to be of kind NemoDataKind.CONFIGS but got {dxo.data_kind}")

        if not dxo.data:
            raise ValueError("Received config data is empty!")

        self.learner.set_configs(configs=dxo.data)
        self.log_info(fl_ctx, f"Received config with {len(dxo.data)} entries from server.")

        return make_reply(ReturnCode.OK)
