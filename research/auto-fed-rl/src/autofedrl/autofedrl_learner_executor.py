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

from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.executors.learner_executor import LearnerExecutor
from nvflare.security.logging import secure_format_exception

from .autofedrl_constants import AutoFedRLConstants


class AutoFedRLLearnerExecutor(LearnerExecutor):
    def __init__(
        self,
        learner_id,
        train_task=AppConstants.TASK_TRAIN,
        submit_model_task=AppConstants.TASK_SUBMIT_MODEL,
        validate_task=AppConstants.TASK_VALIDATION,
        validate_for_search_task=AutoFedRLConstants.MODEL_VALIDATE_FOR_SEARCH,
    ):
        """Key component to run learner on clients for Auto-FedRL algorithm (https://arxiv.org/abs/2203.06338).

        Args:
            learner_id (str): id pointing to the learner object
            train_task (str, optional): label to dispatch train task. Defaults to AppConstants.TASK_TRAIN.
            submit_model_task (str, optional): label to dispatch submit model task. Defaults to AppConstants.TASK_SUBMIT_MODEL.
            validate_task (str, optional): label to dispatch validation task. Defaults to AppConstants.TASK_VALIDATION.
            validate_for_search_task (str, optional): label to dispatch validate model for hyperparameter search.
        """
        super().__init__(
            learner_id=learner_id,
            train_task=train_task,
            submit_model_task=submit_model_task,
            validate_task=validate_task,
        )
        self.validate_for_search_task = validate_for_search_task

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """Same as LearnerExecutor.execute() apart for additional support for an `validate_for_search_task`."""
        self.log_info(fl_ctx, f"Client trainer got task: {task_name}")

        if not self.is_initialized:
            self.is_initialized = True
            self.initialize(fl_ctx)

        try:
            if task_name == self.train_task:
                return self.train(shareable, fl_ctx, abort_signal)
            elif task_name == self.submit_model_task:
                return self.submit_model(shareable, fl_ctx)
            elif task_name == self.validate_task:
                return self.validate(shareable, fl_ctx, abort_signal)
            elif task_name == self.validate_for_search_task:
                return self.validate_for_search(shareable, fl_ctx, abort_signal)
            else:
                self.log_error(fl_ctx, f"Could not handle task: {task_name}")
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except Exception as e:
            # Task execution error, return EXECUTION_EXCEPTION Shareable
            self.log_exception(fl_ctx, f"learner execute exception: {secure_format_exception(e)}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def validate_for_search(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self.log_debug(fl_ctx, f"validate for search abort_signal {abort_signal.triggered}")

        shareable.set_header(AppConstants.VALIDATE_TYPE, AutoFedRLConstants.MODEL_VALIDATE_FOR_SEARCH)
        validate_result: Shareable = self.learner.validate(shareable, fl_ctx, abort_signal)
        if validate_result and isinstance(validate_result, Shareable):
            return validate_result
        else:
            return make_reply(ReturnCode.EMPTY_RESULT)
