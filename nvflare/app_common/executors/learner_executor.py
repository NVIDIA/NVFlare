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
import threading

from nvflare.apis.dxo import DXO, MetaKey, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learner2 import Learner2
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.app_constant import AppConstants, ValidateType
from nvflare.security.logging import secure_format_exception


class LearnerExecutor(Executor):
    def __init__(
        self,
        learner_id,
        train_task=AppConstants.TASK_TRAIN,
        submit_model_task=AppConstants.TASK_SUBMIT_MODEL,
        validate_task=AppConstants.TASK_VALIDATION,
    ):
        """Key component to run learner on clients.

        Args:
            learner_id (str): id of the learner object
            train_task (str, optional): task name for train. Defaults to AppConstants.TASK_TRAIN.
            submit_model_task (str, optional): task name for submit model. Defaults to AppConstants.TASK_SUBMIT_MODEL.
            validate_task (str, optional): task name for validation. Defaults to AppConstants.TASK_VALIDATION.
        """
        super().__init__()
        self.learner_id = learner_id
        self.learner = None
        self.learner_name = ""
        self.train_task = train_task
        self.submit_model_task = submit_model_task
        self.validate_task = validate_task
        self.is_initialized = False
        self.learner_exe_lock = threading.Lock()  # used ensure only one execution at a time

    def _abort_learner(self, fl_ctx: FLContext):
        if isinstance(self.learner, Learner):
            self.learner.abort(fl_ctx)
        elif isinstance(self.learner, Learner2):
            self.learner.fl_ctx = fl_ctx
            self.learner.abort()

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self._create_learner(fl_ctx)
        elif event_type == EventType.ABORT_TASK:
            try:
                if self.learner:
                    if not self.unsafe:
                        self._abort_learner(fl_ctx)
                    else:
                        self.log_warning(fl_ctx, f"skipped abort of unsafe learner {self.learner.__class__.__name__}")
            except Exception as e:
                self.log_exception(fl_ctx, f"learner abort exception: {secure_format_exception(e)}")
        elif event_type == EventType.END_RUN:
            if not self.unsafe:
                self.finalize(fl_ctx)
            elif self.learner:
                self.log_warning(fl_ctx, f"skipped finalize of unsafe learner {self.learner.__class__.__name__}")

    def _create_learner(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        self.learner = engine.get_component(self.learner_id)
        if self.learner:
            self.learner_name = self.learner.__class__.__name__

        if not isinstance(self.learner, (Learner, Learner2)):
            raise TypeError(f"learner must be Learner or Learner2 type, but got: {type(self.learner)}")

        self.log_info(fl_ctx, f"Got learner: {self.learner_name}")

    def initialize(self, fl_ctx: FLContext):
        try:
            engine = fl_ctx.get_engine()
            if isinstance(self.learner, Learner):
                self.learner.initialize(engine.get_all_components(), fl_ctx)
            elif isinstance(self.learner, Learner2):
                self.learner.fl_ctx = fl_ctx
                self.learner.initialize(engine.get_all_components())
            else:
                raise TypeError(f"learner must be Learner or Learner2 type, but got: {type(self.learner)}")
        except Exception as e:
            self.log_exception(fl_ctx, f"learner initialize exception: {secure_format_exception(e)}")
            raise e

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if isinstance(self.learner, Learner):
            return self._do_execute(task_name, shareable, fl_ctx, abort_signal)

        # For Learner2, do one task at a time since the shareable and fl_ctx are kept in "self".
        with self.learner_exe_lock:
            return self._do_execute(task_name, shareable, fl_ctx, abort_signal)

    def _do_execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self.log_info(fl_ctx, f"Client trainer got task: {task_name}")
        if isinstance(self.learner, Learner2):
            self._setup_learner2(self.learner, shareable, fl_ctx, abort_signal)

        if not self.is_initialized:
            self.is_initialized = True
            self.initialize(fl_ctx)

        if task_name == self.train_task:
            if isinstance(self.learner, Learner):
                return self.train(shareable, fl_ctx, abort_signal)
            elif isinstance(self.learner, Learner2):
                return self.train2(shareable, fl_ctx)
            else:
                self.log_error(fl_ctx, "Learner not configured")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        elif task_name == self.submit_model_task:
            if isinstance(self.learner, Learner):
                return self.submit_model(shareable, fl_ctx)
            elif isinstance(self.learner, Learner2):
                return self.submit_model2(shareable, fl_ctx)
            else:
                self.log_error(fl_ctx, "Learner not configured")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        elif task_name == self.validate_task:
            if isinstance(self.learner, Learner):
                return self.validate(shareable, fl_ctx, abort_signal)
            elif isinstance(self.learner, Learner2):
                return self.validate2(shareable, fl_ctx)
            else:
                self.log_error(fl_ctx, "Learner not configured")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        else:
            self.log_error(fl_ctx, f"Could not handle task: {task_name}")
            return make_reply(ReturnCode.TASK_UNKNOWN)

    def train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self.log_debug(fl_ctx, f"train abort signal: {abort_signal.triggered}")
        shareable.set_header(AppConstants.VALIDATE_TYPE, ValidateType.BEFORE_TRAIN_VALIDATE)
        validate_result: Shareable = self.learner.validate(shareable, fl_ctx, abort_signal)

        train_result = self.learner.train(shareable, fl_ctx, abort_signal)
        if not (train_result and isinstance(train_result, Shareable)):
            return make_reply(ReturnCode.EMPTY_RESULT)

        # if the learner returned the valid BEFORE_TRAIN_VALIDATE result, set the INITIAL_METRICS in
        # the train result, which can be used for best model selection.
        if (
            validate_result
            and isinstance(validate_result, Shareable)
            and validate_result.get_return_code() == ReturnCode.OK
        ):
            try:
                metrics_dxo = from_shareable(validate_result)
                train_dxo = from_shareable(train_result)
                train_dxo.meta[MetaKey.INITIAL_METRICS] = metrics_dxo.data.get(MetaKey.INITIAL_METRICS, 0)
                return train_dxo.to_shareable()
            except ValueError:
                return train_result
        else:
            return train_result

    def submit_model(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        model_name = shareable.get_header(AppConstants.SUBMIT_MODEL_NAME)
        submit_model_result = self.learner.get_model_for_validation(model_name, fl_ctx)
        if submit_model_result and isinstance(submit_model_result, Shareable):
            return submit_model_result
        else:
            return make_reply(ReturnCode.EMPTY_RESULT)

    def validate(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self.log_debug(fl_ctx, f"validate abort_signal {abort_signal.triggered}")

        shareable.set_header(AppConstants.VALIDATE_TYPE, ValidateType.MODEL_VALIDATE)
        validate_result: Shareable = self.learner.validate(shareable, fl_ctx, abort_signal)
        if validate_result and isinstance(validate_result, Shareable):
            return validate_result
        else:
            return make_reply(ReturnCode.EMPTY_RESULT)

    @staticmethod
    def _setup_learner2(learner: Learner2, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal):
        learner.shareable = shareable
        learner.fl_ctx = fl_ctx
        learner.abort_signal = abort_signal
        if not learner.args:
            learner.args = fl_ctx.get_prop(FLContextKey.ARGS)

        if not learner.site_name:
            learner.site_name = fl_ctx.get_identity_name()

        if not learner.job_id:
            learner.job_id = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)

        if not learner.engine:
            learner.engine = fl_ctx.get_engine()
            learner.workspace = learner.engine.get_workspace()
            learner.app_root = learner.workspace.get_run_dir(learner.job_id)

        if shareable:
            learner.current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
            learner.total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS)

    def train2(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        try:
            dxo = from_shareable(shareable)
        except ValueError:
            self.log_error(fl_ctx, "request does not contain DXO")
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        val_result = self.learner.validate_before_train(dxo)
        if isinstance(val_result, str):
            # this is an error code!
            self.log_warning(fl_ctx, f"Learner {self.learner_name}: validate_before_train failed: {val_result}")
            val_result = None

        if val_result and not isinstance(val_result, DXO):
            self.log_warning(
                fl_ctx,
                f"Learner {self.learner_name}: bad result from validate_before_train: expect DXO but got {type(val_result)}",
            )
            val_result = None

        train_result = self.learner.train(dxo)
        if isinstance(train_result, str):
            # this is an error code!
            return make_reply(train_result)

        if not isinstance(train_result, DXO):
            self.log_error(
                fl_ctx, f"Learner {self.learner_name}: bad result from train: expect DXO but got {type(train_result)}"
            )
            return make_reply(ReturnCode.EMPTY_RESULT)

        # if the learner returned the valid BEFORE_TRAIN_VALIDATE result, set the INITIAL_METRICS in
        # the train result, which can be used for best model selection.
        if val_result:
            train_result.meta[MetaKey.INITIAL_METRICS] = val_result.data.get(MetaKey.INITIAL_METRICS, 0)
        return train_result.to_shareable()

    def submit_model2(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        model_name = shareable.get_header(AppConstants.SUBMIT_MODEL_NAME)
        result = self.learner.get_model_for_validation(model_name)
        if isinstance(result, str):
            self.log_error(fl_ctx, f"Learner {self.learner_name} failed to get_model_for_validation: {result}")
            return make_reply(result)

        if isinstance(result, DXO):
            return result.to_shareable()
        else:
            self.log_error(
                fl_ctx,
                f"Learner {self.learner_name} bad result from get_model_for_validation: expect DXO but got {type(result)}",
            )
            return make_reply(ReturnCode.EMPTY_RESULT)

    def validate2(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        try:
            dxo = from_shareable(shareable)
        except ValueError:
            self.log_error(fl_ctx, "request does not contain DXO")
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        model_owner = shareable.get(ReservedHeaderKey.HEADERS).get(AppConstants.MODEL_OWNER)
        if model_owner:
            self.log_info(fl_ctx, f"Evaluating model from {model_owner} on {fl_ctx.get_identity_name()}")
        else:
            model_owner = "global_model"  # evaluating global model during training

        validate_type = shareable.get_header(AppConstants.VALIDATE_TYPE)
        if validate_type == ValidateType.BEFORE_TRAIN_VALIDATE:
            result = self.learner.validate_before_train(dxo)
            method_name = "validate_before_train"
        elif validate_type == ValidateType.MODEL_VALIDATE:
            result = self.learner.validate(dxo, model_owner)
            method_name = "validate"
        else:
            return make_reply(ReturnCode.VALIDATE_TYPE_UNKNOWN)

        if isinstance(result, str):
            self.log_error(fl_ctx, f"Learner {self.learner_name} failed to {method_name}: {result}")
            return make_reply(result)

        if isinstance(result, DXO):
            return result.to_shareable()
        else:
            self.log_error(
                fl_ctx, f"Learner {self.learner_name}: bad result from {method_name}: expect DXO but got {type(result)}"
            )
            return make_reply(ReturnCode.EMPTY_RESULT)

    def finalize(self, fl_ctx: FLContext):
        try:
            if isinstance(self.learner, Learner):
                self.learner.finalize(fl_ctx)
            elif isinstance(self.learner, Learner2):
                self.learner.fl_ctx = fl_ctx
                self.learner.finalize()
        except Exception as e:
            self.log_exception(fl_ctx, f"learner finalize exception: {secure_format_exception(e)}")
