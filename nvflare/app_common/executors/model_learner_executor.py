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
import threading

from nvflare.apis.dxo import MetaKey
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.abstract.model_learner import ModelLearner
from nvflare.app_common.app_constant import AppConstants, ValidateType
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.fuel.utils.validation_utils import check_object_type
from nvflare.security.logging import secure_format_exception


class ModelLearnerExecutor(Executor):
    def __init__(
        self,
        learner_id,
        train_task=AppConstants.TASK_TRAIN,
        submit_model_task=AppConstants.TASK_SUBMIT_MODEL,
        validate_task=AppConstants.TASK_VALIDATION,
        configure_task=AppConstants.TASK_CONFIGURE,
    ):
        """Key component to run learner on clients.

        Args:
            learner_id (str): id of the learner object
            train_task (str, optional): task name for train. Defaults to AppConstants.TASK_TRAIN.
            submit_model_task (str, optional): task name for submit model. Defaults to AppConstants.TASK_SUBMIT_MODEL.
            validate_task (str, optional): task name for validation. Defaults to AppConstants.TASK_VALIDATION.
            configure_task (str, optional): task name for configure. Defaults to AppConstants.TASK_CONFIGURE.
        """
        super().__init__()
        self.learner_id = learner_id
        self.learner = None
        self.learner_name = ""
        self.is_initialized = False
        self.learner_exe_lock = threading.Lock()  # used ensure only one execution at a time

        self.task_funcs = {
            train_task: self.train,
            submit_model_task: self.submit_model,
            validate_task: self.validate,
            configure_task: self.configure,
        }

    def _abort(self, fl_ctx: FLContext):
        self.learner.fl_ctx = fl_ctx
        self.learner.abort()

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self._create_learner(fl_ctx)
        elif event_type == EventType.ABORT_TASK:
            try:
                if self.learner:
                    if not self.unsafe:
                        self._abort(fl_ctx)
                    else:
                        self.log_warning(fl_ctx, f"skipped abort of unsafe learner {self.learner_name}")
            except Exception as e:
                self.log_exception(fl_ctx, f"learner abort exception: {secure_format_exception(e)}")
        elif event_type == EventType.END_RUN:
            if not self.unsafe:
                self.finalize(fl_ctx)
            elif self.learner:
                self.log_warning(fl_ctx, f"skipped finalize of unsafe learner {self.learner_name}")

    def _create_learner(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        self.learner = engine.get_component(self.learner_id)
        if self.learner:
            self.learner_name = self.learner.__class__.__name__

        check_object_type("learner", self.learner, ModelLearner)
        self.log_info(fl_ctx, f"Got learner: {self.learner_name}")

    def initialize(self, fl_ctx: FLContext):
        try:
            engine = fl_ctx.get_engine()
            self.learner.fl_ctx = fl_ctx
            self.learner.engine = engine
            self.learner.initialize()
        except Exception as e:
            self.log_exception(fl_ctx, f"initialize error from {self.learner_name}: {secure_format_exception(e)}")
            raise e

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        # Do one task at a time since the shareable and fl_ctx are kept in "self".
        with self.learner_exe_lock:
            return self._do_execute(task_name, shareable, fl_ctx, abort_signal)

    def _do_execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self.log_info(fl_ctx, f"Client trainer got task: {task_name}")
        self._setup_learner(self.learner, shareable, fl_ctx, abort_signal)

        if not self.is_initialized:
            self.is_initialized = True
            self.initialize(fl_ctx)

        task_func = self.task_funcs.get(task_name)
        if task_func is not None:
            return task_func(shareable, fl_ctx)
        else:
            self.log_error(fl_ctx, f"Unknown task: {task_name}")
            return make_reply(ReturnCode.TASK_UNKNOWN)

    @staticmethod
    def _setup_learner(learner: ModelLearner, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal):
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
            learner.workspace_root = learner.workspace.get_root_dir()
            learner.job_root = learner.workspace.get_run_dir(learner.job_id)
            learner.app_root = learner.workspace.get_app_dir(learner.job_id)

        if shareable:
            learner.current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
            learner.total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS)

    def train(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        try:
            shareable.set_header(AppConstants.VALIDATE_TYPE, ValidateType.BEFORE_TRAIN_VALIDATE)
            model = FLModelUtils.from_shareable(shareable)
        except ValueError:
            self.log_error(fl_ctx, "request does not contain DXO")
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        try:
            val_result = self.learner.validate(model)
        except Exception as e:
            self.log_exception(
                fl_ctx, f"Learner {self.learner_name} failed to pretrain validate: {secure_format_exception(e)}"
            )
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        if isinstance(val_result, str):
            # this is an error code!
            self.log_warning(fl_ctx, f"Learner {self.learner_name}: pretrain validate failed: {val_result}")
            val_result = None

        if val_result:
            if not isinstance(val_result, FLModel):
                self.log_warning(
                    fl_ctx,
                    f"Learner {self.learner_name}: pretrain validate: expect FLModel but got {type(val_result)}",
                )
                val_result = None
            elif not val_result.metrics:
                self.log_warning(
                    fl_ctx,
                    f"Learner {self.learner_name}: pretrain validate: no metrics",
                )
                val_result = None

        try:
            train_result = self.learner.train(model)
        except Exception as e:
            self.log_exception(fl_ctx, f"Learner {self.learner_name} failed to train: {secure_format_exception(e)}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        if isinstance(train_result, str):
            # this is an error code!
            return make_reply(train_result)

        if not isinstance(train_result, FLModel):
            self.log_error(
                fl_ctx,
                f"Learner {self.learner_name}: bad result from train: expect FLModel but got {type(train_result)}",
            )
            return make_reply(ReturnCode.EMPTY_RESULT)

        # if the learner returned the valid BEFORE_TRAIN_VALIDATE result, set the INITIAL_METRICS in
        # the train result, which can be used for best model selection.
        if val_result:
            FLModelUtils.set_meta_prop(
                model=train_result,
                key=MetaKey.INITIAL_METRICS,
                value=val_result.metrics,
            )

        return FLModelUtils.to_shareable(train_result)

    def submit_model(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        model_name = shareable.get_header(AppConstants.SUBMIT_MODEL_NAME)
        try:
            result = self.learner.get_model(model_name)
        except Exception as e:
            self.log_exception(fl_ctx, f"Learner {self.learner_name} failed to get_model: {secure_format_exception(e)}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        if isinstance(result, str):
            self.log_error(fl_ctx, f"Learner {self.learner_name} failed to get_model: {result}")
            return make_reply(result)

        if isinstance(result, FLModel):
            return FLModelUtils.to_shareable(result)
        else:
            self.log_error(
                fl_ctx,
                f"Learner {self.learner_name} bad result from get_model: expect DXO but got {type(result)}",
            )
            return make_reply(ReturnCode.EMPTY_RESULT)

    def validate(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        try:
            model = FLModelUtils.from_shareable(shareable)
        except ValueError:
            self.log_error(fl_ctx, "request does not contain valid model")
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        try:
            result = self.learner.validate(model)
        except Exception as e:
            self.log_exception(fl_ctx, f"Learner {self.learner_name} failed to validate: {secure_format_exception(e)}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        if isinstance(result, str):
            self.log_error(fl_ctx, f"Learner {self.learner_name} failed to validate: {result}")
            return make_reply(result)

        if isinstance(result, FLModel):
            return FLModelUtils.to_shareable(result)
        else:
            self.log_error(
                fl_ctx, f"Learner {self.learner_name}: bad result from validate: expect FLModel but got {type(result)}"
            )
            return make_reply(ReturnCode.EMPTY_RESULT)

    def configure(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        try:
            model = FLModelUtils.from_shareable(shareable)
        except ValueError:
            self.log_error(fl_ctx, "request does not contain valid model data")
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        rc = ReturnCode.OK
        try:
            self.learner.configure(model)
        except Exception as e:
            self.log_exception(fl_ctx, f"Learner {self.learner_name} failed to configure: {secure_format_exception(e)}")
            rc = ReturnCode.EXECUTION_EXCEPTION

        return make_reply(rc)

    def finalize(self, fl_ctx: FLContext):
        try:
            self.learner.fl_ctx = fl_ctx
            self.learner.finalize()
        except Exception as e:
            self.log_exception(fl_ctx, f"learner finalize exception: {secure_format_exception(e)}")
