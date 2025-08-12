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
import gc
import time

from nvflare.apis.controller_spec import Task
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_copy, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model import model_learnable_to_dxo
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.app_constant import AppConstants, ValidateType
from nvflare.app_common.ccwf.client_ctl import ClientSideController
from nvflare.app_common.ccwf.common import Constant, ModelType, make_task_name
from nvflare.fuel.utils.validation_utils import check_non_empty_str, check_positive_number
from nvflare.security.logging import secure_format_exception


class CrossSiteEvalClientController(ClientSideController):
    def __init__(
        self,
        task_name_prefix=Constant.TN_PREFIX_CROSS_SITE_EVAL,
        submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL,
        validation_task_name=AppConstants.TASK_VALIDATION,
        persistor_id=AppConstants.DEFAULT_PERSISTOR_ID,
        get_model_timeout=Constant.GET_MODEL_TIMEOUT,
    ):
        check_positive_number("get_model_timeout", get_model_timeout)
        check_non_empty_str("submit_model_task_name", submit_model_task_name)
        check_non_empty_str("validation_task_name", validation_task_name)
        check_non_empty_str("persistor_id", persistor_id)

        super().__init__(
            task_name_prefix=task_name_prefix,
            learn_task_name="",
            shareable_generator_id="",
            persistor_id=persistor_id,
        )
        self.eval_task_name = make_task_name(task_name_prefix, Constant.BASENAME_EVAL)
        self.prep_model_task_name = make_task_name(task_name_prefix, Constant.BASENAME_PREP_MODEL)
        self.ask_for_model_task_name = make_task_name(task_name_prefix, Constant.BASENAME_ASK_FOR_MODEL)
        self.submit_model_task_name = submit_model_task_name  # this is for the learner executor
        self.validation_task_name = validation_task_name
        self.my_local_model = None
        self.global_model_inventory = None
        self.submit_model_executor = None
        self.validate_executor = None
        self.inventory = None
        self.get_model_timeout = get_model_timeout
        self.prepared_models = {}  # model key => model shareable

    def start_run(self, fl_ctx: FLContext):
        super().start_run(fl_ctx)
        runner = fl_ctx.get_prop(FLContextKey.RUNNER)
        if self.submit_model_task_name:
            self.submit_model_executor = runner.find_executor(self.submit_model_task_name)
            if not self.submit_model_executor:
                self.system_panic(f"no executor for task {self.submit_model_task_name}", fl_ctx)
                return

        if self.validation_task_name:
            self.validate_executor = runner.find_executor(self.validation_task_name)
            if not self.validate_executor:
                self.system_panic(f"no executor for task {self.validation_task_name}", fl_ctx)
                return

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name == self.eval_task_name:
            # server assigned task
            result = self._do_eval(shareable, fl_ctx, abort_signal)
            gc.collect()
            return result

        elif task_name == self.prep_model_task_name:
            # server assigned task
            return self._prepare_model(shareable, fl_ctx, abort_signal)

        elif task_name == self.ask_for_model_task_name:
            # client-assigned task
            return self._process_get_model_request(shareable, fl_ctx)

        return super().execute(task_name, shareable, fl_ctx, abort_signal)

    def process_config(self, fl_ctx: FLContext):
        eval_local = self.get_config_prop(Constant.EVAL_LOCAL)
        eval_global = self.get_config_prop(Constant.EVAL_GLOBAL)
        evaluators = self.get_config_prop(Constant.EVALUATORS)
        evaluatees = self.get_config_prop(Constant.EVALUATEES)
        global_client = self.get_config_prop(Constant.GLOBAL_CLIENT)

        if eval_local and self.me in evaluatees:
            # do I have any local model?
            if not self.submit_model_executor:
                return make_reply(Constant.RC_NO_LOCAL_MODEL)

        if self.me in evaluators:
            # I am required to evaluate others
            if not self.validate_executor:
                return make_reply(Constant.RC_UNABLE_TO_EVAL)

        reply = make_reply(ReturnCode.OK)
        if eval_global and self.me == global_client:
            # do I have global models?
            assert isinstance(self.persistor, ModelPersistor)
            self.inventory = self.persistor.get_model_inventory(fl_ctx)
            if self.inventory:
                assert isinstance(self.inventory, dict)
                reply[Constant.GLOBAL_NAMES] = list(self.inventory.keys())
        return reply

    def start_workflow(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        pass

    @staticmethod
    def _model_key(model_type: str, model_name: str):
        return f"{model_type}::{model_name}"

    def _get_prepared_model(self, model_type: str, model_name: str):
        return self.prepared_models.get(self._model_key(model_type, model_name))

    def _set_prepared_model(self, model_type: str, model_name: str, model):
        self.prepared_models[self._model_key(model_type, model_name)] = model

    def _clear_prepared_models(self):
        self.prepared_models = {}

    def _do_eval(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        model_type = shareable.get(Constant.MODEL_TYPE)
        model_owner = shareable.get(Constant.MODEL_OWNER)
        model_name = shareable.get(Constant.MODEL_NAME)
        current_round = shareable.get(AppConstants.CURRENT_ROUND)

        # ask the model owner for model
        req = Shareable()
        req[Constant.MODEL_NAME] = model_name
        req[Constant.MODEL_TYPE] = model_type

        if not self.validate_executor:
            self.log_error(fl_ctx, "got eval request but I don't have a validator")
            return make_reply(Constant.RC_UNABLE_TO_EVAL)

        self.update_status(action=f"eval:get_model:{model_type}:{model_name}@{model_owner}", last_round=current_round)

        my_name = fl_ctx.get_identity_name()
        if my_name == model_owner:
            self.log_info(fl_ctx, f"use prepared model {model_type} {model_name} for validation")
            model_to_validate = self._get_prepared_model(model_type, model_name)
            if not model_to_validate:
                self.log_error(fl_ctx, f"no prepared model {model_type} {model_name} to validate for myself!")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            assert isinstance(model_to_validate, Shareable)
            model_to_validate = make_copy(model_to_validate)
        else:
            self.log_info(fl_ctx, f"asking client {model_owner} for model {model_type} {model_name}")

            task = Task(
                name=self.ask_for_model_task_name,
                data=req,
                timeout=int(self.get_model_timeout),
                secure=self.is_task_secure(fl_ctx),
            )

            resp = self.broadcast_and_wait(
                task=task,
                targets=[model_owner],
                min_responses=1,
                fl_ctx=fl_ctx,
            )

            assert isinstance(resp, dict)
            reply = resp.get(model_owner)
            if not reply:
                self.log_error(fl_ctx, f"failed to ask client {model_owner} for model")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            if not isinstance(reply, Shareable):
                self.log_error(
                    fl_ctx,
                    f"bad reply from {model_owner} for get-model request: expect Shareable but got {type(reply)}",
                )
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            rc = reply.get_return_code()
            if rc != ReturnCode.OK:
                self.log_error(fl_ctx, f"client {model_owner} failed to respond to get-model request: {rc=}")
                return make_reply(rc)

            if abort_signal.triggered:
                self.log_error(fl_ctx, "aborted get-model request - signal received")
                return make_reply(ReturnCode.TASK_ABORTED)

            model_to_validate = reply
            self.log_info(fl_ctx, f"got {model_type} {model_name} from client {model_owner}")

        model_to_validate.set_header(AppConstants.VALIDATE_TYPE, ValidateType.MODEL_VALIDATE)
        model_to_validate.set_header(FLContextKey.TASK_NAME, self.validation_task_name)
        if model_type == ModelType.LOCAL:
            model_to_validate.set_header(AppConstants.MODEL_OWNER, model_owner)

        self.update_status(action=f"eval:validate:{model_type}:{model_name}@{model_owner}", last_round=current_round)

        self.log_info(fl_ctx, f"invoking {type(self.validate_executor)} to do {self.validation_task_name}")
        start = time.time()
        result = self.validate_executor.execute(
            task_name=self.validation_task_name, shareable=model_to_validate, abort_signal=abort_signal, fl_ctx=fl_ctx
        )
        self.log_info(
            fl_ctx,
            f"{type(self.validate_executor)} finished {self.validation_task_name} in {time.time() - start} seconds",
        )

        assert isinstance(result, Shareable)
        result.set_header(Constant.MODEL_TYPE, model_type)
        result.set_header(Constant.MODEL_NAME, model_name)
        result.set_header(Constant.MODEL_OWNER, model_owner)
        result.set_header(AppConstants.CURRENT_ROUND, current_round)
        self.update_status(action=f"eval:finished:{model_type}:{model_name}@{model_owner}", last_round=current_round)
        return result

    def _prepare_global_model(self, model_name, fl_ctx: FLContext):
        # get it from model inventory
        if not self.inventory:
            self.log_error(fl_ctx, "got request to prepare global model but I don't have global models")
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        assert isinstance(self.persistor, ModelPersistor)
        model_learnable = self.persistor.get(model_name, fl_ctx)
        dxo = model_learnable_to_dxo(model_learnable)
        model = dxo.to_shareable()
        self._set_prepared_model(ModelType.GLOBAL, model_name, model)
        return make_reply(ReturnCode.OK)

    def _prepare_local_model(self, model_name, fl_ctx: FLContext, abort_signal: Signal):
        if not self.submit_model_executor:
            self.log_error(fl_ctx, "got request to prepare local model but I don't have local models")
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        task_data = Shareable()
        task_data.set_header(AppConstants.SUBMIT_MODEL_NAME, model_name)
        task_data.set_header(FLContextKey.TASK_NAME, self.submit_model_task_name)

        try:
            start = time.time()
            self.log_info(fl_ctx, f"invoking {type(self.submit_model_executor)} to do {self.submit_model_task_name}")
            result = self.submit_model_executor.execute(
                task_name=self.submit_model_task_name, shareable=task_data, fl_ctx=fl_ctx, abort_signal=abort_signal
            )
            time_used = time.time() - start
            self.log_info(
                fl_ctx,
                f"{type(self.submit_model_executor)} finished {self.submit_model_task_name} in {time_used} seconds",
            )

        except Exception as ex:
            self.log_exception(
                fl_ctx, f"failed to get local model from submit_model_executor: {secure_format_exception(ex)}"
            )
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        assert isinstance(result, Shareable)
        rc = result.get_return_code(ReturnCode.OK)
        if rc != ReturnCode.OK:
            self.log_error(fl_ctx, f"failed to get local model from submit_model_executor: {rc}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        self.log_info(fl_ctx, f"got local model from {type(self.submit_model_executor)}")
        self._set_prepared_model(ModelType.LOCAL, model_name, result)
        return make_reply(ReturnCode.OK)

    def _prepare_model(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        model_type = shareable.get(Constant.MODEL_TYPE)
        model_name = shareable.get(Constant.MODEL_NAME)
        self.update_status(action=f"eval:prep_model:{model_type}:{model_name}")
        self._clear_prepared_models()
        if model_type == ModelType.GLOBAL:
            return self._prepare_global_model(model_name, fl_ctx)
        else:
            return self._prepare_local_model(model_name, fl_ctx, abort_signal)

    def _process_get_model_request(self, request: Shareable, fl_ctx: FLContext) -> Shareable:
        peer_ctx = fl_ctx.get_peer_context()
        assert isinstance(peer_ctx, FLContext)
        client_name = peer_ctx.get_identity_name()
        model_type = request.get(Constant.MODEL_TYPE)
        model_name = request.get(Constant.MODEL_NAME)

        self.log_info(fl_ctx, f"got request for {model_type} {model_name} from client {client_name}")
        model = self._get_prepared_model(model_type, model_name)
        if not model:
            self.log_error(
                fl_ctx, f"got request for {model_type} {model_name} from client {client_name} but I don't have it"
            )
            return make_reply(ReturnCode.BAD_REQUEST_DATA)
        else:
            self.log_info(fl_ctx, f"sent requested {model_type} {model_name} to client {client_name}")
            return make_copy(model)

    def do_learn_task(self, name: str, task_data: Shareable, fl_ctx: FLContext, abort_signal: Signal):
        pass
