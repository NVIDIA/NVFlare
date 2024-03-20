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

from nvflare.apis.controller_spec import ClientTask, Task
from nvflare.apis.dxo import from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.workspace import Workspace
from nvflare.app_common.app_constant import AppConstants, ModelName
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.ccwf.common import Constant, ModelType, make_task_name
from nvflare.app_common.ccwf.server_ctl import ServerSideController
from nvflare.app_common.ccwf.val_result_manager import EvalResultManager
from nvflare.fuel.utils.validation_utils import (
    DefaultValuePolicy,
    check_positive_number,
    check_str,
    validate_candidate,
    validate_candidates,
)


class CrossSiteEvalServerController(ServerSideController):
    def __init__(
        self,
        task_name_prefix=Constant.TN_PREFIX_CROSS_SITE_EVAL,
        start_task_timeout=Constant.START_TASK_TIMEOUT,
        configure_task_timeout=Constant.CONFIG_TASK_TIMEOUT,
        eval_task_timeout=30,
        task_check_period: float = Constant.TASK_CHECK_INTERVAL,
        job_status_check_interval: float = Constant.JOB_STATUS_CHECK_INTERVAL,
        progress_timeout: float = Constant.WORKFLOW_PROGRESS_TIMEOUT,
        private_p2p: bool = True,
        participating_clients=None,
        evaluators=None,
        evaluatees=None,
        global_model_client=None,
        max_status_report_interval: float = Constant.PER_CLIENT_STATUS_REPORT_TIMEOUT,
        eval_result_dir=AppConstants.CROSS_VAL_DIR,
    ):
        if not evaluatees:
            evaluatees = []

        if not evaluators:
            evaluators = []

        super().__init__(
            num_rounds=1,
            task_name_prefix=task_name_prefix,
            start_task_timeout=start_task_timeout,
            configure_task_timeout=configure_task_timeout,
            task_check_period=task_check_period,
            job_status_check_interval=job_status_check_interval,
            participating_clients=participating_clients,
            starting_client="",
            starting_client_policy=DefaultValuePolicy.EMPTY,
            max_status_report_interval=max_status_report_interval,
            result_clients="",
            result_clients_policy=DefaultValuePolicy.EMPTY,
            progress_timeout=progress_timeout,
            private_p2p=private_p2p,
        )

        check_str("eval_result_dir", eval_result_dir)
        check_positive_number("eval_task_timeout", eval_task_timeout)

        if not global_model_client:
            global_model_client = ""
        self.global_model_client = global_model_client
        self.eval_task_name = make_task_name(task_name_prefix, Constant.BASENAME_EVAL)
        self.eval_task_timeout = eval_task_timeout
        self.eval_local = False
        self.eval_global = False
        self.evaluators = evaluators
        self.evaluatees = evaluatees
        self.eval_result_dir = eval_result_dir
        self.global_names = {}
        self.eval_manager = None
        self.current_round = 0

    def start_controller(self, fl_ctx: FLContext):
        super().start_controller(fl_ctx)

        self.evaluators = validate_candidates(
            var_name="evaluators",
            candidates=self.evaluators,
            base=self.participating_clients,
            default_policy=DefaultValuePolicy.ALL,
            allow_none=False,
        )

        self.evaluatees = validate_candidates(
            var_name="evaluatees",
            candidates=self.evaluatees,
            base=self.participating_clients,
            default_policy=DefaultValuePolicy.ALL,
            allow_none=True,
        )

        self.global_model_client = validate_candidate(
            var_name="global_model_client",
            candidate=self.global_model_client,
            base=self.participating_clients,
            default_policy=DefaultValuePolicy.ANY,
            allow_none=True,
        )

        if self.global_model_client:
            self.eval_global = True

        if self.evaluatees:
            self.eval_local = True

        if not self.eval_global and not self.eval_local:
            raise RuntimeError("nothing to evaluate: you must set evaluatees and/or eval_global")

        workspace: Workspace = self._engine.get_workspace()
        run_dir = workspace.get_run_dir(fl_ctx.get_job_id())
        cross_val_path = os.path.join(run_dir, self.eval_result_dir)
        cross_val_results_dir = os.path.join(cross_val_path, AppConstants.CROSS_VAL_RESULTS_DIR_NAME)
        self.eval_manager = EvalResultManager(cross_val_results_dir)

    def prepare_config(self):
        return {
            Constant.EVAL_LOCAL: self.eval_local,
            Constant.EVAL_GLOBAL: self.eval_global,
            Constant.EVALUATORS: self.evaluators,
            Constant.EVALUATEES: self.evaluatees,
            Constant.GLOBAL_CLIENT: self.global_model_client,
        }

    def process_config_reply(self, client_name: str, reply: Shareable, fl_ctx: FLContext) -> bool:
        global_names = reply.get(Constant.GLOBAL_NAMES)
        if global_names:
            for m in global_names:
                if m not in self.global_names:
                    self.global_names[m] = client_name
                    self.log_info(fl_ctx, f"got global model name {m} from {client_name}")
        return True

    def _ask_to_evaluate(
        self, current_round: int, model_name: str, model_type: str, model_owner: str, fl_ctx: FLContext
    ):
        self.log_info(
            fl_ctx,
            f"R{current_round}: asking {self.evaluators} to evaluate {model_type} model '{model_name}' "
            f"on client '{model_owner}'",
        )

        # Create validation task and broadcast to all participating clients.
        task_data = Shareable()
        task_data[AppConstants.CURRENT_ROUND] = current_round
        task_data[Constant.MODEL_OWNER] = model_owner  # client that holds the model
        task_data[Constant.MODEL_NAME] = model_name
        task_data[Constant.MODEL_TYPE] = model_type

        task = Task(
            name=self.eval_task_name,
            data=task_data,
            result_received_cb=self._process_eval_result,
            timeout=self.eval_task_timeout,
        )

        self.broadcast(
            task=task,
            fl_ctx=fl_ctx,
            targets=self.evaluators,
            min_responses=len(self.evaluators),
            wait_time_after_min_received=0,
        )

    def sub_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        if not self.global_names and not self.evaluatees:
            self.system_panic("there are neither global models nor local models to evaluate!", fl_ctx)
            return

        # ask everyone to evaluate global model
        if self.eval_global:
            if len(self.global_names) == 0:
                self.log_warning(fl_ctx, "no global models to evaluate!")

        for m, owner in self.global_names.items():
            self._ask_to_evaluate(
                current_round=self.current_round,
                model_name=m,
                model_type=ModelType.GLOBAL,
                model_owner=owner,
                fl_ctx=fl_ctx,
            )
            self.current_round += 1

        # ask everyone to eval everyone else's local model
        train_clients = fl_ctx.get_prop(Constant.PROP_KEY_TRAIN_CLIENTS)
        for c in self.evaluatees:
            if train_clients and c not in train_clients:
                # this client does not have local models
                self.log_info(fl_ctx, f"ignore client {c} since it does not have local models")
                continue

            self._ask_to_evaluate(
                current_round=self.current_round,
                model_name=ModelName.BEST_MODEL,
                model_type=ModelType.LOCAL,
                model_owner=c,
                fl_ctx=fl_ctx,
            )
            self.current_round += 1

    def is_sub_flow_done(self, fl_ctx: FLContext) -> bool:
        return self.get_num_standing_tasks() == 0

    def _process_eval_result(self, client_task: ClientTask, fl_ctx: FLContext):
        # Find name of the client sending this
        result = client_task.result
        client_name = client_task.client.name
        self._accept_eval_result(client_name=client_name, result=result, fl_ctx=fl_ctx)

    def _accept_eval_result(self, client_name: str, result: Shareable, fl_ctx: FLContext):
        model_owner = result.get_header(Constant.MODEL_OWNER, "")
        model_type = result.get_header(Constant.MODEL_TYPE)
        model_name = result.get_header(Constant.MODEL_NAME)

        if model_type == ModelType.GLOBAL:
            # global model
            model_owner = "GLOBAL_" + model_name
            model_info = model_owner
        else:
            model_info = f"{model_name} of {model_owner}"

        # Fire event. This needs to be a new local context per each client
        fl_ctx.set_prop(AppConstants.MODEL_OWNER, model_owner, private=True, sticky=False)
        fl_ctx.set_prop(AppConstants.DATA_CLIENT, client_name, private=True, sticky=False)
        fl_ctx.set_prop(AppConstants.VALIDATION_RESULT, result, private=True, sticky=False)
        self.fire_event(AppEventType.VALIDATION_RESULT_RECEIVED, fl_ctx)

        rc = result.get_return_code(ReturnCode.OK)
        if rc != ReturnCode.OK:
            self.log_error(fl_ctx, f"bad evaluation result from client {client_name} on model {model_info}")
        else:
            dxo = from_shareable(result)
            location = self.eval_manager.add_result(evaluatee=model_owner, evaluator=client_name, result=dxo)
            self.log_info(fl_ctx, f"saved evaluation result from {client_name} on model {model_info} in {location}")
