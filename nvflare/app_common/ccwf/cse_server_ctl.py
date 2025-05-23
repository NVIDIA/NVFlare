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
import time

from nvflare.apis.controller_spec import ClientTask, Task
from nvflare.apis.dxo import from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.workspace import Workspace
from nvflare.app_common.app_constant import AppConstants, ModelName
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.ccwf.common import Constant, ModelType, make_task_name
from nvflare.app_common.ccwf.eval_gen import parallel_eval_generator
from nvflare.app_common.ccwf.server_ctl import ServerSideController
from nvflare.app_common.ccwf.val_result_manager import EvalResultManager
from nvflare.fuel.utils.validation_utils import (
    DefaultValuePolicy,
    check_non_negative_int,
    check_positive_number,
    check_str,
    validate_candidate,
    validate_candidates,
)


class _TaskPropKey:
    MODEL_NAME = "model_name"
    MODEL_TYPE = "model_type"
    MODEL_READY = "model_ready"


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
        max_parallel_actions=1,
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
            result_clients=None,
            result_clients_policy=DefaultValuePolicy.EMPTY,
            progress_timeout=progress_timeout,
            private_p2p=private_p2p,
        )

        check_str("eval_result_dir", eval_result_dir)
        check_positive_number("eval_task_timeout", eval_task_timeout)
        check_non_negative_int("max_parallel_actions", max_parallel_actions)

        if not global_model_client:
            global_model_client = ""
        self.global_model_client = global_model_client
        self.prep_model_task_name = make_task_name(task_name_prefix, Constant.BASENAME_PREP_MODEL)
        self.eval_task_name = make_task_name(task_name_prefix, Constant.BASENAME_EVAL)
        self.eval_task_timeout = eval_task_timeout
        self.max_parallel_actions = max_parallel_actions
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

    def _ask_to_eval(self, evals: list, model_type: str, model_name: str, abort_signal: Signal, fl_ctx: FLContext):
        self.current_round += 1
        self.log_info(fl_ctx, f"R{self.current_round}: {evals} to evaluate {model_type} model '{model_name}'")

        # Create validation task and broadcast to all participating clients.
        tasks = []
        for evaluator, evaluatee in evals:
            task_data = Shareable()
            task_data[AppConstants.CURRENT_ROUND] = self.current_round
            task_data[Constant.MODEL_OWNER] = evaluatee  # client that holds the model
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
                targets=[evaluator],
                min_responses=1,
                wait_time_after_min_received=0,
            )
            tasks.append(task)

        # wait until all tasks are done
        while self.get_num_standing_tasks() > 0:
            if abort_signal.triggered:
                # cancel all tasks
                for t in tasks:
                    self.cancel_task(t, fl_ctx=fl_ctx)

                self.log_info(fl_ctx, f"abort signal received - cancelled {len(tasks)} pending tasks")
                return
            time.sleep(0.5)

    def _evaluate_global_models(self, abort_signal: Signal, fl_ctx: FLContext):
        if not self.eval_global:
            return

        if len(self.global_names) == 0:
            self.log_warning(fl_ctx, "no global models to evaluate!")
            return

        for model_name, owner in self.global_names.items():
            self._evaluate_one_global_model(model_name, owner, abort_signal, fl_ctx)

    def _ask_to_prepare_model(self, model_type, model_name, owners, abort_signal: Signal, fl_ctx: FLContext) -> bool:
        task_data = Shareable(
            {
                Constant.MODEL_NAME: model_name,
                Constant.MODEL_TYPE: model_type,
            }
        )

        task = Task(
            name=self.prep_model_task_name,
            data=task_data,
            result_received_cb=self._process_prep_model_result,
            timeout=self.eval_task_timeout,
        )
        task.set_prop(_TaskPropKey.MODEL_NAME, model_name)
        task.set_prop(_TaskPropKey.MODEL_TYPE, model_type)

        model_ready = {k: False for k in owners}
        task.set_prop(_TaskPropKey.MODEL_READY, model_ready)

        self.log_info(fl_ctx, f"asking {owners} to prepare model: {model_type=} {model_name=}")
        self.broadcast_and_wait(
            task=task,
            fl_ctx=fl_ctx,
            targets=owners,
            min_responses=len(owners),
            wait_time_after_min_received=0,
            abort_signal=abort_signal,
        )

        # check whether models are ready on all sites
        for client_name, ready in model_ready.items():
            if not ready:
                self.log_error(fl_ctx, f"client {client_name} failed to prepare model: {model_type=} {model_name=}")
                return False
        self.log_info(fl_ctx, f"All of {owners} successfully prepared model: {model_type=} {model_name=}")
        return True

    def _evaluate_one_global_model(self, model_name, model_owner, abort_signal: Signal, fl_ctx: FLContext):
        # ask model owners to prepare for eval
        model_ready = self._ask_to_prepare_model(ModelType.GLOBAL, model_name, [model_owner], abort_signal, fl_ctx)
        if not model_ready:
            self.log_error(fl_ctx, f"skipped global model evaluation because {model_owner} failed to prep")
            return

        self._do_eval_actions(self.evaluators, [model_owner], ModelType.GLOBAL, model_name, abort_signal, fl_ctx)

    def _do_eval_actions(self, evaluators, evaluatees, model_type, model_name, abort_signal: Signal, fl_ctx: FLContext):
        self.log_info(fl_ctx, f"Start to evaluate {model_type} {model_name}: {evaluators} => {evaluatees}")
        for evals in parallel_eval_generator(evaluators, evaluatees, self.max_parallel_actions):
            self._ask_to_eval(evals, model_type, model_name, abort_signal, fl_ctx)
            if abort_signal.triggered:
                self.log_info(fl_ctx, f"Abort evaluating {model_type} {model_name} - signal received")
                return
        self.log_info(fl_ctx, f"Finished evaluating {model_type} {model_name}: {evaluators} => {evaluatees}")

    def _evaluate_local_models(self, abort_signal: Signal, fl_ctx: FLContext):
        train_clients = fl_ctx.get_prop(Constant.PROP_KEY_TRAIN_CLIENTS)
        evaluatees = []
        for c in self.evaluatees:
            if train_clients and c not in train_clients:
                # this client does not have local models
                self.log_info(fl_ctx, f"ignore client {c} since it does not have local models")
            else:
                evaluatees.append(c)

        if not evaluatees:
            self.log_info(fl_ctx, "skipped local evaluation because no client has local models")
            return

        # ask model owners to prepare for eval
        model_name = ModelName.BEST_MODEL

        model_ready = self._ask_to_prepare_model(ModelType.LOCAL, model_name, evaluatees, abort_signal, fl_ctx)
        if not model_ready:
            self.log_error(fl_ctx, "skipped local model evaluation because some clients failed to prep")
            return

        self._do_eval_actions(self.evaluators, evaluatees, ModelType.LOCAL, model_name, abort_signal, fl_ctx)

    def sub_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        if not self.global_names and not self.evaluatees:
            self.system_panic("there are neither global models nor local models to evaluate!", fl_ctx)
            return

        # ask everyone to evaluate global model
        self._evaluate_global_models(abort_signal, fl_ctx)

        # ask everyone to eval everyone else's local model
        self._evaluate_local_models(abort_signal, fl_ctx)

    def is_sub_flow_done(self, fl_ctx: FLContext) -> bool:
        return self.get_num_standing_tasks() == 0

    def _process_eval_result(self, client_task: ClientTask, fl_ctx: FLContext):
        # Find name of the client sending this
        result = client_task.result
        client_name = client_task.client.name
        self._accept_eval_result(client_name=client_name, result=result, fl_ctx=fl_ctx)

    def _process_prep_model_result(self, client_task: ClientTask, fl_ctx: FLContext):
        task = client_task.task
        result = client_task.result
        assert isinstance(result, Shareable)
        rc = result.get_return_code()

        model_ready = task.get_prop(_TaskPropKey.MODEL_READY)
        assert isinstance(model_ready, dict)
        model_type = task.get_prop(_TaskPropKey.MODEL_TYPE)
        model_name = task.get_prop(_TaskPropKey.MODEL_NAME)

        client_name = client_task.client.name
        if rc == ReturnCode.OK:
            model_ready[client_name] = True
            self.log_info(fl_ctx, f"client {client_name} successfully prepared {model_type=} {model_name=}")
        else:
            self.log_error(fl_ctx, f"client {client_name} failed to prepare {model_type=} {model_name=}: {rc=}")

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
