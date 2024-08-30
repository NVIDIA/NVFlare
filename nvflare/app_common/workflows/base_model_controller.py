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
import random
from abc import ABC, abstractmethod
from typing import Callable, List, Union

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, OperatorMethod, Task, TaskOperatorKey
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.abstract.learnable_persistor import LearnablePersistor
from nvflare.app_common.abstract.model import ModelLearnable, ModelLearnableKey, make_model_learnable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.utils.fl_component_wrapper import FLComponentWrapper
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.fuel.utils.validation_utils import check_non_negative_int, check_positive_int, check_str
from nvflare.security.logging import secure_format_exception


class BaseModelController(Controller, FLComponentWrapper, ABC):
    def __init__(
        self,
        persistor_id: str = AppConstants.DEFAULT_PERSISTOR_ID,
        ignore_result_error: bool = False,
        allow_empty_global_weights: bool = False,
        task_check_period: float = 0.5,
    ):
        """FLModel based controller.

        Args:
            persistor_id (str, optional): ID of the persistor component. Defaults to AppConstants.DEFAULT_PERSISTOR_ID ("persistor").
            ignore_result_error (bool, optional): whether this controller can proceed if client result has errors.
                Defaults to False.
            allow_empty_global_weights (bool, optional): whether to allow empty global weights. Some pipelines can have
                empty global weights at first round, such that clients start training from scratch without any global info.
                Defaults to False.
            task_check_period (float, optional): interval for checking status of tasks. Defaults to 0.5.
        """
        super().__init__(task_check_period=task_check_period)

        # Check arguments
        check_str("persistor_id", persistor_id)
        if not isinstance(task_check_period, (int, float)):
            raise TypeError(f"task_check_period must be an int or float but got {type(task_check_period)}")
        elif task_check_period <= 0:
            raise ValueError("task_check_period must be greater than 0.")
        self._task_check_period = task_check_period
        self._persistor_id = persistor_id
        self.persistor = None

        # config data
        self._ignore_result_error = ignore_result_error
        self._allow_empty_global_weights = allow_empty_global_weights

        # model related
        self._results = []

    def start_controller(self, fl_ctx: FLContext) -> None:
        self.fl_ctx = fl_ctx
        self.info("Initializing BaseModelController workflow.")

        self.engine = self.fl_ctx.get_engine()

        if self._persistor_id:
            self.persistor = self.engine.get_component(self._persistor_id)
            if not isinstance(self.persistor, LearnablePersistor):
                self.warning(
                    f"Persistor {self._persistor_id} must be a LearnablePersistor type object, "
                    f"but got {type(self.persistor)}"
                )
                self.persistor = None

        FLComponentWrapper.initialize(self)

    def _build_shareable(self, data: FLModel = None) -> Shareable:
        data_shareable: Shareable = FLModelUtils.to_shareable(data)
        data_shareable.add_cookie(
            AppConstants.CONTRIBUTION_ROUND, data_shareable.get_header(AppConstants.CURRENT_ROUND)
        )

        return data_shareable

    def broadcast_model(
        self,
        task_name: str = AppConstants.TASK_TRAIN,
        data: FLModel = None,
        targets: Union[List[Client], List[str], None] = None,
        min_responses: int = None,
        timeout: int = 0,
        wait_time_after_min_received: int = 0,
        blocking: bool = True,
        callback: Callable[[FLModel], None] = None,
    ) -> List:
        """Send a task with data to a list of targets.

        Args:
            task_name (str, optional): name of the task. Defaults to "train".
            data (FLModel, optional): FLModel to be sent to clients. If no data is given, send empty FLModel.
            targets (List[str], optional): the list of target client names or None (all clients). Defaults to None.
            min_responses (int, optional): the minimum number of responses expected. If None, must receive responses from
              all clients that the task has been sent to. Defaults to None.
            timeout (int, optional): time to wait for clients to perform task. Defaults to 0, i.e., never time out.
            wait_time_after_min_received (int, optional): time to wait after
                minimum number of clients responses has been received. Defaults to 0.
            blocking (bool, optional): whether to block to wait for task result. Defaults to True.
            callback (Callable[[FLModel], None], optional): callback when a result is received, only called when blocking=False. Defaults to None.

        Returns:
            List[FLModel] if blocking=True else None
        """

        if not isinstance(task_name, str):
            raise TypeError("task_name must be a string but got {}".format(type(task_name)))
        if data and not isinstance(data, FLModel):
            raise TypeError("data must be a FLModel or None but got {}".format(type(data)))
        if min_responses is None:
            min_responses = 0  # this is internally used by controller's broadcast to represent all targets
        check_non_negative_int("min_responses", min_responses)
        check_non_negative_int("timeout", timeout)
        check_non_negative_int("wait_time_after_min_received", wait_time_after_min_received)
        if not blocking and not isinstance(callback, Callable):
            raise TypeError("callback must be defined if blocking is False, but got {}".format(type(callback)))

        if not data:
            self.warning("data is None. Sending empty FLModel.")
            data = FLModel(params_type=ParamsType.FULL, params={})

        task = self._prepare_task(data=data, task_name=task_name, timeout=timeout, callback=callback)

        if targets:
            targets = [client.name if isinstance(client, Client) else client for client in targets]
            self.info(f"Sending task {task_name} to {targets}")
        else:
            self.info(f"Sending task {task_name} to all clients")

        if blocking:
            self._results = []  # reset results list
            self.broadcast_and_wait(
                task=task,
                targets=targets,
                min_responses=min_responses,
                wait_time_after_min_received=wait_time_after_min_received,
                fl_ctx=self.fl_ctx,
                abort_signal=self.abort_signal,
            )

            if targets is not None:
                expected_responses = min_responses if min_responses != 0 else len(targets)
                if len(self._results) != expected_responses:
                    self.warning(
                        f"Number of results ({len(self._results)}) is different from number of expected responses ({expected_responses})."
                    )

            # de-reference the internal results before returning
            results = self._results
            self._results = []
            return results
        else:
            self.broadcast(
                task=task,
                targets=targets,
                min_responses=min_responses,
                wait_time_after_min_received=wait_time_after_min_received,
                fl_ctx=self.fl_ctx,
            )

    def _prepare_task(
        self,
        data: FLModel,
        task_name: str,
        timeout: int,
        callback: Callable,
    ):
        # Create task
        data_shareable = self._build_shareable(data)

        operator = {
            TaskOperatorKey.OP_ID: task_name,
            TaskOperatorKey.METHOD: OperatorMethod.BROADCAST,
            TaskOperatorKey.TIMEOUT: timeout,
        }

        task = Task(
            name=task_name,
            data=data_shareable,
            operator=operator,
            props={AppConstants.TASK_PROP_CALLBACK: callback, AppConstants.META_DATA: data.meta},
            timeout=timeout,
            before_task_sent_cb=self._prepare_task_data,
            result_received_cb=self._process_result,
        )

        return task

    def _prepare_task_data(self, client_task: ClientTask, fl_ctx: FLContext) -> None:
        fl_ctx.set_prop(AppConstants.TRAIN_SHAREABLE, client_task.task.data, private=True, sticky=False)
        self.event(AppEventType.BEFORE_TRAIN_TASK)

    def _process_result(self, client_task: ClientTask, fl_ctx: FLContext) -> None:
        self.fl_ctx = fl_ctx
        result = client_task.result
        client_name = client_task.client.name

        # Turn result into FLModel
        result_model = FLModelUtils.from_shareable(result)
        result_model.meta["props"] = client_task.task.props[AppConstants.META_DATA]
        result_model.meta["client_name"] = client_name

        if result_model.current_round is not None:
            self.fl_ctx.set_prop(AppConstants.CURRENT_ROUND, result_model.current_round, private=True, sticky=True)

        self.event(AppEventType.BEFORE_CONTRIBUTION_ACCEPT)
        self._accept_train_result(client_name=client_name, result=result, fl_ctx=fl_ctx)
        self.event(AppEventType.AFTER_CONTRIBUTION_ACCEPT)

        callback = client_task.task.get_prop(AppConstants.TASK_PROP_CALLBACK)
        if callback:
            try:
                callback(result_model)
            except Exception as e:
                self.error(f"Unsuccessful callback {callback} for task {client_task.task.name}: {e}")
        else:
            self._results.append(result_model)

            # Cleanup task result
            client_task.result = None

        gc.collect()

    def process_result_of_unknown_task(
        self, client: Client, task_name: str, client_task_id: str, result: Shareable, fl_ctx: FLContext
    ) -> None:
        if task_name == AppConstants.TASK_TRAIN:
            self._accept_train_result(client_name=client.name, result=result, fl_ctx=fl_ctx)
            self.info(f"Result of unknown task {task_name} sent to aggregator.")
        else:
            self.error("Ignoring result from unknown task.")

    def _accept_train_result(self, client_name: str, result: Shareable, fl_ctx: FLContext):
        self.fl_ctx = fl_ctx
        rc = result.get_return_code()

        current_round = result.get_header(AppConstants.CURRENT_ROUND, None)

        # Raise panic if bad peer context or execution exception.
        if rc and rc != ReturnCode.OK:
            if self._ignore_result_error:
                self.warning(
                    f"Ignore the train result from {client_name} at round {current_round}. Train result error code: {rc}",
                )
            else:
                self.panic(
                    f"Result from {client_name} is bad, error code: {rc}. "
                    f"{self.__class__.__name__} exiting at round {current_round}."
                )
                return

        self.fl_ctx.set_prop(AppConstants.TRAINING_RESULT, result, private=True, sticky=False)

    @abstractmethod
    def run(self):
        """Main `run` routine called by the Controller's `control_flow` to execute the workflow.

        Returns: None.

        """
        raise NotImplementedError

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        self.fl_ctx = fl_ctx
        self.abort_signal = abort_signal
        try:
            self.info("Beginning model controller run.")
            self.event(AppEventType.TRAINING_STARTED)

            self.run()
        except Exception as e:
            error_msg = f"Exception in model controller run: {secure_format_exception(e)}"
            self.exception(error_msg)
            self.panic(error_msg)

    def load_model(self):
        # initialize global model
        model = None
        if self.persistor:
            self.info("loading initial model from persistor")
            global_weights = self.persistor.load(self.fl_ctx)

            if not isinstance(global_weights, ModelLearnable):
                self.panic(
                    f"Expected global weights to be of type `ModelLearnable` but received {type(global_weights)}"
                )
                return

            if global_weights.is_empty():
                if not self._allow_empty_global_weights:
                    # if empty not allowed, further check whether it is available from fl_ctx
                    global_weights = self.fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)

            if not global_weights.is_empty():
                model = FLModel(
                    params_type=ParamsType.FULL,
                    params=global_weights[ModelLearnableKey.WEIGHTS],
                    meta=global_weights[ModelLearnableKey.META],
                )
            elif self._allow_empty_global_weights:
                model = FLModel(params_type=ParamsType.FULL, params={})
            else:
                self.panic(
                    f"Neither `persistor` {self._persistor_id} or `fl_ctx` returned a global model! If this was intended, set `self._allow_empty_global_weights` to `True`."
                )
                return
        else:
            self.info("persistor not configured, creating empty initial FLModel")
            model = FLModel(params_type=ParamsType.FULL, params={})

        # persistor uses Learnable format to save model
        ml = make_model_learnable(weights=model.params, meta_props=model.meta)
        self.fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, ml, private=True, sticky=True)
        self.event(AppEventType.INITIAL_MODEL_LOADED)

        return model

    def get_run_dir(self):
        """Get current run directory."""
        return self.engine.get_workspace().get_run_dir(self.fl_ctx.get_job_id())

    def get_app_dir(self):
        """Get current app directory."""
        return self.engine.get_workspace().get_app_dir(self.fl_ctx.get_job_id())

    def save_model(self, model):
        if self.persistor:
            self.info("Start persist model on server.")
            self.event(AppEventType.BEFORE_LEARNABLE_PERSIST)
            # persistor uses Learnable format to save model
            ml = make_model_learnable(weights=model.params, meta_props=model.meta)
            self.persistor.save(ml, self.fl_ctx)
            self.event(AppEventType.AFTER_LEARNABLE_PERSIST)
            self.info("End persist model on server.")
        else:
            self.error("persistor not configured, model will not be saved")

    def sample_clients(self, num_clients: int = None) -> List[str]:
        clients = [client.name for client in self.engine.get_clients()]

        if num_clients:
            check_positive_int("num_clients", num_clients)
            if num_clients < len(clients):
                random.shuffle(clients)
                clients = clients[0:num_clients]
                self.info(
                    f"num_clients ({num_clients}) is less than the number of available clients. Returning a random subset of ({num_clients}) clients."
                )
            elif num_clients > len(clients):
                self.error(
                    f"num_clients ({num_clients}) is greater than the number of available clients. Returning all ({len(clients)}) available clients."
                )

        self.info(f"Sampled clients: {clients}")

        return clients

    def get_component(self, component_id: str):
        return self.engine.get_component(component_id)

    def build_component(self, config_dict: dict):
        return self.engine.build_component(config_dict)

    def stop_controller(self, fl_ctx: FLContext):
        self.fl_ctx = fl_ctx
        self.finalize()
