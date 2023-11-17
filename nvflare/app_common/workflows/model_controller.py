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

from typing import Any, Union, Optional, List

from nvflare.apis.fl_component import FLComponentHelper
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.apis.client import Client
from nvflare.apis.controller_spec import OperatorMethod, TaskOperatorKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Task
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.security.logging import secure_format_exception
from .scatter_and_gather import ScatterAndGather
from nvflare.app_common.abstract.model import ModelLearnableKey


class ModelController(ScatterAndGather, FLComponentHelper):

    #def __init__(self, **kwargs
    #):
        #self.model = None

        #ScatterAndGather.__init__(self, **kwargs)
        #ModelLearner.__init__(self)

    def start_controller(self, fl_ctx: FLContext) -> None:
        super().start_controller(fl_ctx)
        self.engine = fl_ctx.get_engine()

        if not self._global_weights.is_empty():
            self.model = FLModel(
                params_type=ParamsType.FULL,
                params=self._global_weights[ModelLearnableKey.WEIGHTS],
                meta=self._global_weights[ModelLearnableKey.META]
            )
        else:
            self.model = FLModel(
                params_type=ParamsType.FULL,
                params={}
            )

        self.fl_ctx = fl_ctx
        self.initialize()

    def sample_clients(self, min_clients):
        clients = self.engine.get_clients()
        # TODO: sample clients

        return clients

    def send_model_and_wait(self, targets: Union[List[Client], List[str], None] = None, data: FLModel = None) -> List:
        # Create train_task
        data_shareable: Shareable = FLModelUtils.to_shareable(data)
        data_shareable.set_header(AppConstants.CURRENT_ROUND, self._current_round)
        data_shareable.set_header(AppConstants.NUM_ROUNDS, self._num_rounds)
        data_shareable.add_cookie(AppConstants.CONTRIBUTION_ROUND, self._current_round)

        operator = {
            TaskOperatorKey.OP_ID: self.train_task_name,
            TaskOperatorKey.METHOD: OperatorMethod.BROADCAST,
            TaskOperatorKey.TIMEOUT: self._train_timeout,
            TaskOperatorKey.AGGREGATOR: self.aggregator_id,
        }

        train_task = Task(
            name=self.train_task_name,
            data=data_shareable,
            operator=operator,
            props={},
            timeout=self._train_timeout,
            before_task_sent_cb=self._prepare_train_task_data,
            result_received_cb=self._process_train_result,
        )

        self.info(f"Sending train task to {[client.name for client in targets]}")
        self.broadcast_and_wait(
            task=train_task,
            targets=targets,
            min_responses=self._min_clients,
            wait_time_after_min_received=self._wait_time_after_min_received,
            fl_ctx=self.fl_ctx,
            abort_signal=self.abort_signal,
        )

        return None  # TODO: return results

    def run(self):
        raise NotImplementedError

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        self.fl_ctx = fl_ctx
        self.abort_signal = abort_signal
        try:
            self.log_info(fl_ctx, "Beginning model controller run.")
            self._phase = AppConstants.PHASE_TRAIN

            self.run()
        except Exception as e:
            error_msg = f"Exception in model controller run: {secure_format_exception(e)}"
            self.log_exception(fl_ctx, error_msg)
            self.system_panic(error_msg, fl_ctx)

    def aggregate(self, results: list=None):
        # TODO: write FLModel aggregator
        self.info("Start aggregation.")
        self.fire_event(AppEventType.BEFORE_AGGREGATION, self.fl_ctx)
        aggr_result = self.aggregator.aggregate(self.fl_ctx)
        self.fl_ctx.set_prop(AppConstants.AGGREGATION_RESULT, aggr_result, private=True, sticky=False)
        self.fire_event(AppEventType.AFTER_AGGREGATION, self.fl_ctx)
        self.info("End aggregation.")

        return aggr_result

    def update_model(self, aggr_result):
        self.fire_event(AppEventType.BEFORE_SHAREABLE_TO_LEARNABLE, self.fl_ctx)
        self._global_weights = self.shareable_gen.shareable_to_learnable(aggr_result, self.fl_ctx)

        self.model = FLModel(
            params_type=ParamsType.FULL,
            params=self._global_weights[ModelLearnableKey.WEIGHTS],
            meta=self._global_weights[ModelLearnableKey.META]
        )

        self.fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, self._global_weights, private=True, sticky=True)
        self.fl_ctx.sync_sticky()
        self.fire_event(AppEventType.AFTER_SHAREABLE_TO_LEARNABLE, self.fl_ctx)

    def save_model(self):
        if self.persistor:
            if (
                    self._persist_every_n_rounds != 0
                    and (self._current_round + 1) % self._persist_every_n_rounds == 0
            ) or self._current_round == self._start_round + self._num_rounds - 1:
                self.info("Start persist model on server.")
                self.fire_event(AppEventType.BEFORE_LEARNABLE_PERSIST, self.fl_ctx)
                self.persistor.save(self._global_weights, self.fl_ctx)
                self.fire_event(AppEventType.AFTER_LEARNABLE_PERSIST, self.fl_ctx)
                self.info("End persist model on server.")

    def stop_controller(self, fl_ctx: FLContext):
        super().stop_controller(fl_ctx)
        self.fl_ctx = fl_ctx
        self.finalize()
