# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from monai.fl.client import ClientAlgo
from monai.fl.utils.constants import ExtraItems, FlStatistics, ModelType, WeightType
from monai.fl.utils.exchange_object import ExchangeObject

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants, ModelName, ValidateType


def exchangeobj_from_shareable(shareable: Shareable):
    dxo = from_shareable(shareable)
    eo = ExchangeObject(weights=dxo.data)
    return eo


def weights_to_numpy(exchange_object: ExchangeObject):
    if not exchange_object.is_valid_weights():
        raise ValueError(f"global_model ExchangeObject is not valid: {exchange_object}")

    weights = exchange_object.weights
    for name in weights:
        weights[name] = weights[name].detach().cpu().numpy()
    exchange_object.weights = weights
    return exchange_object


class ClientAlgoExecutor(Executor):
    def __init__(
        self,
        client_algo_id,
        stats_sender_id=None,
        train_task=AppConstants.TASK_TRAIN,
        submit_model_task=AppConstants.TASK_SUBMIT_MODEL,
        validate_task=AppConstants.TASK_VALIDATION,
        key_metric: str = "accuracy",
    ):
        """Key component to run client_algo on clients.

        Args:
            client_algo_id (str): id pointing to the client_algo object
            stats_sender_id (str, optional): id pointing to the LogWriter object
            train_task (str, optional): label to dispatch train task. Defaults to AppConstants.TASK_TRAIN.
            submit_model_task (str, optional): label to dispatch submit model task. Defaults to AppConstants.TASK_SUBMIT_MODEL.
            validate_task (str, optional): label to dispatch validation task. Defaults to AppConstants.TASK_VALIDATION.
        """
        super().__init__()
        self.client_algo_id = client_algo_id
        self.stats_sender_id = stats_sender_id
        self.client_algo = None
        self.train_task = train_task
        self.submit_model_task = submit_model_task
        self.validate_task = validate_task
        self.client_id = None
        self.key_metric = key_metric

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.initialize(fl_ctx)
        elif event_type == EventType.ABORT_TASK:
            try:
                self.log_info(fl_ctx, "Aborting ClientAlgo execution...")
                if self.client_algo:
                    self.client_algo.abort(fl_ctx)
            except Exception as e:
                self.log_exception(fl_ctx, f"client_algo abort exception: {e}")
        elif event_type == EventType.END_RUN:
            self.finalize(fl_ctx)
        elif event_type == EventType.SWAP_OUT:  # only used during simulation
            self.finalize(fl_ctx)

    def initialize(self, fl_ctx: FLContext):
        try:
            self.client_id = fl_ctx.get_identity_name()
            engine = fl_ctx.get_engine()
            stats_sender = engine.get_component(self.stats_sender_id) if self.stats_sender_id else None
            self.client_algo = engine.get_component(self.client_algo_id)
            if not isinstance(self.client_algo, ClientAlgo):
                raise TypeError(f"client_algo must be client_algo type. Got: {type(self.client_algo)}")
            self.client_algo.initialize(
                extra={
                    ExtraItems.CLIENT_NAME: fl_ctx.get_identity_name(),
                    ExtraItems.APP_ROOT: fl_ctx.get_prop(FLContextKey.APP_ROOT),
                    ExtraItems.STATS_SENDER: stats_sender,
                    ExtraItems.LOGGING_FILE: False,
                }
            )
        except Exception as e:
            self.log_exception(fl_ctx, f"client_algo initialize exception: {e}")

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self.log_info(fl_ctx, f"Client trainer got task: {task_name}")

        try:
            if task_name == self.train_task:
                return self.train(shareable, fl_ctx, abort_signal)
            elif task_name == self.submit_model_task:
                return self.submit_model(shareable, fl_ctx)
            elif task_name == self.validate_task:
                return self.validate(shareable, fl_ctx, abort_signal)
            else:
                self.log_error(fl_ctx, f"Could not handle task: {task_name}")
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except Exception as e:
            # Task execution error, return EXECUTION_EXCEPTION Shareable
            self.log_exception(fl_ctx, f"client_algo execute exception: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self.log_debug(fl_ctx, f"train abort signal: {abort_signal.triggered}")

        shareable.set_header(AppConstants.VALIDATE_TYPE, ValidateType.BEFORE_TRAIN_VALIDATE)

        test_report = self.client_algo.evaluate(exchangeobj_from_shareable(shareable))
        test_key_metric = test_report.metrics.get(self.key_metric)
        self.log_info(
            fl_ctx, f"{self.client_id} reported key metric {self.key_metric}: {test_key_metric}"
        )  # only return key metric here
        validate_result = DXO(
            data_kind=DataKind.METRICS, data={MetaKey.INITIAL_METRICS: test_key_metric}
        ).to_shareable()

        self.client_algo.train(exchangeobj_from_shareable(shareable), extra={ExtraItems.ABORT: abort_signal})
        local_weights_eo = self.client_algo.get_weights()

        # check returned weights dict
        if local_weights_eo.weights is None:
            self.log_error(fl_ctx, "Returned exchange object doesn't contain weights.")
            return make_reply(ReturnCode.ERROR)

        # convert MONAI's WeightType to NVFlare's DataKind
        if local_weights_eo.weight_type == WeightType.WEIGHTS:
            data_kind = DataKind.WEIGHTS
        elif local_weights_eo.weight_type == WeightType.WEIGHT_DIFF:
            data_kind = DataKind.WEIGHT_DIFF
        else:
            self.log_error(
                fl_ctx,
                f"Returned `WeightType` not supported. Expected {WeightType.WEIGHTS} or {WeightType.WEIGHT_DIFF},"
                f" but got {local_weights_eo.get_weight_type()}",
            )
            return make_reply(ReturnCode.ERROR)

        # get the number of executed steps
        statistics = local_weights_eo.statistics
        executed_steps = statistics.get(FlStatistics.NUM_EXECUTED_ITERATIONS)
        if executed_steps:
            meta = {MetaKey.NUM_STEPS_CURRENT_ROUND: executed_steps}
        else:
            meta = None

        # Get returned weights
        local_weights_eo = weights_to_numpy(local_weights_eo)
        train_result = DXO(data_kind=data_kind, data=local_weights_eo.weights, meta=meta).to_shareable()
        # Note, optionally could also support returned optimizer state

        # if the client_algo returned the valid BEFORE_TRAIN_VALIDATE result, set the INITIAL_METRICS in
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
        # select MONAI's ModelType based on NVFlare's model_name
        if model_name == ModelName.BEST_MODEL:
            model_type = ModelType.BEST_MODEL
        elif model_name == ModelName.FINAL_MODEL:
            model_type = ModelType.FINAL_MODEL
        else:
            self.log_error(
                fl_ctx,
                f"Requested `ModelName` not supported. Expected {ModelName.BEST_MODEL} or {ModelName.FINAL_MODEL},"
                f" but got {model_name}",
            )
            return make_reply(ReturnCode.ERROR)
        local_weights_eo = self.client_algo.get_weights(extra={ExtraItems.MODEL_TYPE: model_type})
        if local_weights_eo.weights is not None:
            local_weights_eo = weights_to_numpy(local_weights_eo)
            return DXO(data_kind=DataKind.WEIGHTS, data=local_weights_eo.weights).to_shareable()
        else:
            return make_reply(ReturnCode.EMPTY_RESULT)

    def validate(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self.log_debug(fl_ctx, f"validate abort_signal {abort_signal.triggered}")

        shareable.set_header(AppConstants.VALIDATE_TYPE, ValidateType.MODEL_VALIDATE)
        test_report = self.client_algo.evaluate(exchangeobj_from_shareable(shareable))
        if test_report.metrics is not None:
            return DXO(data_kind=DataKind.METRICS, data=test_report.metrics).to_shareable()
        else:
            return make_reply(ReturnCode.EMPTY_RESULT)

    def finalize(self, fl_ctx: FLContext):
        try:
            if self.client_algo:
                self.client_algo.finalize(fl_ctx)
        except Exception as e:
            self.log_exception(fl_ctx, f"client_algo finalize exception: {e}")
