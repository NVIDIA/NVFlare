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

import copy
import os
import random
import time

import numpy as np

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model import ModelLearnable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.np.constants import NPConstants
from nvflare.app_common.utils.file_utils import resolve_path_under_root
from nvflare.fuel.utils.deprecated import warn_deprecated
from nvflare.security.logging import secure_format_exception

_NP_TRAINER_DEPRECATION_MSG = (
    "NPTrainer is deprecated but remains supported for backward compatibility. "
    "Use the Recipe API with the Client API for new projects."
)


class NPTrainer(Executor):
    def __init__(
        self,
        delta=1,
        sleep_time=0,
        train_task_name=AppConstants.TASK_TRAIN,
        submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL,
        validate_model_task_name=AppConstants.TASK_VALIDATION,
        model_name="best_numpy.npy",
        model_dir="model",
    ):
        # Init functions of components should be very minimal. Init
        # is called when json is read. A big init will cause json loading to halt
        # for long time.
        warn_deprecated(_NP_TRAINER_DEPRECATION_MSG, stacklevel=3)
        super().__init__()

        if not (isinstance(delta, float) or isinstance(delta, int)):
            raise TypeError("delta must be an instance of float or int.")

        self._delta = delta
        self._model_name = model_name
        self._model_dir = model_dir
        self._sleep_time = sleep_time
        self._train_task_name = train_task_name
        self._submit_model_task_name = submit_model_task_name
        self._validate_model_task_name = validate_model_task_name

    @staticmethod
    def _model_summary(np_data):
        weights = np_data.get(NPConstants.NUMPY_KEY) if isinstance(np_data, dict) else None
        if weights is None:
            return f"{NPConstants.NUMPY_KEY}=missing"
        return f"shape={getattr(weights, 'shape', None)}, dtype={getattr(weights, 'dtype', None)}"

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        # if event_type == EventType.START_RUN:
        #     Create all major components here. This is a simple app that doesn't need any components.
        # elif event_type == EventType.END_RUN:
        #     # Clean up resources (closing files, joining threads, removing dirs etc.)
        pass

    def _train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal):
        # First we extract DXO from the shareable.
        try:
            incoming_dxo = from_shareable(shareable)
        except Exception as e:
            self.system_panic(
                f"Unable to convert shareable to model definition. Exception {secure_format_exception(e)}", fl_ctx
            )
            return make_reply(ReturnCode.BAD_TASK_DATA)

        # Information about workflow is retrieved from the shareable header.
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND, None)
        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS, None)

        # Ensure that data is of type weights. Extract model data.
        if incoming_dxo.data_kind != DataKind.WEIGHTS:
            self.system_panic("Model DXO should be of kind DataKind.WEIGHTS.", fl_ctx)
            return make_reply(ReturnCode.BAD_TASK_DATA)
        np_data = copy.deepcopy(incoming_dxo.data)

        # Display properties.
        self.log_info(fl_ctx, f"Incoming data kind: {incoming_dxo.data_kind}")
        self.log_info(
            fl_ctx, f"Model received for round {current_round}/{total_rounds}: {self._model_summary(np_data)}"
        )
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")

        # Check abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # Doing some dummy training.
        if np_data:
            if NPConstants.NUMPY_KEY in np_data:
                np_data[NPConstants.NUMPY_KEY] += self._delta
            else:
                self.log_error(fl_ctx, "numpy_key not found in model.")
                return make_reply(ReturnCode.BAD_TASK_DATA)
        else:
            self.log_error(fl_ctx, "No model weights found in shareable.")
            return make_reply(ReturnCode.BAD_TASK_DATA)

        # We check abort_signal regularly to make sure
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # Save local numpy model
        try:
            self._save_local_model(fl_ctx, np_data)
        except Exception as e:
            self.log_error(fl_ctx, f"Exception in saving local model: {secure_format_exception(e)}.")

        self.log_info(
            fl_ctx,
            f"Completed mock training for round {current_round}/{total_rounds}: "
            f"added delta={self._delta} to {NPConstants.NUMPY_KEY}; {self._model_summary(np_data)}",
        )

        # Checking abort signal again.
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # Prepare a DXO for our updated model. Create shareable and return
        fake_metric = random.uniform(0.1, 1.0)
        d = self._delta
        outgoing_np = {NPConstants.NUMPY_KEY: np.array([[d, d, d], [d, d, d], [d, d, d]], dtype=np.float32)}
        outgoing_dxo = DXO(
            data_kind=DataKind.WEIGHT_DIFF,
            data=outgoing_np,
            meta={
                MetaKey.NUM_STEPS_CURRENT_ROUND: 1,
                MetaKey.INITIAL_METRICS: fake_metric,
            },
        )

        # artificial delay
        # if fl_ctx.get_identity_name() == "blue":
        #     time.sleep(3.0)
        # time.sleep(random.uniform(1.0, 5.0))

        return outgoing_dxo.to_shareable()

    def _submit_model(self, fl_ctx: FLContext, abort_signal: Signal):
        # Retrieve the local model saved during training.
        np_data = None
        try:
            np_data = self._load_local_model(fl_ctx)
        except Exception as e:
            self.log_error(fl_ctx, f"Unable to load model: {secure_format_exception(e)}")

        # Checking abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # Create DXO and shareable from model data.
        model_shareable = Shareable()
        if np_data:
            outgoing_dxo = DXO(data_kind=DataKind.WEIGHTS, data=np_data)
            model_shareable = outgoing_dxo.to_shareable()
        else:
            # Set return code.
            self.log_error(fl_ctx, "local model not found.")
            model_shareable.set_return_code(ReturnCode.EXECUTION_RESULT_ERROR)

        return model_shareable

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        # Any long tasks should check abort_signal regularly. Otherwise, abort client
        # will not work.
        count, interval = 0, 0.5
        while count < self._sleep_time:
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            time.sleep(interval)
            count += interval

        self.log_info(fl_ctx, f"Task name: {task_name}")
        try:
            if task_name == self._train_task_name:
                return self._train(shareable=shareable, fl_ctx=fl_ctx, abort_signal=abort_signal)
            elif task_name == self._submit_model_task_name:
                return self._submit_model(fl_ctx=fl_ctx, abort_signal=abort_signal)
            elif task_name == self._validate_model_task_name:
                return self._validate_model(shareable, fl_ctx, abort_signal)
            else:
                # If unknown task name, set RC accordingly.
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except Exception as e:
            self.log_exception(fl_ctx, f"Exception in NPTrainer execute: {secure_format_exception(e)}.")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def _validate_model(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        dxo = from_shareable(shareable)
        self.log_info(fl_ctx, f"Validating model: data_kind={dxo.data_kind}, {self._model_summary(dxo.data)}")
        fake_metric = random.uniform(0.1, 1.0)
        val_results = {"val_accuracy": fake_metric}
        metric_dxo = DXO(data_kind=DataKind.METRICS, data=val_results)
        return metric_dxo.to_shareable()

    def _load_local_model(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        job_id = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
        run_dir = engine.get_workspace().get_run_dir(job_id)
        model_load_path = resolve_path_under_root(
            run_dir, os.path.join(self._model_dir, self._model_name), "model path"
        )
        try:
            np_data = np.load(model_load_path, allow_pickle=False)
        except Exception as e:
            self.log_error(fl_ctx, f"Unable to load local model: {secure_format_exception(e)}")
            return None

        model = ModelLearnable()
        model[NPConstants.NUMPY_KEY] = np_data

        return model

    def _save_local_model(self, fl_ctx: FLContext, model: dict):
        # Save local model
        engine = fl_ctx.get_engine()
        job_id = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
        run_dir = engine.get_workspace().get_run_dir(job_id)
        model_save_path = resolve_path_under_root(
            run_dir, os.path.join(self._model_dir, self._model_name), "model path"
        )
        model_path = os.path.dirname(model_save_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        np.save(model_save_path, model[NPConstants.NUMPY_KEY])
        self.log_info(fl_ctx, f"Saved numpy model to: {model_save_path}")
