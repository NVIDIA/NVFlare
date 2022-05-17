# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.np.constants import NPConstants
from nvflare.app_common.np.np_trainer import NPTrainer as BaseNPTrainer


class NPTrainer(BaseNPTrainer):
    def __init__(
        self,
        delta=1,
        sleep_time=0,
        train_task_name=AppConstants.TASK_TRAIN,
        submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL,
        model_name="best_numpy.npy",
        model_dir="model",
    ):
        super().__init__(
            delta=delta,
            sleep_time=sleep_time,
            train_task_name=train_task_name,
            submit_model_task_name=submit_model_task_name,
            model_name=model_name,
            model_dir=model_dir,
        )

    def _train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal):
        # First we extract DXO from the shareable.
        try:
            incoming_dxo = from_shareable(shareable)
        except BaseException as e:
            self.system_panic(f"Unable to convert shareable to model definition. Exception {e.__str__()}", fl_ctx)
            return make_reply(ReturnCode.BAD_TASK_DATA)

        # Information about workflow is retrieved from the shareable header.
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND, None)
        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS, None)

        # Ensure that data is of type weights. Extract model data.
        if incoming_dxo.data_kind != DataKind.WEIGHTS:
            self.system_panic("Model DXO should be of kind DataKind.WEIGHTS.", fl_ctx)
            return make_reply(ReturnCode.BAD_TASK_DATA)
        weights = incoming_dxo.data

        # Display properties.
        self.log_info(fl_ctx, f"Incoming data kind: {incoming_dxo.data_kind}")
        self.log_info(fl_ctx, f"Model: \n{weights}")
        self.log_info(fl_ctx, f"Current Round: {current_round}")
        self.log_info(fl_ctx, f"Total Rounds: {total_rounds}")
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")

        # Check abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # Doing some dummy training.
        new_weights = {}
        if weights:
            if NPConstants.NUMPY_KEY in weights:
                new_weights[NPConstants.NUMPY_KEY] = weights[NPConstants.NUMPY_KEY] + self._delta
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
            self._save_local_model(fl_ctx, weights)
        except Exception as e:
            self.log_error(fl_ctx, f"Exception in saving local model: {e}.")

        self.log_info(
            fl_ctx,
            f"Model after training: {new_weights}",
        )

        # Checking abort signal again.
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        weights_diff = {k: new_weights[k] - weights[k] for k in new_weights.keys()}

        # Prepare a DXO for our updated model. Create shareable and return
        outgoing_dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data=weights_diff, meta={})
        return outgoing_dxo.to_shareable()
