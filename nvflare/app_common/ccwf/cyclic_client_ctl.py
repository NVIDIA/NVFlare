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
import random

from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.ccwf.client_ctl import ClientSideController
from nvflare.app_common.ccwf.common import Constant, CyclicOrder, ResultType, rotate_to_front
from nvflare.fuel.utils.validation_utils import check_non_empty_str


class CyclicClientController(ClientSideController):
    def __init__(
        self,
        task_name_prefix=Constant.TN_PREFIX_CYCLIC,
        learn_task_name=AppConstants.TASK_TRAIN,
        persistor_id=AppConstants.DEFAULT_PERSISTOR_ID,
        shareable_generator_id=AppConstants.DEFAULT_SHAREABLE_GENERATOR_ID,
        learn_task_check_interval=Constant.LEARN_TASK_CHECK_INTERVAL,
        learn_task_abort_timeout=Constant.LEARN_TASK_ABORT_TIMEOUT,
        learn_task_ack_timeout=Constant.LEARN_TASK_ACK_TIMEOUT,
        final_result_ack_timeout=Constant.FINAL_RESULT_ACK_TIMEOUT,
    ):
        check_non_empty_str("learn_task_name", learn_task_name)
        check_non_empty_str("persistor_id", persistor_id)
        check_non_empty_str("shareable_generator_id", shareable_generator_id)

        super().__init__(
            task_name_prefix=task_name_prefix,
            learn_task_name=learn_task_name,
            persistor_id=persistor_id,
            shareable_generator_id=shareable_generator_id,
            learn_task_check_interval=learn_task_check_interval,
            learn_task_abort_timeout=learn_task_abort_timeout,
            learn_task_ack_timeout=learn_task_ack_timeout,
            final_result_ack_timeout=final_result_ack_timeout,
            allow_busy_task=False,
        )

    @staticmethod
    def _set_task_headers(task_data: Shareable, num_rounds, current_round, client_order):
        task_data.set_header(AppConstants.NUM_ROUNDS, num_rounds)
        task_data.set_header(AppConstants.CURRENT_ROUND, current_round)
        task_data.set_header(Constant.CLIENT_ORDER, client_order)

    def start_workflow(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        clients = self.get_config_prop(Constant.CLIENTS)
        # make sure the starting client is the 1st
        rotate_to_front(self.me, clients)
        cyclic_order = self.get_config_prop(Constant.ORDER)
        self.log_info(fl_ctx, f"Starting cyclic workflow on clients {clients} with order {cyclic_order} ")
        self._set_task_headers(
            task_data=shareable,
            num_rounds=self.get_config_prop(AppConstants.NUM_ROUNDS),
            current_round=self.get_config_prop(Constant.START_ROUND, 0),
            client_order=clients,
        )
        self.set_learn_task(task_data=shareable, fl_ctx=fl_ctx)
        return make_reply(ReturnCode.OK)

    def do_learn_task(self, name: str, data: Shareable, fl_ctx: FLContext, abort_signal: Signal):
        # set status report of starting task
        current_round = data.get_header(AppConstants.CURRENT_ROUND)
        self.update_status(
            last_round=current_round,
            action="start_learn_task",
        )

        # need to prepare the GLOBAL_MODEL prop in case the shareable generator needs it
        # for shareable_to_learnable after training.
        # Note: the "data" shareable contains full weight before training.
        # However, the training process may only return weight diffs. To convert to full weights again,
        # the original weights (GLOBAL_MODEL prop) are needed.
        global_weights = self.shareable_generator.shareable_to_learnable(data, fl_ctx)
        fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, global_weights, private=True, sticky=True)

        data.set_header(FLContextKey.TASK_NAME, name)

        # execute the task
        result = self.execute_learn_task(data, fl_ctx, abort_signal)

        rc = result.get_return_code(ReturnCode.OK)
        if rc != ReturnCode.OK:
            self.log_error(fl_ctx, f"learn executor failed: {rc}")
            self.update_status(action="do_learn_task", error=rc)
            return

        self.last_result = result
        self.last_round = current_round

        # see whether we need to send to next leg
        num_rounds = data.get_header(AppConstants.NUM_ROUNDS)
        current_round = data.get_header(AppConstants.CURRENT_ROUND)
        client_order = data.get_header(Constant.CLIENT_ORDER)

        all_done = False
        assert isinstance(client_order, list)
        my_idx = client_order.index(self.me)

        if my_idx == len(client_order) - 1:
            # I'm the last leg
            num_rounds_done = current_round - self.get_config_prop(Constant.START_ROUND, 0) + 1
            if num_rounds_done >= num_rounds:
                # The RR is done!
                self.log_info(fl_ctx, f"Cyclic Done: number of rounds completed {num_rounds_done}")
                all_done = True
            else:
                # decide the next round order
                cyclic_order = self.get_config_prop(Constant.ORDER)
                if cyclic_order == CyclicOrder.RANDOM:
                    random.shuffle(client_order)
                    # make sure I'm not the first in the new order
                    if client_order[0] == self.me:
                        # put me at the end
                        client_order.pop(0)
                        client_order.append(self.me)
                    result.set_header(Constant.CLIENT_ORDER, client_order)

                current_round += 1
                self.log_info(fl_ctx, f"Starting new round {current_round} on clients: {client_order}")

        last_learnable = self.shareable_generator.shareable_to_learnable(result, fl_ctx)
        if all_done:
            self.record_last_result(fl_ctx, self.last_round, last_learnable)
            self.broadcast_final_result(fl_ctx, ResultType.LAST, last_learnable, round_num=self.last_round)
            return

        # send to next leg
        if my_idx < len(client_order) - 1:
            next_client = client_order[my_idx + 1]
        else:
            next_client = client_order[0]

        next_task_data = self.shareable_generator.learnable_to_shareable(last_learnable, fl_ctx)
        self._set_task_headers(next_task_data, num_rounds, current_round, client_order)
        sent = self.send_learn_task(
            targets=[next_client],
            request=next_task_data,
            fl_ctx=fl_ctx,
        )
        if sent:
            self.log_info(fl_ctx, f"sent learn request to next client {next_client}")
