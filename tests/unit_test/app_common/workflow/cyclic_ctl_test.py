# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


import uuid
from unittest.mock import Mock, patch

import pytest

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, Task
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learnable import Learnable
from nvflare.app_common.workflows.cyclic_ctl import CyclicController, RelayOrder

SITE_1_ID = uuid.uuid4()
SITE_2_ID = uuid.uuid4()
SITE_3_ID = uuid.uuid4()

ORDER_TEST_CASES = [
    (
        RelayOrder.FIXED,
        [Client("site-1", SITE_1_ID), Client("site-2", SITE_2_ID)],
        [Client("site-1", SITE_1_ID), Client("site-2", SITE_2_ID)],
    ),
    (
        ["site-1", "site-2"],
        [Client("site-1", SITE_1_ID), Client("site-2", SITE_2_ID)],
        [Client("site-1", SITE_1_ID), Client("site-2", SITE_2_ID)],
    ),
    (
        ["site-2", "site-1"],
        [Client("site-1", SITE_1_ID), Client("site-2", SITE_2_ID)],
        [Client("site-2", SITE_2_ID), Client("site-1", SITE_1_ID)],
    ),
    (
        ["site-2", "site-1", "site-3"],
        [Client("site-3", SITE_3_ID), Client("site-1", SITE_1_ID), Client("site-2", SITE_2_ID)],
        [Client("site-2", SITE_2_ID), Client("site-1", SITE_1_ID), Client("site-3", SITE_3_ID)],
    ),
]


def gen_shareable(is_early_termination: bool = False, is_not_shareable: bool = False):
    if is_not_shareable:
        return [1, 2, 3]
    return_result = Shareable()
    if is_early_termination:
        return_result.set_return_code(ReturnCode.EARLY_TERMINATION)
    return return_result


PROCESS_RESULT_TEST_CASES = [gen_shareable(is_early_termination=True), gen_shareable(is_not_shareable=True)]


class TestCyclicController:
    @pytest.mark.parametrize("order,active_clients,expected_result", ORDER_TEST_CASES)
    def test_get_relay_orders(self, order, active_clients, expected_result):
        ctl = CyclicController(order=order)
        ctx = FLContext()
        ctl._participating_clients = active_clients
        targets = ctl._get_relay_orders(ctx)
        for c, e_c in zip(targets, expected_result):
            assert c.name == e_c.name
            assert c.token == e_c.token

    def test_control_flow_call_relay_and_wait(self):

        with patch("nvflare.app_common.workflows.cyclic_ctl.CyclicController.relay_and_wait") as mock_method:
            ctl = CyclicController(persist_every_n_rounds=0, snapshot_every_n_rounds=0, num_rounds=1)
            ctl.shareable_generator = Mock()
            ctl._participating_clients = [
                Client("site-3", SITE_3_ID),
                Client("site-1", SITE_1_ID),
                Client("site-2", SITE_2_ID),
            ]

            abort_signal = Signal()
            fl_ctx = FLContext()

            with patch.object(ctl.shareable_generator, "learnable_to_shareable") as mock_method1, patch.object(
                ctl.shareable_generator, "shareable_to_learnable"
            ) as mock_method2:
                mock_method1.return_value = Shareable()
                mock_method2.return_value = Learnable()

                ctl.control_flow(abort_signal, fl_ctx)

                mock_method.assert_called_once()

    @pytest.mark.parametrize("return_result", PROCESS_RESULT_TEST_CASES)
    def test_process_result(self, return_result):
        ctl = CyclicController(
            persist_every_n_rounds=0, snapshot_every_n_rounds=0, num_rounds=1, allow_early_termination=True
        )
        ctl.shareable_generator = Mock()
        ctl._participating_clients = [
            Client("site-3", SITE_3_ID),
            Client("site-1", SITE_1_ID),
            Client("site-2", SITE_2_ID),
        ]

        fl_ctx = FLContext()
        with patch.object(ctl, "cancel_task") as mock_method, patch.object(
            ctl.shareable_generator, "learnable_to_shareable"
        ) as mock_method1, patch.object(ctl.shareable_generator, "shareable_to_learnable") as mock_method2:
            mock_method1.return_value = Shareable()
            mock_method2.return_value = Learnable()

            client_task = ClientTask(
                client=Mock(),
                task=Task(
                    name="__test_task",
                    data=Shareable(),
                ),
            )
            client_task.result = return_result
            ctl._process_result(client_task, fl_ctx)
            mock_method.assert_called_once()
            assert ctl._is_done is True
