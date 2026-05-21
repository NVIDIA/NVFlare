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
from nvflare.app_common.app_constant import AppConstants
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


def make_client_task(result):
    client_task = ClientTask(
        client=Client("site-1", SITE_1_ID),
        task=Task(
            name="__test_task",
            data=Shareable(),
        ),
    )
    client_task.result = result
    return client_task


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

            with (
                patch.object(ctl.shareable_generator, "learnable_to_shareable") as mock_method1,
                patch.object(ctl.shareable_generator, "shareable_to_learnable") as mock_method2,
            ):
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
        with (
            patch.object(ctl, "cancel_task") as mock_method,
            patch.object(ctl.shareable_generator, "learnable_to_shareable") as mock_method1,
            patch.object(ctl.shareable_generator, "shareable_to_learnable") as mock_method2,
        ):
            mock_method1.return_value = Shareable()
            mock_method2.return_value = Learnable()

            client_task = make_client_task(return_result)
            ctl._process_result(client_task, fl_ctx)
            mock_method.assert_called_once()
            assert ctl._is_done is True

    def test_process_result_stops_on_non_ok_rc_without_converting_shareable(self):
        ctl = CyclicController(persist_every_n_rounds=0, snapshot_every_n_rounds=0, num_rounds=1)
        ctl.shareable_generator = Mock()

        fl_ctx = FLContext()
        result = Shareable()
        result.set_return_code(ReturnCode.EXECUTION_EXCEPTION)
        client_task = make_client_task(result)

        with (
            patch.object(ctl, "cancel_task") as mock_cancel,
            patch.object(ctl.shareable_generator, "learnable_to_shareable") as mock_to_shareable,
            patch.object(ctl.shareable_generator, "shareable_to_learnable") as mock_to_learnable,
        ):
            ctl._process_result(client_task, fl_ctx)

            mock_cancel.assert_called_once()
            mock_to_learnable.assert_not_called()
            mock_to_shareable.assert_not_called()
            assert ctl._is_done is True

    def test_process_result_converts_allowed_early_termination_before_stopping(self):
        ctl = CyclicController(
            persist_every_n_rounds=0, snapshot_every_n_rounds=0, num_rounds=1, allow_early_termination=True
        )
        ctl.shareable_generator = Mock()

        fl_ctx = FLContext()
        result = gen_shareable(is_early_termination=True)
        client_task = make_client_task(result)
        learnable = Learnable()

        with (
            patch.object(ctl, "cancel_task") as mock_cancel,
            patch.object(ctl.shareable_generator, "learnable_to_shareable") as mock_to_shareable,
            patch.object(ctl.shareable_generator, "shareable_to_learnable") as mock_to_learnable,
        ):
            mock_to_learnable.return_value = learnable

            ctl._process_result(client_task, fl_ctx)

            mock_cancel.assert_called_once()
            mock_to_learnable.assert_called_once_with(result, fl_ctx)
            mock_to_shareable.assert_not_called()
            assert ctl._last_learnable is learnable
            assert ctl._is_done is True

    def test_process_result_converts_disallowed_early_termination_before_continuing(self):
        ctl = CyclicController(
            persist_every_n_rounds=0, snapshot_every_n_rounds=0, num_rounds=1, allow_early_termination=False
        )
        ctl.shareable_generator = Mock()
        ctl._last_learnable = Learnable()
        ctl._current_round = 3

        fl_ctx = FLContext()
        result = gen_shareable(is_early_termination=True)
        client_task = make_client_task(result)
        learnable = Learnable()
        next_shareable = Shareable()

        with (
            patch.object(ctl, "cancel_task") as mock_cancel,
            patch.object(ctl.shareable_generator, "learnable_to_shareable") as mock_to_shareable,
            patch.object(ctl.shareable_generator, "shareable_to_learnable") as mock_to_learnable,
        ):
            mock_to_learnable.return_value = learnable
            mock_to_shareable.return_value = next_shareable

            ctl._process_result(client_task, fl_ctx)

            mock_cancel.assert_not_called()
            mock_to_learnable.assert_called_once_with(result, fl_ctx)
            mock_to_shareable.assert_called_once_with(learnable, fl_ctx)
            assert ctl._last_learnable is learnable
            assert client_task.task.data is next_shareable
            assert client_task.task.data.get_header(AppConstants.CURRENT_ROUND) == ctl._current_round
            assert client_task.task.data.get_header(AppConstants.NUM_ROUNDS) == ctl._num_rounds
            assert client_task.task.data.get_cookie(AppConstants.CONTRIBUTION_ROUND) == ctl._current_round
            assert ctl._is_done is False

    def test_process_result_converts_ok_result(self):
        ctl = CyclicController(persist_every_n_rounds=0, snapshot_every_n_rounds=0, num_rounds=1)
        ctl.shareable_generator = Mock()
        ctl._current_round = 2

        fl_ctx = FLContext()
        result = Shareable()
        client_task = make_client_task(result)
        learnable = Learnable()
        next_shareable = Shareable()

        with (
            patch.object(ctl, "cancel_task") as mock_cancel,
            patch.object(ctl.shareable_generator, "learnable_to_shareable") as mock_to_shareable,
            patch.object(ctl.shareable_generator, "shareable_to_learnable") as mock_to_learnable,
        ):
            mock_to_learnable.return_value = learnable
            mock_to_shareable.return_value = next_shareable

            ctl._process_result(client_task, fl_ctx)

            mock_cancel.assert_not_called()
            mock_to_learnable.assert_called_once_with(result, fl_ctx)
            mock_to_shareable.assert_called_once_with(learnable, fl_ctx)
            assert ctl._last_learnable is learnable
            assert client_task.task.data is next_shareable
            assert client_task.task.data.get_header(AppConstants.CURRENT_ROUND) == ctl._current_round
            assert client_task.task.data.get_header(AppConstants.NUM_ROUNDS) == ctl._num_rounds
            assert client_task.task.data.get_cookie(AppConstants.CONTRIBUTION_ROUND) == ctl._current_round
