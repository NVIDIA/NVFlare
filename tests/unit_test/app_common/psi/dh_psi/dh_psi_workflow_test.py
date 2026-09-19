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
from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, Task, TaskCompletionStatus
from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.wf_comm_server import _TASK_KEY_MANAGER, WFCommServer
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import PSIConst
from nvflare.app_common.psi.dh_psi.dh_psi_workflow import DhPSIWorkFlow, SiteSize
from nvflare.app_common.psi.psi_controller import PSIController
from nvflare.app_common.workflows.broadcast_operator import BroadcastAndWait


class TestDhPSIWorkflow:
    def test_get_ordered_sites(self):
        wf = DhPSIWorkFlow()
        data_1 = {PSIConst.ITEMS_SIZE: 1000}
        data_2 = {PSIConst.ITEMS_SIZE: 500}
        data_3 = {PSIConst.ITEMS_SIZE: 667}
        dxo_1 = DXO(data_kind=DataKind.PSI, data=data_1)
        dxo_2 = DXO(data_kind=DataKind.PSI, data=data_2)
        dxo_3 = DXO(data_kind=DataKind.PSI, data=data_3)
        results = {"site-1": dxo_1, "site-2": dxo_2, "site-3": dxo_3}
        ordered_sites = wf.get_ordered_sites(results=results)

        assert ordered_sites[0].size <= ordered_sites[1].size
        assert ordered_sites[1].size <= ordered_sites[2].size

    def test_prepare_log_keeps_response_count_without_site_sizes(self):
        wf = DhPSIWorkFlow()
        wf.fl_ctx = FLContext()
        wf.fl_ctx.get_engine = MagicMock()
        wf.fl_ctx.get_engine.return_value.get_clients.return_value = ["private-site-alpha", "private-site-beta"]
        wf.controller = MagicMock()
        wf.log_info = MagicMock()
        results = {
            "private-site-alpha": DXO(data_kind=DataKind.PSI, data={PSIConst.ITEMS_SIZE: 91001}),
            "private-site-beta": DXO(data_kind=DataKind.PSI, data={PSIConst.ITEMS_SIZE: 92002}),
        }

        with patch("nvflare.app_common.psi.dh_psi.dh_psi_workflow.BroadcastAndWait") as broadcast_and_wait:
            broadcast_and_wait.return_value.broadcast_and_wait.return_value = results
            wf.prepare_sites(Signal())

        message = wf.log_info.call_args.args[1]
        assert message == f"{PSIConst.TASK_PREPARE} received 2 participant responses"
        assert "private-site" not in message
        assert "91001" not in message
        assert "92002" not in message

    def test_prepare_filters_empty_participant_without_exposing_private_values(self):
        wf = DhPSIWorkFlow()
        wf.fl_ctx = FLContext()
        wf.fl_ctx.get_engine = MagicMock()
        wf.fl_ctx.get_engine.return_value.get_clients.return_value = ["private-site-empty", "private-site-active"]
        wf.controller = MagicMock()
        wf.log_info = MagicMock()
        results = {
            "private-site-empty": DXO(data_kind=DataKind.PSI, data={PSIConst.ITEMS_SIZE: 0}),
            "private-site-active": DXO(data_kind=DataKind.PSI, data={PSIConst.ITEMS_SIZE: 91001}),
        }

        with patch("nvflare.app_common.psi.dh_psi.dh_psi_workflow.BroadcastAndWait") as broadcast_and_wait:
            broadcast_and_wait.return_value.broadcast_and_wait.return_value = results
            wf.prepare_sites(Signal())

        assert wf.ordered_sites == [SiteSize("private-site-active", 91001)]
        message = wf.log_info.call_args.args[1]
        assert message == f"{PSIConst.TASK_PREPARE} received 2 participant responses"
        assert "private-site" not in message
        assert "91001" not in message

    def test_prepare_rejects_all_empty_participants_with_value_free_error(self):
        wf = DhPSIWorkFlow()
        wf.fl_ctx = FLContext()
        wf.fl_ctx.get_engine = MagicMock()
        wf.fl_ctx.get_engine.return_value.get_clients.return_value = ["private-site-alpha", "private-site-beta"]
        wf.controller = MagicMock()
        wf.log_info = MagicMock()
        abort_signal = Signal()
        results = {
            "private-site-alpha": DXO(data_kind=DataKind.PSI, data={PSIConst.ITEMS_SIZE: 0}),
            "private-site-beta": DXO(data_kind=DataKind.PSI, data={PSIConst.ITEMS_SIZE: 0}),
        }

        with (
            patch("nvflare.app_common.psi.dh_psi.dh_psi_workflow.BroadcastAndWait") as broadcast_and_wait,
            pytest.raises(RuntimeError) as error,
        ):
            broadcast_and_wait.return_value.broadcast_and_wait.return_value = results
            wf.prepare_sites(abort_signal)

        message = str(error.value)
        assert abort_signal.triggered
        assert "no item" in message.lower()
        assert "private-site" not in message

    def test_prepare_rejects_partial_results_without_identifying_missing_participant(self):
        wf = DhPSIWorkFlow()
        wf.fl_ctx = FLContext()
        wf.fl_ctx.get_engine = MagicMock()
        wf.fl_ctx.get_engine.return_value.get_clients.return_value = [
            "private-site-alpha",
            "private-site-beta",
        ]
        wf.controller = MagicMock()
        wf.log_info = MagicMock()
        results = {
            "private-site-alpha": DXO(data_kind=DataKind.PSI, data={PSIConst.ITEMS_SIZE: 91001}),
        }

        with (
            patch("nvflare.app_common.psi.dh_psi.dh_psi_workflow.BroadcastAndWait") as broadcast_and_wait,
            pytest.raises(RuntimeError) as error,
        ):
            broadcast_and_wait.return_value.broadcast_and_wait.return_value = results
            wf.prepare_sites(Signal())

        message = str(error.value)
        assert "incomplete" in message
        assert "private-site" not in message
        assert "91001" not in message

    def test_run_logs_phase_counts_without_site_sizes(self):
        wf = DhPSIWorkFlow()
        wf.fl_ctx = FLContext()
        wf.log_info = MagicMock()
        wf.ordered_sites = [SiteSize("private-site-alpha", 91001), SiteSize("private-site-beta", 92002)]
        wf.forward_pass = MagicMock(return_value=SiteSize("private-site-beta", 50001))
        wf.forward_processed = {"private-site-beta": 50001}
        wf.backward_pass = MagicMock(return_value={"private-site-alpha": 50001})
        wf.check_processed_sites = MagicMock()
        wf.check_final_intersection_sizes = MagicMock()
        wf.log_pass_time_taken = MagicMock()

        wf.run(Signal())

        messages = "\n".join(call.args[1] for call in wf.log_info.call_args_list)
        assert "ordered 2 PSI participants" in messages
        assert "forward pass retained 1 intermediate-result holders" in messages
        assert "backward pass processed 1 participants" in messages
        assert "private-site" not in messages
        assert "91001" not in messages
        assert "92002" not in messages
        assert "50001" not in messages

    @pytest.mark.parametrize("check_name", ["check_processed_sites", "check_final_intersection_sizes"])
    def test_validation_errors_do_not_include_site_sizes(self, check_name):
        wf = DhPSIWorkFlow()
        wf.ordered_sites = [SiteSize("private-site-alpha", 91001), SiteSize("private-site-beta", 92002)]
        wf.backward_processed = {"private-site-alpha": 50001, "private-site-beta": 50002}

        with pytest.raises(RuntimeError) as error:
            if check_name == "check_processed_sites":
                wf.check_processed_sites(SiteSize("private-site-beta", 50001), {"private-site-alpha": 1})
            else:
                wf.check_final_intersection_sizes(SiteSize("private-site-beta", 50001))

        message = str(error.value)
        assert "private-site" not in message
        assert "91001" not in message
        assert "92002" not in message
        assert "50001" not in message
        assert "50002" not in message

    def test_missing_pairwise_input_raises_value_free_error(self):
        wf = DhPSIWorkFlow()
        ordered_sites = [SiteSize("private-site-alpha", 91001), SiteSize("private-site-beta", 92002)]

        with pytest.raises(RuntimeError) as error:
            wf.pairwise_requests(ordered_sites, setup_msgs={})

        message = str(error.value)
        assert "incomplete" in message
        assert "private-site" not in message
        assert "91001" not in message
        assert "92002" not in message

    def test_missing_size_key_raises_value_free_error(self):
        wf = DhPSIWorkFlow()
        wf.abort_signal = Signal()
        wf.prepare_setup_messages = MagicMock(return_value={})
        intersect_site = SiteSize("private-intersection-holder", 50001)
        ordered_sites = [intersect_site, SiteSize("private-site-alpha", 91001)]

        with pytest.raises(RuntimeError) as error:
            wf.parallel_backward_pass(ordered_sites, intersect_site)

        message = str(error.value)
        assert "incomplete" in message
        assert "private-" not in message
        assert "50001" not in message
        assert "91001" not in message

    def test_partial_result_map_raises_value_free_error(self):
        wf = DhPSIWorkFlow()
        wf.abort_signal = Signal()
        operator = MagicMock()
        operator.broadcast_and_wait.return_value = {}
        wf._new_broadcast_operator = MagicMock(return_value=operator)

        with pytest.raises(RuntimeError) as error:
            wf.process_requests(SiteSize("private-site-alpha", 91001), {"request": "encrypted"})

        message = str(error.value)
        assert "incomplete" in message
        assert "private-site-alpha" not in message
        assert "91001" not in message

    def test_partial_nested_response_map_raises_before_intersection(self):
        wf = DhPSIWorkFlow()
        wf.abort_signal = Signal()
        operator = MagicMock()
        operator.broadcast_and_wait.return_value = {
            "private-intersection-holder": DXO(
                data_kind=DataKind.PSI,
                data={PSIConst.RESPONSE_MSG: {"private-site-alpha": "encrypted-response"}},
            )
        }
        wf._new_broadcast_operator = MagicMock(return_value=operator)
        request_msgs = {
            "private-site-alpha": "encrypted-request-a",
            "private-site-beta": "encrypted-request-b",
        }

        with pytest.raises(RuntimeError) as error:
            wf.process_requests(SiteSize("private-intersection-holder", 50001), request_msgs)

        message = str(error.value)
        assert "incomplete" in message
        assert "private-site" not in message
        assert "50001" not in message

    def test_malformed_result_map_raises_value_free_error(self):
        wf = DhPSIWorkFlow()
        wf.abort_signal = Signal()
        operator = MagicMock()
        operator.multicasts_and_wait.return_value = {
            "private-site-alpha": DXO(data_kind=DataKind.PSI, data={}),
        }
        wf._new_broadcast_operator = MagicMock(return_value=operator)
        ordered_sites = [SiteSize("private-site-alpha", 91001), SiteSize("private-site-beta", 92002)]

        with pytest.raises(RuntimeError) as error:
            wf.pairwise_setup(ordered_sites)

        message = str(error.value)
        assert "malformed" in message
        assert "private-site" not in message
        assert "91001" not in message
        assert "92002" not in message

    @pytest.mark.parametrize("log_client_names", [False, True])
    def test_result_callback_participant_identity_logging(self, log_client_names, caplog):
        wf = DhPSIWorkFlow()
        wf.fl_ctx = FLContext()
        wf.controller = MagicMock()
        if log_client_names:
            bop = BroadcastAndWait(wf.fl_ctx, wf.controller)
        else:
            bop = wf._new_broadcast_operator()

        callback_ctx = self._callback_context()
        peer_ctx = callback_ctx.get_peer_context()

        task = Task(name=PSIConst.TASK, data=Shareable())
        client_task = ClientTask(Client("private-site-alpha", "token"), task)
        client_task.result = DXO(data_kind=DataKind.PSI, data={PSIConst.ITEMS_SIZE: 91001}).to_shareable()

        with caplog.at_level("INFO"):
            bop.results_cb(client_task, callback_ctx)

        messages = caplog.text
        assert ("private-site-alpha" in messages) is log_client_names
        assert "91001" not in messages
        assert f"Processing {PSIConst.TASK}" in messages
        assert "Received result" in messages
        assert "private-site-alpha" in bop.results
        assert callback_ctx.get_peer_context() is peer_ctx

    @pytest.mark.parametrize(
        "return_code,raises",
        [(ReturnCode.EXECUTION_EXCEPTION, False), (ReturnCode.EXECUTION_RESULT_ERROR, True)],
    )
    def test_psi_error_callback_does_not_log_participant_identity(self, return_code, raises, caplog):
        wf = DhPSIWorkFlow()
        wf.fl_ctx = FLContext()
        wf.controller = PSIController(psi_workflow_id="psi_workflow")
        bop = wf._new_broadcast_operator()
        callback_ctx = self._callback_context()
        peer_ctx = callback_ctx.get_peer_context()

        task = Task(name=PSIConst.TASK, data=Shareable())
        client_task = ClientTask(Client("private-site-alpha", "token"), task)
        client_task.result = make_reply(return_code)

        with caplog.at_level("INFO"):
            if raises:
                with pytest.raises(ValueError) as error:
                    bop.results_cb(client_task, callback_ctx)
                exception_message = str(error.value)
            else:
                bop.results_cb(client_task, callback_ctx)
                exception_message = ""

        combined = f"{caplog.text}\n{exception_message}"
        assert "private-site-alpha" not in combined
        assert "a PSI participant" in combined
        assert callback_ctx.get_peer_context() is peer_ctx

    @pytest.mark.parametrize("task_mode", ["broadcast", "multicast"])
    @pytest.mark.parametrize("log_client_names", [False, True])
    def test_workflow_communicator_callback_error_logging_respects_identity_policy(
        self, task_mode, log_client_names, caplog
    ):
        controller = PSIController(psi_workflow_id="psi_workflow")
        bop = BroadcastAndWait(FLContext(), controller, log_client_names=log_client_names)
        client = Client("private-site-alpha", "token")
        if task_mode == "broadcast":
            controller.broadcast_and_wait = MagicMock()
            bop.broadcast_and_wait(PSIConst.TASK, Shareable(), FLContext(), targets=[client])
            task = controller.broadcast_and_wait.call_args.args[0]
        else:
            task = bop.get_tasks(PSIConst.TASK, {client.name: Shareable()})[client.name]
        task.props[_TASK_KEY_MANAGER] = MagicMock()
        client_task = ClientTask(client, task)
        communicator = WFCommServer()
        communicator._client_task_map[client_task.id] = client_task
        callback_ctx = self._callback_context()

        with caplog.at_level("ERROR"):
            communicator.process_submission(
                client=client,
                task_name=PSIConst.TASK,
                task_id=client_task.id,
                result=make_reply(ReturnCode.EXECUTION_RESULT_ERROR),
                fl_ctx=callback_ctx,
            )

        assert task.completion_status == TaskCompletionStatus.ERROR
        assert ("private-site-alpha" in caplog.text) is log_client_names
        assert "a PSI participant" in caplog.text

    @staticmethod
    def _callback_context():
        fl_ctx = FLContext()
        fl_ctx.put(ReservedKey.IDENTITY_NAME, "server", private=False, sticky=False)
        fl_ctx.put(ReservedKey.ENGINE, MagicMock(), private=True, sticky=False)
        peer_ctx = FLContext()
        peer_ctx.put(ReservedKey.IDENTITY_NAME, "private-site-alpha", private=False, sticky=False)
        fl_ctx.set_peer_context(peer_ctx)
        return fl_ctx
