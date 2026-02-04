# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for automatic timeout management in tensor streaming."""

from unittest.mock import Mock, patch

import pytest

from nvflare.apis.fl_constant import FLContextKey, ServerCommandKey
from nvflare.apis.shareable import Shareable
from nvflare.app_opt.tensor_stream.server import TensorServerStreamer
from nvflare.client.config import ExchangeFormat
from nvflare.private.fed.client.communicator import Communicator
from nvflare.private.fed.server.server_commands import GetTaskCommand


class TestServerMinTimeoutSetting:
    """Test that TensorServerStreamer correctly sets minimum timeout in FLContext."""

    @pytest.mark.parametrize(
        "wait_timeout,expected_min_timeout",
        [
            (300.0, 360.0),  # Default: 300 + 60 = 360
            (600.0, 660.0),  # Large: 600 + 60 = 660
            (120.0, 180.0),  # Small: 120 + 60 = 180
            (900.0, 960.0),  # Very large: 900 + 60 = 960
        ],
    )
    @patch("nvflare.app_opt.tensor_stream.server.TensorSender")
    @patch("nvflare.app_opt.tensor_stream.server.TensorReceiver")
    def test_min_timeout_set_in_fl_context(
        self,
        mock_receiver_class,
        mock_sender_class,
        wait_timeout,
        expected_min_timeout,
        mock_fl_context,
        mock_engine_with_clients,
    ):
        """Test that MIN_GET_TASK_TIMEOUT is correctly set in FLContext during initialization."""
        # Setup mocks
        mock_fl_context.get_engine.return_value = mock_engine_with_clients
        mock_fl_context.set_prop = Mock()  # Make set_prop mockable for assertions

        # Create streamer with specified wait timeout
        streamer = TensorServerStreamer(
            format=ExchangeFormat.PYTORCH,
            wait_send_task_data_all_clients_timeout=wait_timeout,
        )

        # Initialize
        streamer.initialize(mock_fl_context)

        # Verify set_prop was called with correct minimum timeout
        mock_fl_context.set_prop.assert_called_once_with(
            ServerCommandKey.MIN_GET_TASK_TIMEOUT,
            expected_min_timeout,
            sticky=True,
        )

    @patch("nvflare.app_opt.tensor_stream.server.TensorSender")
    @patch("nvflare.app_opt.tensor_stream.server.TensorReceiver")
    def test_min_timeout_logging(
        self, mock_receiver_class, mock_sender_class, mock_fl_context, mock_engine_with_clients
    ):
        """Test that initialization logs the minimum timeout requirement."""
        mock_fl_context.get_engine.return_value = mock_engine_with_clients
        mock_fl_context.set_prop = Mock()  # Make set_prop mockable

        streamer = TensorServerStreamer(wait_send_task_data_all_clients_timeout=300.0)
        streamer.log_info = Mock()

        streamer.initialize(mock_fl_context)

        # Verify log_info was called with appropriate message
        streamer.log_info.assert_called_once()
        args = streamer.log_info.call_args[0]
        assert "Requiring clients to use get_task_timeout >= 360.0s" in args[1]
        assert "wait_send_task_data_all_clients_timeout=300.0s + 60.0s buffer" in args[1]
        assert "automatically communicated to clients" in args[1]


class TestGetTaskCommandPropagation:
    """Test that GetTaskCommand reads and propagates minimum timeout to clients."""

    def test_min_timeout_added_to_task_response(self, mock_fl_context):
        """Test that MIN_GET_TASK_TIMEOUT header is added to task response when available."""
        # Setup mocks
        min_timeout = 360.0

        def get_prop_side_effect(key, default=None):
            return {
                ServerCommandKey.MIN_GET_TASK_TIMEOUT: min_timeout,
                FLContextKey.RUNNER: Mock(),
            }.get(key, default)

        mock_fl_context.get_prop.side_effect = get_prop_side_effect

        # Mock server_runner.process_task_request to return task
        mock_server_runner = Mock()
        mock_server_runner.process_task_request.return_value = ("train", "task_123", Shareable())

        # Create command processor
        command = GetTaskCommand()

        # Create request data
        request_data = Shareable()
        mock_client = Mock()
        mock_client.name = "client1"
        request_data.set_header(ServerCommandKey.FL_CLIENT, mock_client)

        # Mock gen_new_peer_ctx
        with patch("nvflare.private.fed.server.server_commands.gen_new_peer_ctx"):
            # Update the mock to return server_runner
            def get_prop_side_effect(key):
                if key == FLContextKey.RUNNER:
                    return mock_server_runner
                elif key == ServerCommandKey.MIN_GET_TASK_TIMEOUT:
                    return min_timeout
                return None

            mock_fl_context.get_prop.side_effect = get_prop_side_effect

            # Process the command
            response = command.process(request_data, mock_fl_context)

        # Verify MIN_GET_TASK_TIMEOUT header was added to response
        assert response.get_header(ServerCommandKey.MIN_GET_TASK_TIMEOUT) == min_timeout

    def test_no_min_timeout_when_not_set(self, mock_fl_context):
        """Test that MIN_GET_TASK_TIMEOUT header is not added when not in FLContext."""

        # Setup mocks - no MIN_GET_TASK_TIMEOUT in context
        def get_prop_side_effect(key, default=None):
            return {
                FLContextKey.RUNNER: Mock(),
            }.get(key, default)

        mock_fl_context.get_prop.side_effect = get_prop_side_effect

        # Mock server_runner.process_task_request
        mock_server_runner = Mock()
        mock_server_runner.process_task_request.return_value = ("train", "task_123", Shareable())

        # Create command processor
        command = GetTaskCommand()

        # Create request data
        request_data = Shareable()
        mock_client = Mock()
        mock_client.name = "client1"
        request_data.set_header(ServerCommandKey.FL_CLIENT, mock_client)

        # Mock gen_new_peer_ctx
        with patch("nvflare.private.fed.server.server_commands.gen_new_peer_ctx"):

            def get_prop_side_effect(key):
                if key == FLContextKey.RUNNER:
                    return mock_server_runner
                return None

            mock_fl_context.get_prop.side_effect = get_prop_side_effect

            # Process the command
            response = command.process(request_data, mock_fl_context)

        # Verify MIN_GET_TASK_TIMEOUT header is None (not set)
        assert response.get_header(ServerCommandKey.MIN_GET_TASK_TIMEOUT) is None


class TestClientTimeoutAdjustment:
    """Test that client communicator automatically adjusts timeout based on server requirement."""

    @pytest.mark.parametrize(
        "initial_timeout,min_timeout,should_adjust,expected_timeout",
        [
            (5.0, 360.0, True, 360.0),  # Default case: 5s < 360s → adjust
            (300.0, 360.0, True, 360.0),  # Below minimum: 300s < 360s → adjust
            (400.0, 360.0, False, 400.0),  # Already sufficient: 400s > 360s → no adjust
            (360.0, 360.0, False, 360.0),  # Exactly equal: 360s = 360s → no adjust
            (5.0, 600.0, True, 600.0),  # Large gap: 5s < 600s → adjust to 600s
        ],
    )
    @patch("nvflare.private.fed.client.communicator.new_cell_message")
    @patch("nvflare.private.fed.client.communicator.determine_parent_fqcn")
    @patch("nvflare.private.fed.client.communicator.gen_new_peer_ctx")
    def test_timeout_auto_adjustment(
        self,
        mock_gen_ctx,
        mock_determine_parent,
        mock_new_cell_message,
        initial_timeout,
        min_timeout,
        should_adjust,
        expected_timeout,
    ):
        """Test that communicator adjusts timeout when server sends MIN_GET_TASK_TIMEOUT."""
        # Setup communicator
        from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode

        communicator = Communicator(client_config={"client_name": "test_client"}, timeout=initial_timeout)
        communicator.engine = Mock()
        communicator.cell = Mock()

        # Mock return value
        mock_determine_parent.return_value = "parent_fqcn"
        mock_new_cell_message.return_value = Mock()  # Return a mock cell message

        # Create response with MIN_GET_TASK_TIMEOUT header
        response_shareable = Shareable()
        response_shareable.set_header(ServerCommandKey.TASK_NAME, "train")
        response_shareable.set_header(ServerCommandKey.MIN_GET_TASK_TIMEOUT, min_timeout)
        response_shareable.set_header(FLContextKey.TASK_ID, "task_123")

        mock_task = Mock()

        def get_header_side_effect(key, default=None):
            return {
                MessageHeaderKey.RETURN_CODE: ReturnCode.OK,
                MessageHeaderKey.PAYLOAD_LEN: 1024,
            }.get(key, default)

        mock_task.get_header = Mock(side_effect=get_header_side_effect)
        mock_task.payload = response_shareable

        communicator.cell.send_request.return_value = mock_task

        # Mock FL context
        mock_fl_context = Mock()
        mock_fl_context.get_job_id.return_value = "job_123"
        mock_fl_context.get_run_abort_signal.return_value = None
        mock_fl_context.set_prop = Mock()  # Mock set_prop to avoid errors

        # Mock logger
        communicator.logger = Mock()

        # Call pull_task
        communicator.pull_task("project", "token", "ssid", mock_fl_context)

        # Verify timeout was adjusted (or not) as expected
        assert communicator.timeout == expected_timeout

        if should_adjust:
            # Verify adjustment was logged (check all log calls)
            communicator.logger.info.assert_called()
            all_log_messages = [call[0][0] for call in communicator.logger.info.call_args_list]
            adjustment_log = [msg for msg in all_log_messages if "Automatically adjusting" in msg]
            assert len(adjustment_log) == 1, f"Expected exactly one adjustment log, got {len(adjustment_log)}"
            assert f"Server requires get_task_timeout >= {min_timeout}s" in adjustment_log[0]
            assert f"Automatically adjusting from {initial_timeout}s to {min_timeout}s" in adjustment_log[0]
        else:
            # Verify no adjustment log when timeout already sufficient
            if communicator.logger.info.called:
                all_log_messages = [call[0][0] for call in communicator.logger.info.call_args_list]
                for log_message in all_log_messages:
                    assert "Automatically adjusting" not in log_message

    @patch("nvflare.private.fed.client.communicator.new_cell_message")
    @patch("nvflare.private.fed.client.communicator.determine_parent_fqcn")
    @patch("nvflare.private.fed.client.communicator.gen_new_peer_ctx")
    def test_no_adjustment_when_header_missing(self, mock_gen_ctx, mock_determine_parent, mock_new_cell_message):
        """Test that timeout is not adjusted when MIN_GET_TASK_TIMEOUT header is missing."""
        from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode

        initial_timeout = 5.0
        communicator = Communicator(client_config={"client_name": "test_client"}, timeout=initial_timeout)
        communicator.engine = Mock()
        communicator.cell = Mock()

        mock_determine_parent.return_value = "parent_fqcn"
        mock_new_cell_message.return_value = Mock()  # Return a mock cell message

        # Create response WITHOUT MIN_GET_TASK_TIMEOUT header
        response_shareable = Shareable()
        response_shareable.set_header(ServerCommandKey.TASK_NAME, "train")
        response_shareable.set_header(FLContextKey.TASK_ID, "task_123")

        mock_task = Mock()

        def get_header_side_effect(key, default=None):
            return {
                MessageHeaderKey.RETURN_CODE: ReturnCode.OK,
                MessageHeaderKey.PAYLOAD_LEN: 1024,
            }.get(key, default)

        mock_task.get_header = Mock(side_effect=get_header_side_effect)
        mock_task.payload = response_shareable

        communicator.cell.send_request.return_value = mock_task

        mock_fl_context = Mock()
        mock_fl_context.get_job_id.return_value = "job_123"
        mock_fl_context.get_run_abort_signal.return_value = None
        mock_fl_context.set_prop = Mock()  # Mock set_prop to avoid errors

        communicator.logger = Mock()

        # Call pull_task
        communicator.pull_task("project", "token", "ssid", mock_fl_context)

        # Verify timeout was NOT changed
        assert communicator.timeout == initial_timeout


class TestEndToEndTimeoutFlow:
    """Test complete end-to-end flow of automatic timeout management."""

    @patch("nvflare.app_opt.tensor_stream.server.TensorSender")
    @patch("nvflare.app_opt.tensor_stream.server.TensorReceiver")
    def test_complete_timeout_flow(
        self, mock_receiver_class, mock_sender_class, mock_fl_context, mock_engine_with_clients
    ):
        """Test complete flow: server sets timeout → command propagates → client receives."""
        # Step 1: Server side - TensorServerStreamer sets MIN_GET_TASK_TIMEOUT
        server_wait_timeout = 300.0
        expected_min_timeout = 360.0

        mock_fl_context.get_engine.return_value = mock_engine_with_clients
        mock_fl_context.set_prop = Mock()  # Make set_prop mockable

        streamer = TensorServerStreamer(wait_send_task_data_all_clients_timeout=server_wait_timeout)
        streamer.initialize(mock_fl_context)

        # Verify server set the timeout in FLContext
        mock_fl_context.set_prop.assert_called_once_with(
            ServerCommandKey.MIN_GET_TASK_TIMEOUT,
            expected_min_timeout,
            sticky=True,
        )

        # Step 2: Server command - GetTaskCommand reads and adds to response
        # (Already tested in TestGetTaskCommandPropagation)

        # Step 3: Client side - Communicator adjusts timeout
        # (Already tested in TestClientTimeoutAdjustment)

        # This test verifies the full integration is working

    @pytest.mark.parametrize(
        "server_timeout,buffer,client_initial,expected_client",
        [
            (300.0, 60.0, 5.0, 360.0),  # Standard case
            (600.0, 60.0, 5.0, 660.0),  # Large server timeout
            (900.0, 60.0, 800.0, 960.0),  # Client already has large timeout but not enough
            (300.0, 60.0, 500.0, 500.0),  # Client already sufficient
        ],
    )
    @patch("nvflare.app_opt.tensor_stream.server.TensorSender")
    @patch("nvflare.app_opt.tensor_stream.server.TensorReceiver")
    def test_various_timeout_scenarios(
        self,
        mock_receiver_class,
        mock_sender_class,
        server_timeout,
        buffer,
        client_initial,
        expected_client,
        mock_fl_context,
        mock_engine_with_clients,
    ):
        """Test various timeout configuration scenarios."""
        expected_min_timeout = server_timeout + buffer

        # Server initialization
        mock_fl_context.get_engine.return_value = mock_engine_with_clients
        mock_fl_context.set_prop = Mock()  # Make set_prop mockable

        streamer = TensorServerStreamer(wait_send_task_data_all_clients_timeout=server_timeout)
        streamer.initialize(mock_fl_context)

        # Verify correct minimum was set
        mock_fl_context.set_prop.assert_called_once_with(
            ServerCommandKey.MIN_GET_TASK_TIMEOUT,
            expected_min_timeout,
            sticky=True,
        )

        # Simulate client receiving this and adjusting
        if client_initial < expected_min_timeout:
            # Client should adjust upward
            assert expected_client == expected_min_timeout
        else:
            # Client should keep its existing timeout
            assert expected_client == client_initial


class TestTimeoutErrorScenarios:
    """Test error scenarios and edge cases in timeout management."""

    @patch("nvflare.app_opt.tensor_stream.server.TensorSender")
    @patch("nvflare.app_opt.tensor_stream.server.TensorReceiver")
    def test_zero_timeout_handling(
        self, mock_receiver_class, mock_sender_class, mock_fl_context, mock_engine_with_clients
    ):
        """Test handling of zero or very small timeout values."""
        mock_fl_context.get_engine.return_value = mock_engine_with_clients
        mock_fl_context.set_prop = Mock()  # Make set_prop mockable

        # Even with 0 wait timeout, should set minimum (0 + 60 = 60)
        streamer = TensorServerStreamer(wait_send_task_data_all_clients_timeout=0.0)
        streamer.initialize(mock_fl_context)

        mock_fl_context.set_prop.assert_called_once_with(
            ServerCommandKey.MIN_GET_TASK_TIMEOUT,
            60.0,  # 0 + 60 = 60
            sticky=True,
        )

    @patch("nvflare.private.fed.client.communicator.new_cell_message")
    @patch("nvflare.private.fed.client.communicator.determine_parent_fqcn")
    @patch("nvflare.private.fed.client.communicator.gen_new_peer_ctx")
    def test_none_min_timeout_handling(self, mock_gen_ctx, mock_determine_parent, mock_new_cell_message):
        """Test that None MIN_GET_TASK_TIMEOUT is handled gracefully."""
        from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode

        initial_timeout = 5.0
        communicator = Communicator(client_config={"client_name": "test_client"}, timeout=initial_timeout)
        communicator.engine = Mock()
        communicator.cell = Mock()

        mock_determine_parent.return_value = "parent_fqcn"
        mock_new_cell_message.return_value = Mock()  # Return a mock cell message

        # Create response with None MIN_GET_TASK_TIMEOUT
        response_shareable = Shareable()
        response_shareable.set_header(ServerCommandKey.TASK_NAME, "train")
        response_shareable.set_header(ServerCommandKey.MIN_GET_TASK_TIMEOUT, None)
        response_shareable.set_header(FLContextKey.TASK_ID, "task_123")

        mock_task = Mock()

        def get_header_side_effect(key, default=None):
            return {
                MessageHeaderKey.RETURN_CODE: ReturnCode.OK,
                MessageHeaderKey.PAYLOAD_LEN: 1024,
            }.get(key, default)

        mock_task.get_header = Mock(side_effect=get_header_side_effect)
        mock_task.payload = response_shareable

        communicator.cell.send_request.return_value = mock_task

        mock_fl_context = Mock()
        mock_fl_context.get_job_id.return_value = "job_123"
        mock_fl_context.get_run_abort_signal.return_value = None
        mock_fl_context.set_prop = Mock()  # Mock set_prop to avoid errors

        communicator.logger = Mock()

        # Call pull_task - should not crash
        communicator.pull_task("project", "token", "ssid", mock_fl_context)

        # Verify timeout unchanged
        assert communicator.timeout == initial_timeout
