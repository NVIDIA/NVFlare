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

from unittest.mock import Mock, patch

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_opt.tensor_stream.receiver import TensorReceiver
from nvflare.app_opt.tensor_stream.sender import TensorSender
from nvflare.app_opt.tensor_stream.server import TensorServerStreamer
from nvflare.client.config import ExchangeFormat


class TestTensorServerStreamer:
    """Test cases for TensorServerStreamer class."""

    @pytest.mark.parametrize(
        "format_type,tasks,entry_timeout,wait_all_clients_timeout,expected_tasks",
        [
            (ExchangeFormat.PYTORCH, None, 30.0, 300.0, ["train"]),  # Default values
            (ExchangeFormat.NUMPY, ["custom_train"], 10.0, 120.0, ["custom_train"]),  # Custom values
            (ExchangeFormat.PYTORCH, ["train", "validate"], 45.0, 600.0, ["train", "validate"]),  # Multiple tasks
        ],
    )
    def test_init_parameters(self, format_type, tasks, entry_timeout, wait_all_clients_timeout, expected_tasks):
        """Test TensorServerStreamer initialization with various parameters."""
        streamer = TensorServerStreamer(
            format=format_type,
            tasks=tasks,
            entry_timeout=entry_timeout,
            wait_send_task_data_all_clients_timeout=wait_all_clients_timeout,
        )

        assert streamer.format == format_type
        assert streamer.tasks == expected_tasks
        assert streamer.entry_timeout == entry_timeout
        assert streamer.wait_task_data_sent_to_all_clients_timeout == wait_all_clients_timeout
        # All components should be None before initialization
        assert streamer.engine is None
        assert streamer.sender is None
        assert streamer.receiver is None
        # Counters are now defaultdict instances
        assert isinstance(streamer.num_task_data_sent, dict)
        assert isinstance(streamer.num_task_skipped, dict)
        assert isinstance(streamer.data_cleaned, dict)

    @patch("nvflare.app_opt.tensor_stream.server.TensorSender")
    @patch("nvflare.app_opt.tensor_stream.server.TensorReceiver")
    def test_initialize_success(
        self, mock_receiver_class, mock_sender_class, mock_fl_context, mock_engine_with_clients
    ):
        """Test successful initialization of TensorServerStreamer."""
        # Setup mocks
        mock_fl_context.get_engine.return_value = mock_engine_with_clients
        mock_receiver_instance = Mock(spec=TensorReceiver)
        mock_sender_instance = Mock(spec=TensorSender)
        mock_receiver_class.return_value = mock_receiver_instance
        mock_sender_class.return_value = mock_sender_instance

        # Create and initialize streamer
        streamer = TensorServerStreamer(format=ExchangeFormat.PYTORCH)
        streamer.initialize(mock_fl_context)

        # Verify engine assignment
        assert streamer.engine == mock_engine_with_clients

        # Verify receiver creation
        mock_receiver_class.assert_called_once_with(
            mock_engine_with_clients, FLContextKey.TASK_RESULT, ExchangeFormat.PYTORCH
        )
        assert streamer.receiver == mock_receiver_instance

        # Verify sender is now created during initialization
        mock_sender_class.assert_called_once_with(
            mock_engine_with_clients, FLContextKey.TASK_DATA, ExchangeFormat.PYTORCH, ["train"]
        )
        assert streamer.sender == mock_sender_instance

    @pytest.mark.parametrize(
        "engine_value,receiver_exception,expected_error_message",
        [
            (None, None, "Engine not found"),
            (Mock(), None, "Engine is not a StreamableEngine"),  # Wrong engine type
            ("valid", Exception("Receiver creation failed"), "Receiver creation failed"),
        ],
    )
    @patch("nvflare.app_opt.tensor_stream.server.TensorSender")
    @patch("nvflare.app_opt.tensor_stream.server.TensorReceiver")
    def test_initialize_errors(
        self,
        mock_receiver_class,
        mock_sender_class,
        engine_value,
        receiver_exception,
        expected_error_message,
        mock_fl_context,
        mock_engine_with_clients,
    ):
        """Test initialization error scenarios."""
        # Setup mock based on test parameters
        if engine_value == "valid":
            mock_fl_context.get_engine.return_value = mock_engine_with_clients
            mock_receiver_class.side_effect = receiver_exception
        else:
            mock_fl_context.get_engine.return_value = engine_value

        streamer = TensorServerStreamer()
        streamer.system_panic = Mock()

        streamer.initialize(mock_fl_context)

        # Verify system_panic was called with appropriate message
        streamer.system_panic.assert_called_once()
        args = streamer.system_panic.call_args[0]
        assert expected_error_message in args[0]
        assert args[1] == mock_fl_context

    @patch("nvflare.app_opt.tensor_stream.server.TensorSender")
    @patch("nvflare.app_opt.tensor_stream.server.TensorReceiver")
    def test_handle_event_start_run(
        self, mock_receiver_class, mock_sender_class, mock_fl_context, mock_engine_with_clients
    ):
        """Test handling START_RUN event."""
        mock_fl_context.get_engine.return_value = mock_engine_with_clients

        streamer = TensorServerStreamer()

        # Handle START_RUN event
        streamer.handle_event(EventType.START_RUN, mock_fl_context)

        # Verify initialization was called (engine, receiver, and sender should be created)
        assert streamer.engine == mock_engine_with_clients
        assert streamer.receiver is not None
        # Sender should now be created during initialization
        assert streamer.sender is not None

    def test_handle_event_before_task_data_filter(self, mock_fl_context):
        """Test handling BEFORE_TASK_DATA_FILTER event - tracks task IDs."""
        streamer = TensorServerStreamer()

        # Mock the context to return current_round and task_id
        current_round = 1
        task_id = "task_123"
        mock_fl_context.get_prop.side_effect = lambda key: {
            AppConstants.CURRENT_ROUND: current_round,
            FLContextKey.TASK_ID: task_id,
        }.get(key)

        # Handle BEFORE_TASK_DATA_FILTER event
        streamer.handle_event(EventType.BEFORE_TASK_DATA_FILTER, mock_fl_context)

        # Verify task_id was added to seen_tasks for the current round
        assert task_id in streamer.seen_tasks[current_round]

    def test_handle_event_after_task_data_filter(self, mock_fl_context, mock_engine_with_clients):
        """Test handling AFTER_TASK_DATA_FILTER event."""
        streamer = TensorServerStreamer()
        streamer.engine = mock_engine_with_clients

        # Mock sender (now created during initialization)
        mock_sender = Mock(spec=TensorSender)
        mock_sender.store_tensors = Mock()
        streamer.sender = mock_sender

        streamer.send_tensors_to_client = Mock()
        streamer.wait_sending_task_data_all_clients = Mock()
        streamer.try_to_clean_task_data = Mock()

        # Handle AFTER_TASK_DATA_FILTER event
        streamer.handle_event(EventType.AFTER_TASK_DATA_FILTER, mock_fl_context)

        # Verify store_tensors was called
        mock_sender.store_tensors.assert_called_once_with(mock_fl_context)

        # Verify send_tensors_to_client was called
        streamer.send_tensors_to_client.assert_called_once_with(mock_fl_context)
        # Verify wait_sending_task_data_all_clients was called
        streamer.wait_sending_task_data_all_clients.assert_called_once_with(3, mock_fl_context)
        # Verify try_to_clean_task_data was called
        streamer.try_to_clean_task_data.assert_called_once_with(3, mock_fl_context)

    def test_handle_event_after_task_data_filter_exception(self, mock_fl_context, mock_engine_with_clients):
        """Test handling AFTER_TASK_DATA_FILTER event when store_tensors raises exception."""
        streamer = TensorServerStreamer()
        streamer.engine = mock_engine_with_clients

        # Mock sender to raise exception during store_tensors
        mock_sender = Mock(spec=TensorSender)
        mock_sender.store_tensors = Mock(side_effect=Exception("Store failed"))
        streamer.sender = mock_sender

        streamer.send_tensors_to_client = Mock()
        streamer.wait_sending_task_data_all_clients = Mock()
        streamer.try_to_clean_task_data = Mock()

        # Handle AFTER_TASK_DATA_FILTER event - should propagate the exception
        with pytest.raises(Exception, match="Store failed"):
            streamer.handle_event(EventType.AFTER_TASK_DATA_FILTER, mock_fl_context)

        # Verify store_tensors was called and raised exception
        mock_sender.store_tensors.assert_called_once_with(mock_fl_context)

        # Other methods should not be called due to the exception
        streamer.send_tensors_to_client.assert_not_called()
        streamer.wait_sending_task_data_all_clients.assert_not_called()
        streamer.try_to_clean_task_data.assert_not_called()

    def test_handle_event_before_task_result_filter(self, mock_fl_context):
        """Test handling BEFORE_TASK_RESULT_FILTER event."""
        streamer = TensorServerStreamer()
        mock_receiver = Mock(spec=TensorReceiver)
        streamer.receiver = mock_receiver

        # Handle BEFORE_TASK_RESULT_FILTER event
        streamer.handle_event(EventType.BEFORE_TASK_RESULT_FILTER, mock_fl_context)

        # Verify receiver.set_ctx_with_tensors was called
        mock_receiver.set_ctx_with_tensors.assert_called_once_with(mock_fl_context)

    def test_sender_creation_on_after_task_data_filter(self, mock_fl_context, mock_engine_with_clients):
        """Test that TensorSender is already created during initialization, not during AFTER_TASK_DATA_FILTER event."""
        streamer = TensorServerStreamer(format=ExchangeFormat.NUMPY, tasks=["custom_task"])
        streamer.engine = mock_engine_with_clients

        # Mock sender (created during initialization)
        mock_sender = Mock(spec=TensorSender)
        mock_sender.store_tensors = Mock()
        streamer.sender = mock_sender

        # Mock other methods to avoid side effects
        streamer.send_tensors_to_client = Mock()
        streamer.wait_sending_task_data_all_clients = Mock()
        streamer.try_to_clean_task_data = Mock()

        # Handle AFTER_TASK_DATA_FILTER event
        streamer.handle_event(EventType.AFTER_TASK_DATA_FILTER, mock_fl_context)

        # Verify store_tensors was called
        mock_sender.store_tensors.assert_called_once_with(mock_fl_context)

    @pytest.mark.parametrize(
        "send_result,expected_sent,expected_skipped",
        [
            (True, 1, 0),  # Successful send
            (False, 0, 1),  # Send failed (ValueError caught)
        ],
    )
    def test_send_tensors_to_client(
        self, send_result, expected_sent, expected_skipped, mock_fl_context, mock_engine_with_clients
    ):
        """Test tensor sending to client with different outcomes."""
        current_round = 1
        mock_fl_context.get_prop.return_value = current_round

        streamer = TensorServerStreamer(entry_timeout=5.0)
        streamer.engine = mock_engine_with_clients

        # Mock sender
        mock_sender = Mock(spec=TensorSender)
        if not send_result:
            # Simulate ValueError being raised during send
            mock_sender.send.side_effect = ValueError("No tensor data found")
        else:
            mock_sender.send.return_value = None  # send() doesn't return a value
        streamer.sender = mock_sender

        # Mock log_warning to verify error logging
        streamer.log_warning = Mock()

        # Initial state - counters should be empty dicts
        assert streamer.num_task_data_sent[current_round] == 0
        assert streamer.num_task_skipped[current_round] == 0

        # Send tensors (store_tensors is called before this in the event handler)
        streamer.send_tensors_to_client(mock_fl_context)

        # Verify send was called
        mock_sender.send.assert_called_once_with(mock_fl_context, 5.0)

        if not send_result:
            # Verify error was logged
            streamer.log_warning.assert_called_once()

        # Verify counters were updated correctly for the current round
        assert streamer.num_task_data_sent[current_round] == expected_sent
        assert streamer.num_task_skipped[current_round] == expected_skipped

    def test_send_tensors_to_client_value_error(self, mock_fl_context):
        """Test tensor sending when sender raises ValueError."""
        current_round = 1
        mock_fl_context.get_prop.return_value = current_round

        streamer = TensorServerStreamer()

        # Mock sender to raise ValueError during send
        mock_sender = Mock(spec=TensorSender)
        mock_sender.send.side_effect = ValueError("No tensor data found")
        streamer.sender = mock_sender

        # Mock log_warning
        streamer.log_warning = Mock()

        # Send tensors (store_tensors is called before this in the event handler)
        streamer.send_tensors_to_client(mock_fl_context)

        # Verify log_warning was called
        streamer.log_warning.assert_called_once()
        args = streamer.log_warning.call_args[0]
        assert "No tensors to send to client" in args[1]
        assert "No tensor data found" in args[1]

        # Verify counter was incremented for skipped
        assert streamer.num_task_skipped[current_round] == 1
        assert streamer.num_task_data_sent[current_round] == 0

    @pytest.mark.parametrize(
        "start_time,num_sent,num_skipped,num_clients,timeout,time_progression,should_timeout",
        [
            (1000.0, 2, 1, 3, 300.0, None, False),  # Success: all clients processed
            (1000.0, 1, 0, 3, 1.0, [1000.5, 1001.5], True),  # Timeout scenario
        ],
    )
    @patch("time.sleep")
    @patch("time.time")
    def test_wait_sending_task_data_all_clients(
        self,
        mock_time,
        mock_sleep,
        start_time,
        num_sent,
        num_skipped,
        num_clients,
        timeout,
        time_progression,
        should_timeout,
        mock_fl_context,
    ):
        """Test waiting for clients to complete in various scenarios."""
        current_round = 1
        mock_fl_context.get_prop.return_value = current_round

        streamer = TensorServerStreamer(wait_send_task_data_all_clients_timeout=timeout)
        streamer.start_sending_time[current_round] = start_time
        streamer.num_task_data_sent[current_round] = num_sent
        streamer.num_task_skipped[current_round] = num_skipped
        streamer.system_panic = Mock()

        if time_progression:
            mock_time.side_effect = time_progression

        # Execute the method
        streamer.wait_sending_task_data_all_clients(num_clients, mock_fl_context)

        if should_timeout:
            # Should have called sleep and system_panic due to timeout
            mock_sleep.assert_called_with(0.1)
            streamer.system_panic.assert_called_once()
            args = streamer.system_panic.call_args[0]
            assert "Timeout waiting for all clients" in args[0]
            assert f"Sent to {num_sent} out of {num_clients}" in args[0]
            assert f"skipped {num_skipped}" in args[0]
        else:
            # Should sleep once and then exit since all clients are processed
            mock_sleep.assert_called_once_with(0.1)
            streamer.system_panic.assert_not_called()

    @pytest.mark.parametrize(
        "num_sent,num_clients,should_clean,expected_log_message",
        [
            (3, 3, True, "Tensors were sent to all clients"),  # All clients received
            (1, 3, False, None),  # Not all clients received
            (5, 3, True, "Tensors were sent to all clients"),  # More than expected (edge case)
        ],
    )
    @patch("nvflare.app_opt.tensor_stream.server.clean_task_data")
    def test_try_to_clean_task_data(
        self,
        mock_clean_task_data,
        num_sent,
        num_clients,
        should_clean,
        expected_log_message,
        mock_fl_context,
        mock_engine_with_clients,
    ):
        """Test cleaning task data based on different client reception scenarios."""
        current_round = 1
        mock_fl_context.get_prop.return_value = current_round

        streamer = TensorServerStreamer()
        streamer.engine = mock_engine_with_clients
        streamer.num_task_data_sent[current_round] = num_sent
        streamer.data_cleaned[current_round] = False
        streamer.log_info = Mock()

        # Clean task data
        streamer.try_to_clean_task_data(num_clients, mock_fl_context)

        if should_clean:
            # Verify clean_task_data was called
            mock_clean_task_data.assert_called_once_with(mock_fl_context)
            # Verify state was updated for the current round
            assert streamer.data_cleaned[current_round] is True
            # Verify log message
            streamer.log_info.assert_called_once()
            log_args = streamer.log_info.call_args[0]
            assert expected_log_message in log_args[1]
            assert f"Sent {num_sent} out of {num_clients}" in log_args[1]
        else:
            # Verify clean_task_data was NOT called
            mock_clean_task_data.assert_not_called()
            # Verify state was not updated for the current round
            assert streamer.data_cleaned[current_round] is False
            # Verify no log message
            streamer.log_info.assert_not_called()

    @patch("nvflare.app_opt.tensor_stream.server.TensorSender")
    @patch("nvflare.app_opt.tensor_stream.server.TensorReceiver")
    def test_complete_workflow(self, mock_receiver_class, mock_sender_class, mock_fl_context, mock_engine_with_clients):
        """Test complete workflow: initialization, event handling, and tensor operations."""
        current_round = 1
        task_id = "task_123"

        # Setup mocks
        mock_fl_context.get_engine.return_value = mock_engine_with_clients
        mock_fl_context.get_prop.side_effect = lambda key: {
            "AppConstants.CURRENT_ROUND": current_round,
            FLContextKey.TASK_ID: task_id,
        }.get(
            key, current_round
        )  # Default to current_round for other keys

        mock_sender_instance = Mock(spec=TensorSender)
        mock_receiver_instance = Mock(spec=TensorReceiver)
        mock_receiver_instance.tensors = {}
        mock_sender_class.return_value = mock_sender_instance
        mock_receiver_class.return_value = mock_receiver_instance

        # Create streamer
        streamer = TensorServerStreamer(format=ExchangeFormat.PYTORCH, entry_timeout=10.0)

        # Step 1: Handle START_RUN event (initialization)
        streamer.handle_event(EventType.START_RUN, mock_fl_context)

        # Verify initialization (both receiver and sender created during initialization)
        assert streamer.engine == mock_engine_with_clients
        assert streamer.sender == mock_sender_instance  # Sender now created during initialization
        assert streamer.receiver == mock_receiver_instance

        # Step 2: Handle BEFORE_TASK_DATA_FILTER event (track task)
        streamer.handle_event(EventType.BEFORE_TASK_DATA_FILTER, mock_fl_context)
        assert streamer.data_cleaned[current_round] is False

        # Step 3: Handle AFTER_TASK_DATA_FILTER event (store and send tensors)
        with patch.object(streamer, "wait_sending_task_data_all_clients") as mock_wait:
            with patch.object(streamer, "try_to_clean_task_data") as mock_clean:
                mock_sender_instance.store_tensors.return_value = None
                mock_sender_instance.send.return_value = None
                streamer.handle_event(EventType.AFTER_TASK_DATA_FILTER, mock_fl_context)

                # Verify sender methods were called
                mock_sender_instance.store_tensors.assert_called_once_with(mock_fl_context)
                mock_sender_instance.send.assert_called_once_with(mock_fl_context, 10.0)

                # Verify counter incremented for the current round
                assert streamer.num_task_data_sent[current_round] == 1

                # Verify wait was called
                mock_wait.assert_called_once_with(3, mock_fl_context)

                # Verify cleanup was attempted
                mock_clean.assert_called_once_with(3, mock_fl_context)

        # Step 4: Handle BEFORE_TASK_RESULT_FILTER event
        mock_receiver_instance.wait_for_tensors.return_value = None
        streamer.handle_event(EventType.BEFORE_TASK_RESULT_FILTER, mock_fl_context)

        # Verify receiver methods were called
        mock_receiver_instance.wait_for_tensors.assert_called_once()
        mock_receiver_instance.set_ctx_with_tensors.assert_called_once_with(mock_fl_context)

    @pytest.mark.parametrize(
        "format_param,tasks_param,expected_format,expected_tasks",
        [
            (ExchangeFormat.NUMPY, None, ExchangeFormat.NUMPY, ["train"]),
            (
                ExchangeFormat.PYTORCH,
                ["train", "validate", "test"],
                ExchangeFormat.PYTORCH,
                ["train", "validate", "test"],
            ),
        ],
    )
    @patch("nvflare.app_opt.tensor_stream.server.TensorSender")
    @patch("nvflare.app_opt.tensor_stream.server.TensorReceiver")
    def test_parameter_propagation(
        self,
        mock_receiver_class,
        mock_sender_class,
        format_param,
        tasks_param,
        expected_format,
        expected_tasks,
        mock_fl_context,
        mock_engine_with_clients,
    ):
        """Test that format and tasks parameters are properly propagated to receiver and sender."""
        mock_fl_context.get_engine.return_value = mock_engine_with_clients

        streamer = TensorServerStreamer(format=format_param, tasks=tasks_param)
        streamer.initialize(mock_fl_context)

        # Verify receiver was created with correct format during initialization
        mock_receiver_class.assert_called_once_with(
            mock_engine_with_clients,
            FLContextKey.TASK_RESULT,
            expected_format,
        )

        # Verify sender was also created with correct format and tasks during initialization
        mock_sender_class.assert_called_once_with(
            mock_engine_with_clients,
            FLContextKey.TASK_DATA,
            expected_format,
            expected_tasks,
        )
