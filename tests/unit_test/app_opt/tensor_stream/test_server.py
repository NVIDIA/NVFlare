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
from nvflare.apis.streaming import StreamableEngine
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
            wait_all_clients_timeout=wait_all_clients_timeout,
        )

        assert streamer.format == format_type
        assert streamer.tasks == expected_tasks
        assert streamer.entry_timeout == entry_timeout
        assert streamer.wait_all_clients_timeout == wait_all_clients_timeout
        # All components should be None before initialization
        assert streamer.engine is None
        assert streamer.sender is None
        assert streamer.receiver is None
        assert streamer.num_task_data_sent == 0
        assert streamer.num_task_skipped == 0
        assert streamer.data_cleaned is False

    @patch("nvflare.app_opt.tensor_stream.server.TensorSender")
    @patch("nvflare.app_opt.tensor_stream.server.TensorReceiver")
    def test_initialize_success(
        self, mock_receiver_class, mock_sender_class, mock_fl_context, mock_engine_with_clients
    ):
        """Test successful initialization of TensorServerStreamer."""
        # Setup mocks
        mock_fl_context.get_engine.return_value = mock_engine_with_clients
        mock_sender_instance = Mock(spec=TensorSender)
        mock_receiver_instance = Mock(spec=TensorReceiver)
        mock_sender_class.return_value = mock_sender_instance
        mock_receiver_class.return_value = mock_receiver_instance

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

        # Verify sender creation
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
    @patch("nvflare.app_opt.tensor_stream.server.TensorReceiver")
    def test_initialize_errors(
        self,
        mock_receiver_class,
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

        # Verify initialization was called (components should be created)
        assert streamer.engine == mock_engine_with_clients
        assert streamer.sender is not None
        assert streamer.receiver is not None

    @pytest.mark.parametrize(
        "data_cleaned_initial,expected_sent_after,expected_skipped_after,expected_data_cleaned_after",
        [
            (True, 0, 0, False),  # Data was cleaned, counters should reset
            (False, 5, 2, False),  # Data was not cleaned, counters should not reset
        ],
    )
    def test_handle_event_before_task_data_filter(
        self,
        data_cleaned_initial,
        expected_sent_after,
        expected_skipped_after,
        expected_data_cleaned_after,
        mock_fl_context,
    ):
        """Test handling BEFORE_TASK_DATA_FILTER event with different data cleaned states."""
        streamer = TensorServerStreamer()
        streamer.data_cleaned = data_cleaned_initial
        streamer.num_task_data_sent = 5
        streamer.num_task_skipped = 2

        # Handle BEFORE_TASK_DATA_FILTER event
        streamer.handle_event(EventType.BEFORE_TASK_DATA_FILTER, mock_fl_context)

        # Verify counters behavior based on data_cleaned state
        assert streamer.num_task_data_sent == expected_sent_after
        assert streamer.num_task_skipped == expected_skipped_after
        assert streamer.data_cleaned == expected_data_cleaned_after

    def test_handle_event_after_task_data_filter(self, mock_fl_context, mock_engine_with_clients):
        """Test handling AFTER_TASK_DATA_FILTER event."""
        streamer = TensorServerStreamer()
        streamer.engine = mock_engine_with_clients
        streamer.send_tensors_to_client = Mock()
        streamer.wait_clients_to_complete = Mock()
        streamer.try_to_clean_task_data = Mock()

        # Handle AFTER_TASK_DATA_FILTER event
        streamer.handle_event(EventType.AFTER_TASK_DATA_FILTER, mock_fl_context)

        # Verify send_tensors_to_client was called
        streamer.send_tensors_to_client.assert_called_once_with(mock_fl_context)
        # Verify wait_clients_to_complete was called
        streamer.wait_clients_to_complete.assert_called_once_with(3, mock_fl_context)
        # Verify try_to_clean_task_data was called
        streamer.try_to_clean_task_data.assert_called_once_with(3, mock_fl_context)

    def test_handle_event_after_task_data_filter_exception(self, mock_fl_context, mock_engine_with_clients):
        """Test handling AFTER_TASK_DATA_FILTER event when send_tensors_to_client raises exception."""
        streamer = TensorServerStreamer()
        streamer.engine = mock_engine_with_clients
        streamer.send_tensors_to_client = Mock(side_effect=Exception("Send failed"))
        streamer.wait_clients_to_complete = Mock()
        streamer.try_to_clean_task_data = Mock()

        # Handle AFTER_TASK_DATA_FILTER event - should propagate the exception
        with pytest.raises(Exception, match="Send failed"):
            streamer.handle_event(EventType.AFTER_TASK_DATA_FILTER, mock_fl_context)

        # Verify send_tensors_to_client was called and raised exception
        streamer.send_tensors_to_client.assert_called_once_with(mock_fl_context)

        # Other methods should not be called due to the exception
        streamer.wait_clients_to_complete.assert_not_called()
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

    @pytest.mark.parametrize(
        "send_result,expected_sent,expected_skipped",
        [
            (True, 1, 0),  # Successful send
            (False, 0, 1),  # Skipped send
        ],
    )
    def test_send_tensors_to_client(
        self, send_result, expected_sent, expected_skipped, mock_fl_context, mock_engine_with_clients
    ):
        """Test tensor sending to client with different outcomes."""
        streamer = TensorServerStreamer(entry_timeout=5.0)
        streamer.engine = mock_engine_with_clients

        # Mock sender
        mock_sender = Mock(spec=TensorSender)
        mock_sender.send.return_value = send_result
        streamer.sender = mock_sender

        # Initial state
        assert streamer.num_task_data_sent == 0
        assert streamer.num_task_skipped == 0

        # Send tensors
        streamer.send_tensors_to_client(mock_fl_context)

        # Verify sender.send was called with correct parameters
        mock_sender.send.assert_called_once_with(mock_fl_context, 5.0)

        # Verify counters were updated correctly
        assert streamer.num_task_data_sent == expected_sent
        assert streamer.num_task_skipped == expected_skipped

    def test_send_tensors_to_client_value_error(self, mock_fl_context):
        """Test tensor sending when sender raises ValueError."""
        streamer = TensorServerStreamer()

        # Mock sender to raise ValueError
        mock_sender = Mock(spec=TensorSender)
        mock_sender.send.side_effect = ValueError("No tensor data found")
        streamer.sender = mock_sender

        # Mock system_panic
        streamer.system_panic = Mock()

        # Send tensors
        streamer.send_tensors_to_client(mock_fl_context)

        # Verify system_panic was called
        streamer.system_panic.assert_called_once()
        args = streamer.system_panic.call_args
        assert args[0][1] == mock_fl_context
        assert "Failed to send tensors" in args[0][0]
        assert "No tensor data found" in args[0][0]

    @pytest.mark.parametrize(
        "start_time,num_sent,num_skipped,num_clients,timeout,time_progression,should_timeout",
        [
            (1000.0, 2, 1, 3, 300.0, None, False),  # Success: all clients processed
            (1000.0, 1, 0, 3, 1.0, [1000.5, 1001.5], True),  # Timeout scenario
            (None, 0, 0, 3, 300.0, None, False),  # No start time, should return immediately
        ],
    )
    @patch("time.sleep")
    @patch("time.time")
    def test_wait_clients_to_complete(
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
        streamer = TensorServerStreamer(wait_all_clients_timeout=timeout)
        streamer.start_sending_time = start_time
        streamer.num_task_data_sent = num_sent
        streamer.num_task_skipped = num_skipped
        streamer.system_panic = Mock()

        if time_progression:
            mock_time.side_effect = time_progression

        # Execute the method
        streamer.wait_clients_to_complete(num_clients, mock_fl_context)

        if start_time is None:
            # Should return immediately without sleeping
            mock_sleep.assert_not_called()
            streamer.system_panic.assert_not_called()
        elif should_timeout:
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
        streamer = TensorServerStreamer()
        streamer.engine = mock_engine_with_clients
        streamer.num_task_data_sent = num_sent
        streamer.data_cleaned = False
        streamer.log_info = Mock()

        # Clean task data
        streamer.try_to_clean_task_data(num_clients, mock_fl_context)

        if should_clean:
            # Verify clean_task_data was called
            mock_clean_task_data.assert_called_once_with(mock_fl_context)
            # Verify state was updated
            assert streamer.data_cleaned is True
            # Verify log message
            streamer.log_info.assert_called_once()
            log_args = streamer.log_info.call_args[0]
            assert expected_log_message in log_args[1]
            assert f"Sent {num_sent} out of {num_clients}" in log_args[1]
        else:
            # Verify clean_task_data was NOT called
            mock_clean_task_data.assert_not_called()
            # Verify state was not updated
            assert streamer.data_cleaned is False
            # Verify no log message
            streamer.log_info.assert_not_called()

    @patch("nvflare.app_opt.tensor_stream.server.TensorSender")
    @patch("nvflare.app_opt.tensor_stream.server.TensorReceiver")
    def test_complete_workflow(self, mock_receiver_class, mock_sender_class, mock_fl_context, mock_engine_with_clients):
        """Test complete workflow: initialization, event handling, and tensor operations."""
        # Setup mocks
        mock_fl_context.get_engine.return_value = mock_engine_with_clients
        mock_sender_instance = Mock(spec=TensorSender)
        mock_receiver_instance = Mock(spec=TensorReceiver)
        mock_sender_class.return_value = mock_sender_instance
        mock_receiver_class.return_value = mock_receiver_instance

        # Create streamer
        streamer = TensorServerStreamer(format=ExchangeFormat.PYTORCH, entry_timeout=10.0)

        # Step 1: Handle START_RUN event (initialization)
        streamer.handle_event(EventType.START_RUN, mock_fl_context)

        # Verify initialization
        assert streamer.engine == mock_engine_with_clients
        assert streamer.sender == mock_sender_instance
        assert streamer.receiver == mock_receiver_instance

        # Step 2: Handle BEFORE_TASK_DATA_FILTER event
        streamer.handle_event(EventType.BEFORE_TASK_DATA_FILTER, mock_fl_context)
        assert streamer.data_cleaned is False

        # Step 3: Handle AFTER_TASK_DATA_FILTER event (send tensors)
        with patch.object(streamer, "wait_clients_to_complete") as mock_wait:
            with patch.object(streamer, "try_to_clean_task_data") as mock_clean:
                mock_sender_instance.send.return_value = True  # Mock successful send
                streamer.handle_event(EventType.AFTER_TASK_DATA_FILTER, mock_fl_context)

                # Verify sender was called
                mock_sender_instance.send.assert_called_once_with(mock_fl_context, 10.0)

                # Verify counter incremented
                assert streamer.num_task_data_sent == 1

                # Verify wait was called
                mock_wait.assert_called_once_with(3, mock_fl_context)

                # Verify cleanup was attempted
                mock_clean.assert_called_once_with(3, mock_fl_context)

        # Step 4: Handle BEFORE_TASK_RESULT_FILTER event
        streamer.handle_event(EventType.BEFORE_TASK_RESULT_FILTER, mock_fl_context)

        # Verify receiver was called
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

        # Verify receiver was created with correct format
        mock_receiver_class.assert_called_once_with(
            mock_engine_with_clients,
            FLContextKey.TASK_RESULT,
            expected_format,
        )

        # Verify sender was created with correct format and tasks
        mock_sender_class.assert_called_once_with(
            mock_engine_with_clients,
            FLContextKey.TASK_DATA,
            expected_format,
            expected_tasks,
        )
