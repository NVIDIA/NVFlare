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
        "format_type,root_keys,entry_timeout,expected_root_keys",
        [
            (ExchangeFormat.PYTORCH, None, 30.0, [""]),  # Default values
            (ExchangeFormat.NUMPY, ["encoder", "decoder"], 10.0, ["encoder", "decoder"]),  # Custom values
            (ExchangeFormat.PYTORCH, ["model"], 45.0, ["model"]),  # Partial custom values
        ],
    )
    def test_init_parameters(self, format_type, root_keys, entry_timeout, expected_root_keys):
        """Test TensorServerStreamer initialization with various parameters."""
        streamer = TensorServerStreamer(format=format_type, root_keys=root_keys, entry_timeout=entry_timeout)

        assert streamer.format == format_type
        assert streamer.root_keys == expected_root_keys
        assert streamer.entry_timeout == entry_timeout
        # All components should be None before initialization
        assert streamer.engine is None
        assert streamer.sender is None
        assert streamer.receiver is None
        assert streamer.num_task_data_sent == 0
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
        streamer = TensorServerStreamer(format=ExchangeFormat.PYTORCH, root_keys=["model"])
        streamer.initialize(mock_fl_context)

        # Verify engine assignment
        assert streamer.engine == mock_engine_with_clients

        # Verify receiver creation
        mock_receiver_class.assert_called_once_with(
            mock_engine_with_clients, FLContextKey.TASK_RESULT, ExchangeFormat.PYTORCH
        )
        assert streamer.receiver == mock_receiver_instance

        # Verify sender creation
        mock_sender_class.assert_called_once_with(mock_engine_with_clients, FLContextKey.TASK_DATA, ["model"])
        assert streamer.sender == mock_sender_instance

    def test_initialize_no_engine(self, mock_fl_context):
        """Test initialization when no engine is found."""
        mock_fl_context.get_engine.return_value = None

        streamer = TensorServerStreamer()

        # Mock system_panic to verify it's called
        streamer.system_panic = Mock()

        streamer.initialize(mock_fl_context)

        # Verify system_panic was called with appropriate message
        streamer.system_panic.assert_called_once()
        args = streamer.system_panic.call_args[0]
        assert "Engine not found" in args[0]
        assert args[1] == mock_fl_context

    def test_initialize_wrong_engine_type(self, mock_fl_context):
        """Test initialization when engine is not a StreamableEngine."""
        # Return a generic Mock instead of StreamableEngine
        wrong_engine = Mock()
        mock_fl_context.get_engine.return_value = wrong_engine

        streamer = TensorServerStreamer()
        streamer.system_panic = Mock()

        streamer.initialize(mock_fl_context)

        # Verify system_panic was called
        streamer.system_panic.assert_called_once()
        args = streamer.system_panic.call_args[0]
        assert "Engine is not a StreamableEngine" in args[0]
        assert args[1] == mock_fl_context

    @patch("nvflare.app_opt.tensor_stream.server.TensorReceiver")
    def test_initialize_receiver_exception(self, mock_receiver_class, mock_fl_context, mock_engine_with_clients):
        """Test initialization when TensorReceiver creation fails."""
        mock_fl_context.get_engine.return_value = mock_engine_with_clients
        mock_receiver_class.side_effect = Exception("Receiver creation failed")

        streamer = TensorServerStreamer()
        streamer.system_panic = Mock()

        streamer.initialize(mock_fl_context)

        # Verify system_panic was called with exception message
        streamer.system_panic.assert_called_once_with("Receiver creation failed", mock_fl_context)

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

    def test_handle_event_before_task_data_filter(self, mock_fl_context):
        """Test handling BEFORE_TASK_DATA_FILTER event."""
        streamer = TensorServerStreamer()
        streamer.data_cleaned = True  # Set to True initially

        # Handle BEFORE_TASK_DATA_FILTER event
        streamer.handle_event(EventType.BEFORE_TASK_DATA_FILTER, mock_fl_context)

        # Verify data_cleaned flag is reset
        assert streamer.data_cleaned is False

    def test_handle_event_after_task_data_filter(self, mock_fl_context, mock_engine_with_clients):
        """Test handling AFTER_TASK_DATA_FILTER event."""
        streamer = TensorServerStreamer()
        streamer.engine = mock_engine_with_clients
        streamer.send_tensors_to_client = Mock()
        streamer.try_to_clean_task_data = Mock()

        # Handle AFTER_TASK_DATA_FILTER event
        streamer.handle_event(EventType.AFTER_TASK_DATA_FILTER, mock_fl_context)

        # Verify send_tensors_to_client was called
        streamer.send_tensors_to_client.assert_called_once_with(mock_fl_context)
        # Verify try_to_clean_task_data was called
        streamer.try_to_clean_task_data.assert_called_once_with(mock_fl_context)

    def test_handle_event_after_task_data_filter_exception(self, mock_fl_context):
        """Test handling AFTER_TASK_DATA_FILTER event when send_tensors_to_client raises exception."""
        streamer = TensorServerStreamer()
        streamer.send_tensors_to_client = Mock(side_effect=Exception("Send failed"))
        streamer.system_panic = Mock()

        # Handle AFTER_TASK_DATA_FILTER event
        streamer.handle_event(EventType.AFTER_TASK_DATA_FILTER, mock_fl_context)

        # Verify system_panic was called
        streamer.system_panic.assert_called_once()
        args = streamer.system_panic.call_args
        assert args[0][1] == mock_fl_context
        assert "Failed to send tensors" in args[0][0]

    def test_handle_event_before_task_result_filter(self, mock_fl_context):
        """Test handling BEFORE_TASK_RESULT_FILTER event."""
        streamer = TensorServerStreamer()
        mock_receiver = Mock(spec=TensorReceiver)
        streamer.receiver = mock_receiver

        # Handle BEFORE_TASK_RESULT_FILTER event
        streamer.handle_event(EventType.BEFORE_TASK_RESULT_FILTER, mock_fl_context)

        # Verify receiver.set_ctx_with_tensors was called
        mock_receiver.set_ctx_with_tensors.assert_called_once_with(mock_fl_context)

    def test_send_tensors_to_client_success(self, mock_fl_context, mock_engine_with_clients):
        """Test successful tensor sending to client."""
        streamer = TensorServerStreamer(entry_timeout=5.0)
        streamer.engine = mock_engine_with_clients

        # Mock sender
        mock_sender = Mock(spec=TensorSender)
        streamer.sender = mock_sender

        # Initial state
        assert streamer.num_task_data_sent == 0

        # Send tensors
        streamer.send_tensors_to_client(mock_fl_context)

        # Verify sender.send was called with correct parameters
        mock_sender.send.assert_called_once_with(mock_fl_context, 5.0)

        # Verify counter was incremented
        assert streamer.num_task_data_sent == 1

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

    @patch("nvflare.app_opt.tensor_stream.server.clean_task_data")
    def test_try_to_clean_task_data_all_clients_received(
        self, mock_clean_task_data, mock_fl_context, mock_engine_with_clients
    ):
        """Test cleaning task data when all clients have received tensors."""
        streamer = TensorServerStreamer()
        streamer.engine = mock_engine_with_clients
        streamer.num_task_data_sent = 3  # Same as number of clients
        streamer.log_info = Mock()

        # Clean task data
        streamer.try_to_clean_task_data(mock_fl_context)

        # Verify clean_task_data was called
        mock_clean_task_data.assert_called_once_with(mock_fl_context)

        # Verify state was updated
        assert streamer.data_cleaned is True
        assert streamer.num_task_data_sent == 0

        # Verify log message
        streamer.log_info.assert_called_once()
        log_args = streamer.log_info.call_args[0]
        assert "All tensors sent now" in log_args[1]
        assert "Sent 3 out of 3" in log_args[1]

    @patch("time.sleep")
    def test_try_to_clean_task_data_not_all_clients_received(
        self, mock_sleep, mock_fl_context, mock_engine_with_clients
    ):
        """Test trying to clean task data when not all clients have received tensors."""
        streamer = TensorServerStreamer()
        streamer.engine = mock_engine_with_clients
        streamer.num_task_data_sent = 1  # Less than number of clients (3)
        streamer.data_cleaned = False
        streamer.log_debug = Mock()

        # Mock the condition to break the while loop after one iteration
        def side_effect(*args):
            streamer.data_cleaned = True  # Set to True to break the loop

        mock_sleep.side_effect = side_effect

        # Try to clean task data
        streamer.try_to_clean_task_data(mock_fl_context)

        # Verify debug message was logged
        streamer.log_debug.assert_called_once()
        debug_args = streamer.log_debug.call_args[0]
        assert "Not all sites received the tensors yet" in debug_args[1]
        assert "Sent 1 out of 3" in debug_args[1]

        # Verify sleep was called (waiting for all clients)
        mock_sleep.assert_called_once_with(0.1)

        # Verify counters were not reset (since not all clients received)
        assert streamer.num_task_data_sent == 1

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
        streamer = TensorServerStreamer(format=ExchangeFormat.PYTORCH, root_keys=["model"], entry_timeout=10.0)

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
        with patch.object(streamer, "try_to_clean_task_data") as mock_clean:
            streamer.handle_event(EventType.AFTER_TASK_DATA_FILTER, mock_fl_context)

            # Verify sender was called
            mock_sender_instance.send.assert_called_once_with(mock_fl_context, 10.0)

            # Verify counter incremented
            assert streamer.num_task_data_sent == 1

            # Verify cleanup was attempted
            mock_clean.assert_called_once_with(mock_fl_context)

        # Step 4: Handle BEFORE_TASK_RESULT_FILTER event
        streamer.handle_event(EventType.BEFORE_TASK_RESULT_FILTER, mock_fl_context)

        # Verify receiver was called
        mock_receiver_instance.set_ctx_with_tensors.assert_called_once_with(mock_fl_context)

    def test_multiple_clients_tensor_sending(self, mock_fl_context, mock_engine_with_clients):
        """Test tensor sending workflow with multiple clients."""
        streamer = TensorServerStreamer()
        streamer.engine = mock_engine_with_clients
        streamer.sender = Mock(spec=TensorSender)

        expected_num_task_data_sent = 0

        # Mock the clean function to avoid actual cleaning during test
        with patch("nvflare.app_opt.tensor_stream.server.clean_task_data") as mock_clean:
            num_clients = len(mock_engine_with_clients.get_clients())

            for _ in range(num_clients):
                expected_num_task_data_sent += 1
                streamer.send_tensors_to_client(mock_fl_context)
                assert streamer.num_task_data_sent == expected_num_task_data_sent

                if num_clients != expected_num_task_data_sent:
                    # Should not clean yet (not all clients received)
                    mock_clean.assert_not_called()

            # then call try_to_clean_task_data
            streamer.try_to_clean_task_data(mock_fl_context)

            # Now should clean (all clients received)
            mock_clean.assert_called_once_with(mock_fl_context)
            assert streamer.data_cleaned is True
            assert streamer.num_task_data_sent == 0  # Reset after cleaning

    def test_edge_case_no_clients(self, mock_fl_context):
        """Test behavior when no clients are connected."""
        # Create mock engine with no clients
        mock_engine = Mock(spec=StreamableEngine)
        mock_engine.get_clients = Mock(return_value=[])

        streamer = TensorServerStreamer()
        streamer.engine = mock_engine
        streamer.sender = Mock(spec=TensorSender)

        with patch("nvflare.app_opt.tensor_stream.server.clean_task_data") as mock_clean:
            # Send tensors (should immediately clean since 0 clients)
            streamer.send_tensors_to_client(mock_fl_context)

            # Should clean immediately since no clients to wait for
            assert streamer.num_task_data_sent == 1

    def test_format_parameter_propagation(self, mock_fl_context, mock_engine_with_clients):
        """Test that format parameter is properly propagated to receiver."""
        with patch("nvflare.app_opt.tensor_stream.server.TensorReceiver") as mock_receiver_class:
            mock_fl_context.get_engine.return_value = mock_engine_with_clients

            # Test with numpy format
            streamer = TensorServerStreamer(format=ExchangeFormat.NUMPY)
            streamer.initialize(mock_fl_context)

            # Verify receiver was created with correct format
            mock_receiver_class.assert_called_once_with(
                mock_engine_with_clients,
                FLContextKey.TASK_RESULT,
                ExchangeFormat.NUMPY,  # Should pass the format parameter
            )

    def test_root_keys_parameter_propagation(self, mock_fl_context, mock_engine_with_clients):
        """Test that root_keys parameter is properly propagated to sender."""
        with patch("nvflare.app_opt.tensor_stream.server.TensorSender") as mock_sender_class:
            with patch("nvflare.app_opt.tensor_stream.server.TensorReceiver"):
                mock_fl_context.get_engine.return_value = mock_engine_with_clients

                # Test with custom root keys
                custom_keys = ["encoder", "decoder", "head"]
                streamer = TensorServerStreamer(root_keys=custom_keys)
                streamer.initialize(mock_fl_context)

                # Verify sender was created with correct root keys
                mock_sender_class.assert_called_once_with(
                    mock_engine_with_clients, FLContextKey.TASK_DATA, custom_keys  # Should pass the custom root keys
                )
