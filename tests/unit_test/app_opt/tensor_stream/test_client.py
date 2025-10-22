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
from nvflare.app_opt.tensor_stream.client import TensorClientStreamer
from nvflare.app_opt.tensor_stream.receiver import TensorReceiver
from nvflare.app_opt.tensor_stream.sender import TensorSender
from nvflare.client.config import ExchangeFormat


class TestTensorClientStreamer:
    """Test cases for TensorClientStreamer class."""

    @pytest.mark.parametrize(
        "format_type,tasks,entry_timeout,expected_tasks",
        [
            (ExchangeFormat.PYTORCH, None, 30.0, ["train"]),  # Default values
            (ExchangeFormat.NUMPY, ["custom_task"], 15.0, ["custom_task"]),  # Custom values
            (ExchangeFormat.PYTORCH, ["train", "eval"], 45.0, ["train", "eval"]),  # Multiple tasks
        ],
    )
    def test_init_parameters(self, format_type, tasks, entry_timeout, expected_tasks):
        """Test TensorClientStreamer initialization with various parameters."""
        streamer = TensorClientStreamer(format=format_type, tasks=tasks, entry_timeout=entry_timeout)

        assert streamer.format == format_type
        assert streamer.tasks == expected_tasks
        assert streamer.entry_timeout == entry_timeout
        # All components should be None before initialization
        assert streamer.engine is None
        assert streamer.sender is None
        assert streamer.receiver is None

    @patch("nvflare.app_opt.tensor_stream.client.TensorReceiver")
    def test_initialize_success(self, mock_receiver_class, mock_fl_context, mock_engine_with_clients):
        """Test successful initialization of TensorClientStreamer."""
        # Setup mocks
        mock_fl_context.get_engine.return_value = mock_engine_with_clients
        mock_receiver_instance = Mock(spec=TensorReceiver)
        mock_receiver_class.return_value = mock_receiver_instance

        # Create and initialize streamer
        streamer = TensorClientStreamer(format=ExchangeFormat.PYTORCH, tasks=["train"])
        streamer.initialize(mock_fl_context)

        # Verify engine assignment
        assert streamer.engine == mock_engine_with_clients

        # Verify receiver creation (client receives TASK_DATA from server)
        mock_receiver_class.assert_called_once_with(
            mock_engine_with_clients, FLContextKey.TASK_DATA, ExchangeFormat.PYTORCH
        )
        assert streamer.receiver == mock_receiver_instance

        # Verify sender is not created during initialization
        assert streamer.sender is None

    def test_initialize_no_engine(self, mock_fl_context):
        """Test initialization when no engine is found."""
        mock_fl_context.get_engine.return_value = None

        streamer = TensorClientStreamer()

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

        streamer = TensorClientStreamer()
        streamer.system_panic = Mock()

        streamer.initialize(mock_fl_context)

        # Verify system_panic was called
        streamer.system_panic.assert_called_once()
        args = streamer.system_panic.call_args[0]
        assert "Engine is not a StreamableEngine" in args[0]
        assert args[1] == mock_fl_context

    @patch("nvflare.app_opt.tensor_stream.client.TensorReceiver")
    def test_initialize_receiver_exception(self, mock_receiver_class, mock_fl_context, mock_engine_with_clients):
        """Test initialization when TensorReceiver creation fails."""
        mock_fl_context.get_engine.return_value = mock_engine_with_clients
        mock_receiver_class.side_effect = Exception("Receiver creation failed")

        streamer = TensorClientStreamer()
        streamer.system_panic = Mock()

        streamer.initialize(mock_fl_context)

        # Verify system_panic was called with exception message
        streamer.system_panic.assert_called_once_with("Receiver creation failed", mock_fl_context)

    @patch("nvflare.app_opt.tensor_stream.client.TensorReceiver")
    def test_handle_event_start_run(self, mock_receiver_class, mock_fl_context, mock_engine_with_clients):
        """Test handling START_RUN event."""
        mock_fl_context.get_engine.return_value = mock_engine_with_clients

        streamer = TensorClientStreamer()

        # Handle START_RUN event
        streamer.handle_event(EventType.START_RUN, mock_fl_context)

        # Verify initialization was called (receiver should be created, sender should not)
        assert streamer.engine == mock_engine_with_clients
        assert streamer.sender is None
        assert streamer.receiver is not None

    def test_handle_event_before_task_data_filter(self, mock_fl_context):
        """Test handling BEFORE_TASK_DATA_FILTER event."""
        streamer = TensorClientStreamer()
        mock_receiver = Mock(spec=TensorReceiver)
        streamer.receiver = mock_receiver

        # Mock the required fl_ctx methods
        mock_peer_context = Mock()
        mock_peer_context.get_identity_name.return_value = "client1"
        mock_fl_context.get_peer_context.return_value = mock_peer_context
        mock_fl_context.get_prop.return_value = "task_123"

        # Handle BEFORE_TASK_DATA_FILTER event
        streamer.handle_event(EventType.BEFORE_TASK_DATA_FILTER, mock_fl_context)

        # Verify receiver.wait_for_tensors was called
        mock_receiver.wait_for_tensors.assert_called_once_with("task_123", "client1")

        # Verify receiver.set_ctx_with_tensors was called
        mock_receiver.set_ctx_with_tensors.assert_called_once_with(mock_fl_context)

    @patch("nvflare.app_opt.tensor_stream.client.TensorSender")
    def test_handle_event_after_task_result_filter(self, mock_sender_class, mock_fl_context, mock_engine_with_clients):
        """Test handling AFTER_TASK_RESULT_FILTER event."""
        streamer = TensorClientStreamer(format=ExchangeFormat.PYTORCH, tasks=["train"])
        streamer.engine = mock_engine_with_clients
        streamer.send_tensors_to_server = Mock()

        # Handle AFTER_TASK_RESULT_FILTER event
        streamer.handle_event(EventType.AFTER_TASK_RESULT_FILTER, mock_fl_context)

        # Verify sender was created with correct parameters
        mock_sender_class.assert_called_once_with(
            mock_engine_with_clients, FLContextKey.TASK_RESULT, ExchangeFormat.PYTORCH, ["train"]
        )

        # Verify send_tensors_to_server was called
        streamer.send_tensors_to_server.assert_called_once_with(mock_fl_context)

    @patch("nvflare.app_opt.tensor_stream.client.TensorSender")
    def test_handle_event_after_task_result_filter_exception(
        self, mock_sender_class, mock_fl_context, mock_engine_with_clients
    ):
        """Test handling AFTER_TASK_RESULT_FILTER event when send_tensors_to_server raises exception."""
        streamer = TensorClientStreamer()
        streamer.engine = mock_engine_with_clients
        streamer.send_tensors_to_server = Mock(side_effect=Exception("Send failed"))
        streamer.system_panic = Mock()

        # Handle AFTER_TASK_RESULT_FILTER event
        streamer.handle_event(EventType.AFTER_TASK_RESULT_FILTER, mock_fl_context)

        # Verify system_panic was called
        streamer.system_panic.assert_called_once_with("Send failed", mock_fl_context)

    @patch("nvflare.app_opt.tensor_stream.client.TensorSender")
    def test_handle_event_after_task_result_filter_sender_creation_exception(
        self, mock_sender_class, mock_fl_context, mock_engine_with_clients
    ):
        """Test handling AFTER_TASK_RESULT_FILTER event when TensorSender creation fails."""
        streamer = TensorClientStreamer()
        streamer.engine = mock_engine_with_clients
        mock_sender_class.side_effect = Exception("Sender creation failed")

        # Handle AFTER_TASK_RESULT_FILTER event - should raise exception since sender creation is not wrapped in try-catch
        with pytest.raises(Exception, match="Sender creation failed"):
            streamer.handle_event(EventType.AFTER_TASK_RESULT_FILTER, mock_fl_context)

    @patch("nvflare.app_opt.tensor_stream.client.clean_task_result")
    def test_send_tensors_to_server_success(self, mock_clean_task_result, mock_fl_context):
        """Test successful tensor sending to server."""
        streamer = TensorClientStreamer(entry_timeout=7.0)

        # Mock sender - send() should not raise ValueError for success
        mock_sender = Mock(spec=TensorSender)
        streamer.sender = mock_sender

        # Send tensors
        streamer.send_tensors_to_server(mock_fl_context)

        # Verify sender.store_tensors was called first
        mock_sender.store_tensors.assert_called_once_with(mock_fl_context)

        # Verify sender.send was called with correct parameters
        mock_sender.send.assert_called_once_with(mock_fl_context, 7.0)

        # Verify clean_task_result was called when send succeeds (no ValueError)
        mock_clean_task_result.assert_called_once_with(mock_fl_context)

        # Verify sender is set to None after successful send
        assert streamer.sender is None

    @patch("nvflare.app_opt.tensor_stream.client.TensorSender")
    @patch("nvflare.app_opt.tensor_stream.client.TensorReceiver")
    def test_complete_workflow(self, mock_receiver_class, mock_sender_class, mock_fl_context, mock_engine_with_clients):
        """Test complete client workflow: initialization, receiving task data, sending results."""
        # Setup mocks
        mock_fl_context.get_engine.return_value = mock_engine_with_clients
        mock_peer_context = Mock()
        mock_peer_context.get_identity_name.return_value = "client1"
        mock_fl_context.get_peer_context.return_value = mock_peer_context
        mock_fl_context.get_prop.return_value = "task_123"

        mock_sender_instance = Mock(spec=TensorSender)
        mock_receiver_instance = Mock(spec=TensorReceiver)
        mock_sender_class.return_value = mock_sender_instance
        mock_receiver_class.return_value = mock_receiver_instance

        # Create streamer
        streamer = TensorClientStreamer(format=ExchangeFormat.PYTORCH, tasks=["train"], entry_timeout=5.0)

        # Step 1: Handle START_RUN event (initialization)
        streamer.handle_event(EventType.START_RUN, mock_fl_context)

        # Verify initialization - receiver created, sender not yet created
        assert streamer.engine == mock_engine_with_clients
        assert streamer.sender is None
        assert streamer.receiver == mock_receiver_instance

        # Step 2: Handle BEFORE_TASK_DATA_FILTER event (receive tensors from server)
        streamer.handle_event(EventType.BEFORE_TASK_DATA_FILTER, mock_fl_context)

        # Verify receiver methods were called
        mock_receiver_instance.wait_for_tensors.assert_called_once_with("task_123", "client1")
        mock_receiver_instance.set_ctx_with_tensors.assert_called_once_with(mock_fl_context)

        # Step 3: Handle AFTER_TASK_RESULT_FILTER event (send tensors to server)
        with patch("nvflare.app_opt.tensor_stream.client.clean_task_result") as mock_clean:
            # Mock sender to not raise ValueError for successful send
            streamer.handle_event(EventType.AFTER_TASK_RESULT_FILTER, mock_fl_context)

            # Verify sender was created and called
            mock_sender_class.assert_called_once_with(
                mock_engine_with_clients, FLContextKey.TASK_RESULT, ExchangeFormat.PYTORCH, ["train"]
            )
            # After successful send (no ValueError), sender is set to None to release references
            assert streamer.sender is None
            mock_sender_instance.store_tensors.assert_called_once_with(mock_fl_context)
            mock_sender_instance.send.assert_called_once_with(mock_fl_context, 5.0)

            # Verify cleanup was called
            mock_clean.assert_called_once_with(mock_fl_context)

    @pytest.mark.parametrize(
        "format_type,tasks",
        [
            (ExchangeFormat.NUMPY, ["custom_task"]),
            (ExchangeFormat.PYTORCH, ["train"]),
            (ExchangeFormat.PYTORCH, ["train", "validate"]),
        ],
    )
    def test_parameter_propagation_to_components(self, mock_fl_context, mock_engine_with_clients, format_type, tasks):
        """Test that parameters are properly propagated to sender and receiver."""
        with patch("nvflare.app_opt.tensor_stream.client.TensorReceiver") as mock_receiver_class:
            with patch("nvflare.app_opt.tensor_stream.client.TensorSender") as mock_sender_class:
                mock_fl_context.get_engine.return_value = mock_engine_with_clients

                streamer = TensorClientStreamer(format=format_type, tasks=tasks)
                streamer.initialize(mock_fl_context)

                # Verify receiver created with correct format during initialization
                mock_receiver_class.assert_called_once_with(
                    mock_engine_with_clients, FLContextKey.TASK_DATA, format_type
                )

                # Verify sender is not created during initialization
                mock_sender_class.assert_not_called()

                # Now trigger AFTER_TASK_RESULT_FILTER event to create sender
                streamer.handle_event(EventType.AFTER_TASK_RESULT_FILTER, mock_fl_context)

                # Verify sender created with correct parameters
                mock_sender_class.assert_called_once_with(
                    mock_engine_with_clients, FLContextKey.TASK_RESULT, format_type, tasks
                )

    def test_event_handling_edge_cases(self, mock_fl_context):
        """Test handling of unexpected or unhandled events."""
        streamer = TensorClientStreamer()

        # Mock methods to ensure they're not called for unhandled events
        streamer.initialize = Mock()
        streamer.send_tensors_to_server = Mock()
        mock_receiver = Mock(spec=TensorReceiver)
        streamer.receiver = mock_receiver

        # Handle an unrelated event
        streamer.handle_event("SOME_OTHER_EVENT", mock_fl_context)

        # Verify no methods were called
        streamer.initialize.assert_not_called()
        streamer.send_tensors_to_server.assert_not_called()
        mock_receiver.set_ctx_with_tensors.assert_not_called()

    @patch("nvflare.app_opt.tensor_stream.client.clean_task_result")
    def test_send_tensors_with_cleanup(self, mock_clean_task_result, mock_fl_context):
        """Test that send_tensors_to_server calls cleanup after successful send."""
        custom_timeout = 42.0
        streamer = TensorClientStreamer(entry_timeout=custom_timeout)

        # Mock sender for successful case (returns True)
        mock_sender = Mock(spec=TensorSender)
        mock_sender.send.return_value = True
        mock_sender.root_keys = []  # Add root_keys attribute that will be cleared
        streamer.sender = mock_sender

        # Call send_tensors_to_server
        streamer.send_tensors_to_server(mock_fl_context)

        # Verify sender was called with correct timeout and cleanup was called
        mock_sender.send.assert_called_once_with(mock_fl_context, custom_timeout)
        mock_clean_task_result.assert_called_once_with(mock_fl_context)

    def test_send_tensors_exception_no_cleanup(self, mock_fl_context):
        """Test that exceptions in sender.send prevent cleanup from being called."""
        streamer = TensorClientStreamer()

        # Mock sender to raise exception
        mock_sender = Mock(spec=TensorSender)
        mock_sender.send.side_effect = Exception("Send failed")
        streamer.sender = mock_sender

        # Should raise exception and not call cleanup
        with pytest.raises(Exception, match="Send failed"):
            streamer.send_tensors_to_server(mock_fl_context)

    def test_component_state_after_initialization(self, mock_fl_context, mock_engine_with_clients):
        """Test that components are properly set up after initialization."""
        with patch("nvflare.app_opt.tensor_stream.client.TensorSender") as mock_sender_class:
            with patch("nvflare.app_opt.tensor_stream.client.TensorReceiver") as mock_receiver_class:
                with patch("nvflare.app_opt.tensor_stream.client.clean_task_result"):
                    mock_fl_context.get_engine.return_value = mock_engine_with_clients
                    mock_sender_instance = Mock(spec=TensorSender)
                    mock_receiver_instance = Mock(spec=TensorReceiver)
                    # Mock send to raise ValueError so sender is not cleared (stays set)
                    mock_sender_instance.send.side_effect = ValueError("Send failed")
                    mock_sender_class.return_value = mock_sender_instance
                    mock_receiver_class.return_value = mock_receiver_instance

                    streamer = TensorClientStreamer()

                    # Before initialization
                    assert streamer.engine is None
                    assert streamer.sender is None
                    assert streamer.receiver is None

                    # Initialize
                    streamer.initialize(mock_fl_context)

                    # After initialization - only receiver should be created
                    assert streamer.engine == mock_engine_with_clients
                    assert streamer.sender is None
                    assert streamer.receiver == mock_receiver_instance

                    # Sender should be created when handling AFTER_TASK_RESULT_FILTER
                    # ValueError is caught and passed, sender remains set
                    streamer.handle_event(EventType.AFTER_TASK_RESULT_FILTER, mock_fl_context)
                    # Since send raises ValueError, sender is NOT cleared (exception is caught)
                    assert streamer.sender == mock_sender_instance

    def test_error_propagation_in_event_handling(self, mock_fl_context):
        """Test that exceptions in event handling are properly handled."""
        streamer = TensorClientStreamer()
        streamer.system_panic = Mock()

        # Test exception in receiver.set_ctx_with_tensors during BEFORE_TASK_DATA_FILTER
        mock_receiver = Mock(spec=TensorReceiver)
        mock_receiver.wait_for_tensors.side_effect = Exception("Receiver error")
        streamer.receiver = mock_receiver

        # Mock the required fl_ctx methods
        mock_peer_context = Mock()
        mock_peer_context.get_identity_name.return_value = "client1"
        mock_fl_context.get_peer_context.return_value = mock_peer_context
        mock_fl_context.get_prop.return_value = "task_123"

        # Exception should be caught and system_panic should be called
        streamer.handle_event(EventType.BEFORE_TASK_DATA_FILTER, mock_fl_context)

        # Verify system_panic was called with the exception message
        streamer.system_panic.assert_called_once_with("Receiver error", mock_fl_context)

    def test_initialization_component_creation_order(self, mock_fl_context, mock_engine_with_clients):
        """Test that receiver is created during initialization, sender created later."""
        with patch("nvflare.app_opt.tensor_stream.client.TensorSender") as mock_sender_class:
            with patch("nvflare.app_opt.tensor_stream.client.TensorReceiver") as mock_receiver_class:
                mock_fl_context.get_engine.return_value = mock_engine_with_clients

                # Track call order
                call_order = []

                def receiver_side_effect(*args, **kwargs):
                    call_order.append("receiver")
                    return Mock()

                def sender_side_effect(*args, **kwargs):
                    call_order.append("sender")
                    return Mock()

                mock_receiver_class.side_effect = receiver_side_effect
                mock_sender_class.side_effect = sender_side_effect

                streamer = TensorClientStreamer()
                streamer.initialize(mock_fl_context)

                # Verify only receiver is created during initialization
                assert call_order == ["receiver"]

                # Now trigger sender creation
                streamer.handle_event(EventType.AFTER_TASK_RESULT_FILTER, mock_fl_context)

                # Verify sender is created after receiver
                assert call_order == ["receiver", "sender"]

    @patch("nvflare.app_opt.tensor_stream.client.clean_task_result")
    def test_send_tensors_cleanup_not_called_when_sender_fails(self, mock_clean_task_result, mock_fl_context):
        """Test that clean_task_result is not called when sender fails."""
        streamer = TensorClientStreamer()

        # Mock sender to raise exception
        mock_sender = Mock(spec=TensorSender)
        mock_sender.send.side_effect = Exception("Send failed")
        streamer.sender = mock_sender

        # Should raise exception and not call cleanup
        with pytest.raises(Exception, match="Send failed"):
            streamer.send_tensors_to_server(mock_fl_context)

        # Verify cleanup was not called due to exception
        mock_clean_task_result.assert_not_called()

    def test_different_event_types_integration(self, mock_fl_context, mock_engine_with_clients):
        """Test handling multiple different event types in sequence."""
        with patch("nvflare.app_opt.tensor_stream.client.TensorSender") as mock_sender_class:
            with patch("nvflare.app_opt.tensor_stream.client.TensorReceiver") as mock_receiver_class:
                mock_fl_context.get_engine.return_value = mock_engine_with_clients
                mock_peer_context = Mock()
                mock_peer_context.get_identity_name.return_value = "client1"
                mock_fl_context.get_peer_context.return_value = mock_peer_context
                mock_fl_context.get_prop.return_value = "task_123"

                mock_sender_instance = Mock(spec=TensorSender)
                mock_receiver_instance = Mock(spec=TensorReceiver)
                mock_sender_class.return_value = mock_sender_instance
                mock_receiver_class.return_value = mock_receiver_instance

                streamer = TensorClientStreamer()

                # 1. START_RUN - should create receiver but not sender
                streamer.handle_event(EventType.START_RUN, mock_fl_context)
                assert streamer.engine is not None
                assert streamer.receiver is not None
                assert streamer.sender is None

                # 2. BEFORE_TASK_DATA_FILTER
                streamer.handle_event(EventType.BEFORE_TASK_DATA_FILTER, mock_fl_context)
                mock_receiver_instance.wait_for_tensors.assert_called_once_with("task_123", "client1")
                mock_receiver_instance.set_ctx_with_tensors.assert_called_once()

                # 3. AFTER_TASK_RESULT_FILTER - should create sender and call send
                with patch("nvflare.app_opt.tensor_stream.client.clean_task_result"):
                    # Mock sender to not raise ValueError for successful send
                    streamer.handle_event(EventType.AFTER_TASK_RESULT_FILTER, mock_fl_context)

                    # Verify sender was created
                    mock_sender_class.assert_called_once()
                    # After successful send (no ValueError), sender is set to None to release references
                    assert streamer.sender is None
                    mock_sender_instance.store_tensors.assert_called_once()
                    mock_sender_instance.send.assert_called_once()

                # Verify all components were used
                assert mock_receiver_instance.set_ctx_with_tensors.called
                assert mock_sender_instance.send.called

    @patch("nvflare.app_opt.tensor_stream.client.clean_task_result")
    def test_send_tensors_to_server_no_cleanup_when_send_fails(self, mock_clean_task_result, mock_fl_context):
        """Test that cleanup is not called when sender.send raises ValueError."""
        streamer = TensorClientStreamer(entry_timeout=5.0)

        # Mock sender to raise ValueError (unsuccessful send)
        mock_sender = Mock(spec=TensorSender)
        mock_sender.send.side_effect = ValueError("Send failed")
        streamer.sender = mock_sender

        # Send tensors - ValueError is caught and passed
        streamer.send_tensors_to_server(mock_fl_context)

        # Verify sender.store_tensors was called
        mock_sender.store_tensors.assert_called_once_with(mock_fl_context)

        # Verify sender.send was called
        mock_sender.send.assert_called_once_with(mock_fl_context, 5.0)

        # Verify clean_task_result was NOT called when send raises ValueError
        mock_clean_task_result.assert_not_called()

        # Verify sender is NOT set to None when send fails (exception caught)
        assert streamer.sender == mock_sender

    def test_tasks_parameter_defaults(self):
        """Test that tasks parameter defaults are handled correctly."""
        # Test default value
        streamer = TensorClientStreamer()
        assert streamer.tasks == ["train"]

        # Test custom single task
        streamer = TensorClientStreamer(tasks=["custom_task"])
        assert streamer.tasks == ["custom_task"]

        # Test multiple tasks
        streamer = TensorClientStreamer(tasks=["train", "evaluate", "validate"])
        assert streamer.tasks == ["train", "evaluate", "validate"]

        # Test empty list
        streamer = TensorClientStreamer(tasks=[])
        assert streamer.tasks == []
