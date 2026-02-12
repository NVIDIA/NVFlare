# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""
Test to demonstrate the potential deadlock issue in swarm learning when a client
broadcasts to itself with synchronous self-message processing.

The issue: When _send_direct_message is called for a target that is the same cell,
the message is processed synchronously on the same thread. If the message handler
performs any blocking operation (like waiting for tensor streaming), deadlock occurs.

This test verifies:
1. The synchronous self-message behavior exists (using actual CoreCell)
2. Our proposed fix (excluding self and using set_learn_task) works correctly
3. TensorStreamer-like blocking would cause deadlock in sync self-message

Run with:
    pytest tests/unit_test/app_common/ccwf/test_swarm_self_message_deadlock.py -v
"""

import copy
import random
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from nvflare.apis.fl_constant import ReturnCode as FLReturnCode
from nvflare.apis.fl_context import FLContextManager
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learnable import Learnable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.ccwf.common import Constant
from nvflare.app_common.ccwf.swarm_client_ctl import SwarmClientController
from nvflare.fuel.f3.cellnet.core_cell import CoreCell, Message, MessageHeaderKey, TargetMessage
from nvflare.fuel.f3.cellnet.defs import ReturnCode
from nvflare.fuel.utils.network_utils import get_open_ports


class TestSelfMessageDeadlock(unittest.TestCase):
    """Test cases to demonstrate and verify the self-message deadlock scenario."""

    def test_sync_self_message_blocks_same_thread(self):
        """
        Demonstrate that a synchronous self-message blocks the calling thread.

        This simulates what happens in core_cell.py when a message is sent to
        the same cell (direct_cell) - the message is processed synchronously.
        """
        call_order = []
        lock = threading.Lock()

        def simulate_send_direct_message(handler):
            """Simulates _send_direct_message which calls handler synchronously."""
            with lock:
                call_order.append("send_start")
            # Synchronous call - blocks until handler completes
            handler()
            with lock:
                call_order.append("send_end")

        def blocking_handler():
            """Simulates a handler that would block (e.g., waiting for tensors)."""
            with lock:
                call_order.append("handler_start")
            # Simulate some work
            time.sleep(0.1)
            with lock:
                call_order.append("handler_end")

        # When we call send_direct_message, it blocks until handler completes
        simulate_send_direct_message(blocking_handler)

        # Verify the call order shows synchronous execution
        self.assertEqual(call_order, ["send_start", "handler_start", "handler_end", "send_end"])

    def test_async_local_task_prevents_deadlock(self):
        """
        Demonstrate that queuing a local task (like set_learn_task) prevents deadlock.

        This simulates our proposed fix: instead of sending a message to self,
        we directly queue the task using set_learn_task which is non-blocking.
        """
        call_order = []
        task_queue = []
        task_processed = threading.Event()

        def simulate_set_learn_task(task_data):
            """Simulates set_learn_task which queues the task for async processing."""
            call_order.append("queue_task")
            task_queue.append(task_data)
            # This returns immediately - non-blocking

        def worker_thread():
            """Simulates the _do_learn thread that processes queued tasks."""
            while not task_processed.is_set():
                if task_queue:
                    task_queue.pop(0)
                    call_order.append("process_task_start")
                    time.sleep(0.1)  # Simulate processing
                    call_order.append("process_task_end")
                    task_processed.set()
                time.sleep(0.01)

        # Start worker thread (like _do_learn)
        worker = threading.Thread(target=worker_thread, daemon=True)
        worker.start()

        # Queue task locally (like our fix does)
        call_order.append("main_start")
        simulate_set_learn_task({"round": 1})
        call_order.append("main_continues")  # Main thread is NOT blocked!

        # Wait for task to be processed
        task_processed.wait(timeout=2.0)
        call_order.append("main_end")

        # Verify main thread wasn't blocked while task was queued
        # "main_continues" should appear before "process_task_start"
        main_continues_idx = call_order.index("main_continues")
        process_start_idx = call_order.index("process_task_start")

        self.assertLess(
            main_continues_idx,
            process_start_idx,
            "Main thread should continue before task processing starts (non-blocking)",
        )

    def test_scatter_with_self_in_targets(self):
        """
        Simulate the _scatter scenario where self is in targets.

        Original behavior: broadcast_and_wait to all targets including self
        - This would cause synchronous self-message

        Fixed behavior: exclude self from network broadcast, queue locally
        - This prevents the synchronous call
        """
        me = "site-2"
        clients = ["site-1", "site-2", "site-3"]

        # Track what happens
        network_sends = []
        local_queues = []

        def simulate_send_learn_task(targets):
            """Simulates send_learn_task (network broadcast)."""
            for t in targets:
                network_sends.append(t)

        def simulate_set_learn_task():
            """Simulates set_learn_task (local queue)."""
            local_queues.append(me)

        # Fixed behavior (matches production code order)
        should_queue_locally = me in clients
        network_targets = [t for t in clients if t != me]

        # Queue locally FIRST with deep copy (matches swarm_client_ctl.py lines 501-505)
        # Deep copy needed to avoid race condition with send_learn_task modifying task_data
        if should_queue_locally:
            simulate_set_learn_task()

        # Then send to network targets (matches swarm_client_ctl.py lines 508-513)
        if network_targets:
            simulate_send_learn_task(network_targets)

        # Verify the fix
        self.assertNotIn(me, network_sends, "Self should NOT be in network broadcast targets")
        self.assertIn(me, local_queues, "Self should be queued locally instead")
        self.assertEqual(
            set(network_sends), {"site-1", "site-3"}, "Other clients should still receive network broadcast"
        )


class TestSwarmScatterFix(unittest.TestCase):
    """Test the actual fix logic that should be applied to swarm_client_ctl.py"""

    def test_fix_logic_all_clients_are_trainers(self):
        """Test when all clients including self are trainers."""
        me = "site-1"
        train_clients = ["site-1", "site-2", "site-3"]
        aggr_clients = ["site-1", "site-2", "site-3"]

        # Simulate fixed _scatter logic
        aggr = random.choice(aggr_clients)
        targets = copy.copy(train_clients)
        if aggr not in targets:
            targets.append(aggr)

        should_queue_locally = False
        if me in targets:
            targets.remove(me)
            should_queue_locally = True

        self.assertTrue(should_queue_locally, "Self should be queued locally")
        self.assertNotIn(me, targets, "Self should not be in network targets")
        self.assertGreater(len(targets), 0, "Other targets should remain")

    def test_fix_logic_self_not_a_trainer(self):
        """Test when self is not a trainer (e.g., only aggregator)."""
        me = "site-1"
        train_clients = ["site-2", "site-3"]  # site-1 not a trainer
        aggr_clients = ["site-1"]  # site-1 is aggregator only

        aggr = random.choice(aggr_clients)
        targets = copy.copy(train_clients)
        if aggr not in targets:
            targets.append(aggr)  # site-1 added as aggr

        should_queue_locally = False
        if me in targets:
            targets.remove(me)
            should_queue_locally = True

        self.assertTrue(should_queue_locally, "Self should be queued locally even if only aggregator")
        self.assertNotIn(me, targets)

    def test_fix_logic_only_self(self):
        """Test edge case where self is the only client."""
        me = "site-1"
        train_clients = ["site-1"]
        aggr_clients = ["site-1"]

        aggr = random.choice(aggr_clients)
        targets = copy.copy(train_clients)
        if aggr not in targets:
            targets.append(aggr)

        should_queue_locally = False
        if me in targets:
            targets.remove(me)
            should_queue_locally = True

        # When only self, targets becomes empty after removing self
        self.assertTrue(should_queue_locally)
        self.assertEqual(len(targets), 0, "No network targets when only self")


class TestRealDeadlockScenario(unittest.TestCase):
    """Test that demonstrates a real deadlock scenario with blocking handlers."""

    def test_deadlock_with_blocking_handler_timeout(self):
        """
        This test simulates the ACTUAL deadlock scenario:

        1. Client A (site-2) calls broadcast_and_wait to [site-1, site-2, site-3]
        2. For site-2 (self), _send_direct_message is called synchronously
        3. The message handler (e.g., TensorStreamer) tries to wait for something
        4. DEADLOCK: The thread is blocked waiting for itself

        We use a timeout to detect the deadlock.
        """
        deadlock_detected = threading.Event()
        operation_completed = threading.Event()
        handler_started = threading.Event()

        def blocking_message_handler():
            """
            Simulates a handler that waits for external data (like TensorReceiver.wait_for_tensors).
            In the real scenario, this is waiting for streaming data that would be sent
            by the same thread that's now blocked.
            """
            handler_started.set()
            # Simulate waiting for tensors - this would block forever in real deadlock
            # We use a timeout to detect this
            wait_event = threading.Event()
            # This wait will never complete because no one will set it
            # (the thread that would set it is this same thread, which is blocked)
            result = wait_event.wait(timeout=0.5)  # Short timeout for test
            if not result:
                deadlock_detected.set()
                return False
            return True

        def simulate_sync_self_message():
            """Simulates what happens when _send_direct_message is called for self."""
            # This is synchronous - blocks until handler returns
            success = blocking_message_handler()
            if success:
                operation_completed.set()

        # Run in a thread so we can check for deadlock
        thread = threading.Thread(target=simulate_sync_self_message)
        thread.start()

        # Wait for handler to start
        handler_started.wait(timeout=1.0)

        # Wait for either completion or deadlock detection
        thread.join(timeout=2.0)

        # Verify deadlock was detected (handler timed out waiting)
        self.assertTrue(
            deadlock_detected.is_set(), "Deadlock should be detected - handler blocked waiting for external event"
        )
        self.assertFalse(operation_completed.is_set(), "Operation should not complete due to deadlock")


# =============================================================================
# CORECELL-BASED TESTS
# These tests use actual CoreCell to prove the synchronous self-message behavior
# =============================================================================


class TestCoreCellSelfMessage(unittest.TestCase):
    """Test cases using actual CoreCell to prove synchronous self-message behavior."""

    @classmethod
    def setUpClass(cls):
        """Set up a CoreCell for testing, preserving existing ALL_CELLS state."""
        # Save existing ALL_CELLS state for restoration
        cls._saved_all_cells = dict(CoreCell.ALL_CELLS)

        # Use unique cell name based on port to avoid conflicts
        cls.port = get_open_ports(1)[0]
        cls.cell_name = f"test_cell_self_msg_{cls.port}"
        cls.listening_url = f"tcp://localhost:{cls.port}"

        cls.cell = CoreCell(fqcn=cls.cell_name, root_url=cls.listening_url, secure=False, credentials={})
        cls.cell.start()
        # Wait for cell to be registered in ALL_CELLS (deterministic check)
        for _ in range(50):  # Up to 5 seconds
            if cls.cell_name in CoreCell.ALL_CELLS:
                break
            time.sleep(0.1)
        assert cls.cell_name in CoreCell.ALL_CELLS, "Cell failed to register"

    @classmethod
    def tearDownClass(cls):
        """Clean up the cell and restore ALL_CELLS state."""
        try:
            cls.cell.stop()
        finally:
            # Remove our test cell and restore saved state
            CoreCell.ALL_CELLS.pop(cls.cell_name, None)
            for name, cell in cls._saved_all_cells.items():
                if name not in CoreCell.ALL_CELLS:
                    CoreCell.ALL_CELLS[name] = cell

    def test_self_in_all_cells(self):
        """Verify that the cell is registered in ALL_CELLS (prerequisite for sync self-message)."""
        self.assertIn(self.cell_name, CoreCell.ALL_CELLS)
        self.assertEqual(CoreCell.ALL_CELLS[self.cell_name], self.cell)

    def test_sync_self_message_same_thread_real_cell(self):
        """
        Prove that sending a message to self runs the handler on the SAME thread.

        This is the core of the issue: when _send_direct_message is called,
        it synchronously calls target_cell.process_message() on the same thread.
        """
        handler_thread_id = None
        sender_thread_id = threading.current_thread().ident
        handler_called = threading.Event()

        # Use unique channel/topic for this test to avoid cross-test interference
        test_channel = f"channel_sync_{id(self)}"
        test_topic = f"topic_sync_{id(self)}"

        def message_handler(message: Message):
            nonlocal handler_thread_id
            handler_thread_id = threading.current_thread().ident
            handler_called.set()
            return Message(headers={MessageHeaderKey.RETURN_CODE: ReturnCode.OK})

        self.cell.register_request_cb(channel=test_channel, topic=test_topic, cb=message_handler)

        request = Message(headers={}, payload=b"test")
        self.cell.fire_and_forget(channel=test_channel, topic=test_topic, targets=[self.cell_name], message=request)

        # Verify handler was actually called before comparing thread IDs
        self.assertTrue(handler_called.wait(timeout=5.0), "Handler should have been called")

        # CRITICAL: Handler runs on the SAME thread as sender (synchronous call)
        self.assertEqual(
            sender_thread_id,
            handler_thread_id,
            "Handler should run on the SAME thread as sender (synchronous call)",
        )

    def test_blocking_handler_would_deadlock_real_cell(self):
        """
        Demonstrate that a blocking handler would cause deadlock.
        Uses a timeout to detect the deadlock scenario.
        """
        # Use unique channel/topic for this test
        test_channel = f"channel_deadlock_{id(self)}"
        test_topic = f"topic_deadlock_{id(self)}"

        external_trigger = threading.Event()
        handler_completed = threading.Event()
        deadlock_detected = False

        def blocking_handler(message: Message):
            nonlocal deadlock_detected
            try:
                # Simulate waiting for something that would be triggered by the caller
                result = external_trigger.wait(timeout=1.0)
                if not result:
                    deadlock_detected = True
                return Message(headers={MessageHeaderKey.RETURN_CODE: ReturnCode.OK})
            finally:
                handler_completed.set()

        self.cell.register_request_cb(channel=test_channel, topic=test_topic, cb=blocking_handler)

        request = Message(headers={}, payload=b"test")
        self.cell.fire_and_forget(channel=test_channel, topic=test_topic, targets=[self.cell_name], message=request)

        # Wait for handler to complete before checking result
        self.assertTrue(handler_completed.wait(timeout=5.0), "Handler should have completed")

        # The key assertion: deadlock was detected (handler blocked waiting for external trigger)
        self.assertTrue(deadlock_detected, "Deadlock should be detected (handler timed out waiting)")


class TestBroadcastMultiRequestsToSelf(unittest.TestCase):
    """
    Test using broadcast_multi_requests - the EXACT method used by swarm learning.

    Call path in swarm learning:
    send_learn_task() → WFCommClient.broadcast_and_wait() → engine.send_aux_request()
    → aux_runner.send_aux_request() → cell.broadcast_multi_requests()
    → _send_target_messages() → _send_to_endpoint() → _send_direct_message()
    """

    @classmethod
    def setUpClass(cls):
        """Set up CoreCell for testing, preserving existing ALL_CELLS state."""
        cls._saved_all_cells = dict(CoreCell.ALL_CELLS)

        cls.port = get_open_ports(1)[0]
        cls.cell_name = f"test_cell_broadcast_{cls.port}"
        cls.listening_url = f"tcp://localhost:{cls.port}"

        cls.cell = CoreCell(fqcn=cls.cell_name, root_url=cls.listening_url, secure=False, credentials={})
        cls.cell.start()
        # Wait for cell to be registered in ALL_CELLS (deterministic check)
        for _ in range(50):  # Up to 5 seconds
            if cls.cell_name in CoreCell.ALL_CELLS:
                break
            time.sleep(0.1)
        assert cls.cell_name in CoreCell.ALL_CELLS, "Cell failed to register"

    @classmethod
    def tearDownClass(cls):
        """Clean up the cell and restore ALL_CELLS state."""
        try:
            cls.cell.stop()
        finally:
            CoreCell.ALL_CELLS.pop(cls.cell_name, None)
            for name, cell in cls._saved_all_cells.items():
                if name not in CoreCell.ALL_CELLS:
                    CoreCell.ALL_CELLS[name] = cell

    def test_broadcast_multi_requests_to_self_is_sync(self):
        """
        Test broadcast_multi_requests to self - this is EXACTLY what swarm learning uses.
        """
        handler_thread = None
        handler_completed = threading.Event()

        # Use unique channel/topic for this test
        test_channel = f"aux_broadcast_{id(self)}"
        test_topic = f"learn_task_{id(self)}"

        def request_handler(message: Message):
            nonlocal handler_thread
            handler_thread = threading.current_thread().ident
            handler_completed.set()
            reply = Message()
            reply.set_header(MessageHeaderKey.RETURN_CODE, ReturnCode.OK)
            return reply

        self.cell.register_request_cb(channel=test_channel, topic=test_topic, cb=request_handler)

        sender_thread = threading.current_thread().ident

        request = Message(headers={}, payload=b"learn_task_data")
        target_msgs = {
            self.cell_name: TargetMessage(
                target=self.cell_name, channel=test_channel, topic=test_topic, message=request
            )
        }

        self.cell.broadcast_multi_requests(target_msgs=target_msgs, timeout=5.0, secure=False, optional=False)

        # Wait for handler to complete (should be immediate for sync call)
        self.assertTrue(handler_completed.is_set(), "Handler should have completed")

        # Key assertion: Handler runs on same thread (synchronous)
        self.assertEqual(sender_thread, handler_thread, "Handler should run on SAME thread as sender")

    def test_tensor_wait_simulation_causes_deadlock(self):
        """
        Test with a handler that simulates TensorReceiver.wait_for_tensors().
        """
        tensor_event = threading.Event()
        deadlock_detected = threading.Event()
        handler_completed = threading.Event()

        # Use unique channel/topic for this test
        test_channel = f"tensor_broadcast_{id(self)}"
        test_topic = f"tensor_task_{id(self)}"

        def blocking_handler(message: Message):
            try:
                # Simulate wait_for_tensors() waiting for streaming data
                if not tensor_event.wait(timeout=1.0):
                    deadlock_detected.set()
                reply = Message()
                reply.set_header(MessageHeaderKey.RETURN_CODE, ReturnCode.OK)
                return reply
            finally:
                handler_completed.set()

        self.cell.register_request_cb(channel=test_channel, topic=test_topic, cb=blocking_handler)

        request = Message(headers={}, payload=b"task_with_tensors")
        target_msgs = {
            self.cell_name: TargetMessage(
                target=self.cell_name, channel=test_channel, topic=test_topic, message=request
            )
        }

        self.cell.broadcast_multi_requests(target_msgs=target_msgs, timeout=5.0)

        # Wait for handler to complete before checking result
        self.assertTrue(handler_completed.wait(timeout=5.0), "Handler should have completed")

        self.assertTrue(deadlock_detected.is_set(), "Deadlock should be detected - tensor wait timed out")


class TestSwarmResultSubmissionFix(unittest.TestCase):
    def test_local_submit_when_aggregator_is_self(self):
        class _DummyGatherer:
            def __init__(self, **kwargs):
                self.for_round = kwargs.get("for_round", 0)

        class _DummyEngine:
            def __init__(self):
                self.submit_req_calls = 0

            def send_aux_request(self, **kwargs):
                self.submit_req_calls += 1
                return {"site-1": make_reply(FLReturnCode.OK)}

            def new_context(self):
                return FLContextManager(engine=self, identity_name="site-1", job_id="job").new_context()

        engine = _DummyEngine()
        fl_ctx = FLContextManager(engine=engine, identity_name="site-1", job_id="job").new_context()
        abort_signal = Signal()

        task_data = Shareable()
        task_data.set_header(AppConstants.CURRENT_ROUND, 1)
        task_data.set_header(Constant.AGGREGATOR, "site-1")

        learn_result = make_reply(FLReturnCode.OK)

        ctl = object.__new__(SwarmClientController)
        ctl.me = "site-1"
        ctl.is_trainer = True
        ctl.gatherer = None
        ctl.gatherer_waiter = threading.Event()
        ctl.metric_comparator = object()
        ctl.trainers = ["site-1"]
        ctl.learn_task_timeout = 10
        ctl.min_responses_required = 1
        ctl.wait_time_after_min_resps_received = 0
        ctl.aggregator = object()
        ctl.max_concurrent_submissions = 1
        ctl.request_to_submit_result_max_wait = 10
        ctl.request_to_submit_result_msg_timeout = 1
        ctl.request_to_submit_result_interval = 0
        ctl.request_to_submit_learn_result_task_name = "request_submit"
        ctl.report_learn_result_task_name = "report_result"
        ctl.learn_task_ack_timeout = 5
        ctl.shareable_generator = SimpleNamespace(shareable_to_learnable=lambda _task_data, _ctx: Learnable())
        ctl.get_config_prop = lambda key, default=None: ["site-1"] if key == Constant.CLIENTS else default
        ctl.execute_learn_task = lambda _task_data, _ctx, _abort_signal: learn_result
        ctl.is_task_secure = lambda _ctx: False
        ctl.update_status = lambda **kwargs: None
        ctl.fire_event = lambda *_args, **_kwargs: None
        ctl.log_info = lambda *_args, **_kwargs: None
        ctl.log_debug = lambda *_args, **_kwargs: None
        ctl.log_warning = lambda *_args, **_kwargs: None
        ctl.log_error = lambda *_args, **_kwargs: None
        ctl.broadcast_and_wait = mock.Mock(
            side_effect=AssertionError("broadcast_and_wait must not be called for local result submission")
        )
        ctl._process_learn_result = mock.Mock(return_value=make_reply(FLReturnCode.OK))

        with mock.patch("nvflare.app_common.ccwf.swarm_client_ctl.Gatherer", _DummyGatherer):
            ctl.do_learn_task("train", task_data, fl_ctx, abort_signal)

        ctl.broadcast_and_wait.assert_not_called()
        ctl._process_learn_result.assert_called_once()
        self.assertEqual(engine.submit_req_calls, 1, "submission permission request should still be sent once")

        called_result, called_fl_ctx, called_abort_signal = ctl._process_learn_result.call_args[0]
        self.assertIs(called_result, learn_result)
        self.assertIs(called_abort_signal, abort_signal)
        self.assertIsNot(called_fl_ctx, fl_ctx)
        self.assertEqual(called_fl_ctx.get_peer_context().get_identity_name(), "site-1")


if __name__ == "__main__":
    unittest.main()
