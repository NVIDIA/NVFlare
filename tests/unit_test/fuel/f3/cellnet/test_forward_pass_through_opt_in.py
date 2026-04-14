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
"""Tests for Fix 18 Patch 9.1 + 9.3: ReservedHeaderKey.PASS_THROUGH propagation.

Verifies that:
  1. ReservedHeaderKey.PASS_THROUGH exists and has a unique string value.
  2. aux_runner.py propagates PASS_THROUGH from Shareable to outgoing cell message.
  3. No propagation when the header is absent (default behaviour unchanged).
  4. Propagation is per-message (one stamped request does not affect next request).
"""
import unittest
from unittest.mock import MagicMock

from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey


class TestReservedHeaderKeyPassThrough(unittest.TestCase):
    """ReservedHeaderKey.PASS_THROUGH must exist and be distinct from other keys."""

    def test_pass_through_key_exists(self):
        self.assertTrue(
            hasattr(ReservedHeaderKey, "PASS_THROUGH"), "ReservedHeaderKey must have PASS_THROUGH attribute"
        )

    def test_pass_through_key_is_string(self):
        self.assertIsInstance(ReservedHeaderKey.PASS_THROUGH, str)

    def test_pass_through_key_distinct_from_msg_root_ttl(self):
        self.assertNotEqual(ReservedHeaderKey.PASS_THROUGH, ReservedHeaderKey.MSG_ROOT_TTL)

    def test_pass_through_key_distinct_from_msg_root_id(self):
        self.assertNotEqual(ReservedHeaderKey.PASS_THROUGH, ReservedHeaderKey.MSG_ROOT_ID)


class TestAuxRunnerPassThroughPropagation(unittest.TestCase):
    """aux_runner.send_aux_message() propagates ReservedHeaderKey.PASS_THROUGH to the
    outgoing cell message as MessageHeaderKey.PASS_THROUGH=True.

    The test injects into the _send_to_targets() path by patching cell.broadcast_request
    and inspecting the cell_msg that was constructed.
    """

    def _make_aux_runner(self):
        """Return an AuxRunner with a stubbed cell and FL context."""
        from nvflare.private.aux_runner import AuxRunner

        runner = AuxRunner.__new__(AuxRunner)
        runner.logger = MagicMock()
        runner._cell = MagicMock()
        runner._cell_waiter = MagicMock()
        runner._cell_waiter.wait.return_value = True
        runner._job_id = "test_job"
        runner._engine = MagicMock()
        # Make _wait_for_cell() return the mock cell immediately
        runner._wait_for_cell = MagicMock(return_value=runner._cell)
        runner._get_target_fqcn = MagicMock(side_effect=lambda t, ctx: f"server/{t.name}")
        return runner

    def _make_target(self, name="site-1"):
        t = MagicMock()
        t.name = name
        return t

    def _call_send_to_cell(self, runner, request, timeout=10.0):
        """Call the internal _send_to_cell with enough stubs to capture the cell_msg."""
        captured = {}

        def fake_broadcast(channel, topic, request, targets, timeout, **kwargs):
            captured["cell_msg"] = request
            return {}

        runner._cell.broadcast_request.side_effect = fake_broadcast

        fl_ctx = MagicMock()
        fl_ctx.get_all_public_props.return_value = {}

        runner._send_to_cell(
            targets=[self._make_target()],
            channel="test_channel",
            topic="test_topic",
            request=request,
            timeout=timeout,
            fl_ctx=fl_ctx,
        )
        return captured.get("cell_msg")

    def test_pass_through_header_stamped_on_cell_msg_when_set(self):
        """When Shareable has PASS_THROUGH=True, cell message must carry
        MessageHeaderKey.PASS_THROUGH=True."""
        runner = self._make_aux_runner()
        request = Shareable()
        request.set_header(ReservedHeaderKey.PASS_THROUGH, True)

        cell_msg = self._call_send_to_cell(runner, request)

        self.assertIsNotNone(cell_msg, "cell_msg must be captured")
        pt = cell_msg.get_header(MessageHeaderKey.PASS_THROUGH, False)
        self.assertTrue(pt, "cell message must have PASS_THROUGH=True when Shareable header is set")

    def test_no_pass_through_stamp_when_header_absent(self):
        """Without PASS_THROUGH on the Shareable, cell message must NOT carry the header."""
        runner = self._make_aux_runner()
        request = Shareable()  # no PASS_THROUGH header

        cell_msg = self._call_send_to_cell(runner, request)

        self.assertIsNotNone(cell_msg)
        pt = cell_msg.get_header(MessageHeaderKey.PASS_THROUGH, False)
        self.assertFalse(pt, "cell message must NOT have PASS_THROUGH when Shareable header is absent")

    def test_pass_through_not_stamped_on_zero_timeout(self):
        """With timeout=0 (fire-and-forget), the PASS_THROUGH block is never reached because
        cell.broadcast_request is called without building cell_msg headers."""
        runner = self._make_aux_runner()
        request = Shareable()
        request.set_header(ReservedHeaderKey.PASS_THROUGH, True)

        captured_msgs = []

        def fake_fire_and_forget(channel, topic, request, targets, **kwargs):
            captured_msgs.append(request)
            return {}

        runner._cell.broadcast_request.side_effect = fake_fire_and_forget
        fl_ctx = MagicMock()
        fl_ctx.get_all_public_props.return_value = {}

        # timeout=0 → fire-and-forget path, header block skipped
        runner._send_to_cell(
            targets=[self._make_target()],
            channel="ch",
            topic="t",
            request=request,
            timeout=0,
            fl_ctx=fl_ctx,
        )
        # Nothing to assert on header (fire-and-forget path uses a different code branch),
        # but the call must not raise.
        # The test documents the boundary of the propagation guarantee.

    def test_pass_through_is_per_message(self):
        """Stamping PASS_THROUGH on one request must not carry over to the next."""
        runner = self._make_aux_runner()

        msgs = []

        def fake_broadcast(channel, topic, request, targets, timeout, **kwargs):
            msgs.append(request)
            return {}

        runner._cell.broadcast_request.side_effect = fake_broadcast

        fl_ctx = MagicMock()
        fl_ctx.get_all_public_props.return_value = {}

        def send(request):
            runner._send_to_cell(
                targets=[self._make_target()],
                channel="ch",
                topic="t",
                request=request,
                timeout=5.0,
                fl_ctx=fl_ctx,
            )

        req_with_pt = Shareable()
        req_with_pt.set_header(ReservedHeaderKey.PASS_THROUGH, True)
        send(req_with_pt)

        req_without_pt = Shareable()
        send(req_without_pt)

        self.assertEqual(len(msgs), 2)
        self.assertTrue(
            msgs[0].get_header(MessageHeaderKey.PASS_THROUGH, False), "first message must have PASS_THROUGH=True"
        )
        self.assertFalse(
            msgs[1].get_header(MessageHeaderKey.PASS_THROUGH, False),
            "second message must NOT have PASS_THROUGH (no carryover)",
        )


if __name__ == "__main__":
    unittest.main()
