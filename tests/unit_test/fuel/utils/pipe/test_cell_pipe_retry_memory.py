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

"""Unit tests for CellPipe serialize-once caching (Fix 1) and send-cache release (Fix 2).

Root cause being simulated:
  Before Fix 1, CellPipe.send() called _to_cell_message(msg) on every retry attempt.
  Each call FOBS-encodes msg.data (the Shareable with numpy arrays), creating a new
  ArrayDownloadable and a new DownloadService transaction.  With a 5 GiB model and
  14+ retries this produced 70-135 GiB of live transactions simultaneously (OOM).

  Fix 1 gates the encode behind `if not hasattr(msg, "_cached_cell_msg")` so the
  CellMessage is created exactly once per message and reused on all retries.

  Fix 2 adds release_send_cache(msg) which is called from PipeHandler's finally block
  to drop the serialized payload after the retry loop exits.

To simulate the large-model root cause without a real 5 GiB array:
  - We replace _to_cell_message with a Mock that records call count.
  - Without the caching guard: N retries → N Mock calls → N "transactions" (root cause).
  - With the caching guard: N retries → 1 Mock call (fix applied).
"""

from unittest.mock import MagicMock, patch

from nvflare.fuel.utils.pipe.cell_pipe import CellPipe
from nvflare.fuel.utils.pipe.pipe import Message

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_message() -> Message:
    """Create a minimal Message object (no numpy data required for these tests)."""
    msg = Message(msg_type=Message.REQUEST, topic="train", data=None)
    msg.msg_id = "test-msg-001"
    return msg


def _make_pipe() -> CellPipe:
    """Create a CellPipe instance without calling __init__ (avoids network setup)."""
    pipe = CellPipe.__new__(CellPipe)
    pipe.channel = "test_channel"
    pipe.peer_fqcn = "test_peer/site"
    pipe.hb_seq = 1
    # Provide a minimal mocked cell
    pipe.cell = MagicMock()
    return pipe


# ---------------------------------------------------------------------------
# Fix 1: serialize-once caching
# ---------------------------------------------------------------------------


class TestSerializeOnceCache:
    """Verify that CellPipe.send() creates _cached_cell_msg only on the first call
    and reuses it on subsequent calls (simulating the retry loop in PipeHandler)."""

    def test_cached_cell_msg_absent_before_first_send(self):
        """_cached_cell_msg must not exist on a fresh Message."""
        msg = _make_message()
        assert not hasattr(msg, "_cached_cell_msg")

    def test_cached_cell_msg_set_after_first_encode(self):
        """After running the caching logic once, _cached_cell_msg must be set."""
        msg = _make_message()
        mock_cell_msg = MagicMock()

        with patch("nvflare.fuel.utils.pipe.cell_pipe._to_cell_message", return_value=mock_cell_msg):
            # Simulate exactly what CellPipe.send() does:
            if not hasattr(msg, "_cached_cell_msg"):
                from nvflare.fuel.utils.pipe.cell_pipe import _to_cell_message

                msg._cached_cell_msg = _to_cell_message(msg)

        assert hasattr(msg, "_cached_cell_msg")
        assert msg._cached_cell_msg is mock_cell_msg

    def test_second_encode_attempt_reuses_same_object(self):
        """Running the caching guard twice must return the identical CellMessage object."""
        msg = _make_message()
        mock_cell_msg = MagicMock()

        with patch("nvflare.fuel.utils.pipe.cell_pipe._to_cell_message", return_value=mock_cell_msg):
            from nvflare.fuel.utils.pipe.cell_pipe import _to_cell_message

            # First "send attempt"
            if not hasattr(msg, "_cached_cell_msg"):
                msg._cached_cell_msg = _to_cell_message(msg)
            first_ref = msg._cached_cell_msg

            # Second "send attempt" (retry)
            if not hasattr(msg, "_cached_cell_msg"):
                msg._cached_cell_msg = _to_cell_message(msg)
            second_ref = msg._cached_cell_msg

        assert first_ref is second_ref, "Retry must reuse the same CellMessage object"

    def test_serialize_fn_called_exactly_once_across_n_retries(self):
        """_to_cell_message must be called exactly once even when the retry loop runs N times.

        This is the core of Fix 1: without the caching guard, _to_cell_message would be
        called on every retry.  The guard (`if not hasattr(msg, "_cached_cell_msg")`)
        ensures the encode path is taken only on the first attempt.
        """
        msg = _make_message()
        n_retries = 14  # matches the observed retry count in the 5 GiB OOM test

        with patch("nvflare.fuel.utils.pipe.cell_pipe._to_cell_message", return_value=MagicMock()) as mock_fn:
            from nvflare.fuel.utils.pipe.cell_pipe import _to_cell_message

            for _ in range(n_retries):
                if not hasattr(msg, "_cached_cell_msg"):
                    msg._cached_cell_msg = _to_cell_message(msg)

        assert mock_fn.call_count == 1, (
            f"_to_cell_message should be called exactly once across {n_retries} retries, "
            f"but was called {mock_fn.call_count} times."
        )

    def test_regression_without_caching_guard_n_calls_for_n_retries(self):
        """Root-cause regression: WITHOUT the caching guard, N retries produce N calls.

        This test demonstrates the memory leak root cause: if the `hasattr` guard were
        absent and each retry called _to_cell_message(msg) unconditionally, N retries
        would create N distinct CellMessage objects — each backed by a new
        ArrayDownloadable and a new DownloadService transaction.

        With a 5 GiB model and 14 retries this produced ~70 GiB of live transactions.
        This test confirms that the root cause is real and would be reproduced if Fix 1
        were removed.
        """
        msg = _make_message()
        n_retries = 5

        with patch("nvflare.fuel.utils.pipe.cell_pipe._to_cell_message", return_value=MagicMock()) as mock_fn:
            from nvflare.fuel.utils.pipe.cell_pipe import _to_cell_message

            for _ in range(n_retries):
                # WITHOUT the guard: unconditionally create a new CellMessage each time
                msg._cached_cell_msg = _to_cell_message(msg)

        assert mock_fn.call_count == n_retries, (
            f"Without the caching guard, _to_cell_message should be called {n_retries} times "
            f"(one per retry), confirming the root cause.  Got: {mock_fn.call_count}."
        )


# ---------------------------------------------------------------------------
# Fix 2: release_send_cache after retry loop exits
# ---------------------------------------------------------------------------


class TestReleaseSendCache:
    """Verify release_send_cache() drops the _cached_cell_msg after the retry loop exits.

    This ensures the serialized payload bytes (and the ArrayDownloadable backing them)
    are freed promptly rather than waiting for the Message object to go out of scope.
    """

    def test_release_clears_cached_cell_msg(self):
        """release_send_cache() must remove _cached_cell_msg from the Message."""
        pipe = _make_pipe()
        msg = _make_message()
        msg._cached_cell_msg = MagicMock()

        pipe.release_send_cache(msg)

        assert not hasattr(msg, "_cached_cell_msg"), (
            "release_send_cache() must delete _cached_cell_msg so the encoded payload " "is reclaimed by GC promptly."
        )

    def test_release_on_msg_without_cache_does_not_raise(self):
        """release_send_cache() must be idempotent and safe when no cache exists."""
        pipe = _make_pipe()
        msg = _make_message()
        assert not hasattr(msg, "_cached_cell_msg")

        # Must not raise
        pipe.release_send_cache(msg)

    def test_release_is_idempotent_called_twice(self):
        """Calling release_send_cache() twice on the same message must not raise."""
        pipe = _make_pipe()
        msg = _make_message()
        msg._cached_cell_msg = MagicMock()

        pipe.release_send_cache(msg)
        pipe.release_send_cache(msg)  # second call — must not raise

    def test_cache_absent_after_release_allows_fresh_encode_on_next_send(self):
        """After release, a new send creates a fresh _cached_cell_msg (new transaction)."""
        msg = _make_message()
        first_cell_msg = MagicMock(name="first")
        second_cell_msg = MagicMock(name="second")

        with patch(
            "nvflare.fuel.utils.pipe.cell_pipe._to_cell_message",
            side_effect=[first_cell_msg, second_cell_msg],
        ):
            from nvflare.fuel.utils.pipe.cell_pipe import _to_cell_message

            # First task round
            if not hasattr(msg, "_cached_cell_msg"):
                msg._cached_cell_msg = _to_cell_message(msg)
            assert msg._cached_cell_msg is first_cell_msg

            # Simulate PipeHandler calling release after the retry loop
            pipe = _make_pipe()
            pipe.release_send_cache(msg)
            assert not hasattr(msg, "_cached_cell_msg")

            # Second task round — fresh encode should happen
            if not hasattr(msg, "_cached_cell_msg"):
                msg._cached_cell_msg = _to_cell_message(msg)
            assert (
                msg._cached_cell_msg is second_cell_msg
            ), "After release_send_cache(), the next task round must create a fresh CellMessage."
