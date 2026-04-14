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

"""Unit tests for Bug 1 fix: _create_downloader() must NOT subscribe to msg_root.

Root cause: subscribe_to_msg_root fired delete_transaction() when msg_root was
deleted (after blob delivery ACK), but blob_cb fires asynchronously — secondary
tensor downloads were still in flight, causing "no ref found" FATAL_SYSTEM_ERROR.

Fix: Remove subscribe_to_msg_root from _create_downloader() entirely.
Transaction lifecycle is managed solely by _monitor_tx() (download_service.py).

Tests verify:
  1. _create_downloader() does NOT call subscribe_to_msg_root even when msg_root_id
     is present in fobs_ctx — the race condition is eliminated.
  2. _delete_download_tx_on_msg_root method no longer exists in ViaDownloaderDecomposer
     (the timer-based workaround from PR #4263 is rejected).
  3. DOWNLOAD_COMPLETE_CB is still wired correctly (the async completion gating
     from Fix 16 is unaffected by this change).
"""

from unittest.mock import MagicMock, patch

from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.decomposers.via_downloader import ViaDownloaderDecomposer


class _FakeDecomposer(ViaDownloaderDecomposer):
    def __init__(self):
        super().__init__(max_chunk_size=1024 * 1024, config_var_prefix="np_")

    def to_downloadable(self, items, max_chunk_size, fobs_ctx):
        return MagicMock()

    def download(self, from_fqcn, ref_id, per_request_timeout, cell, secure=False, optional=False, abort_signal=None):
        return None, {}

    def get_download_dot(self):
        return 99

    def native_decompose(self, target, manager=None):
        return b""

    def native_recompose(self, data, manager=None):
        return data

    def supported_type(self):
        return object


def _make_fobs_ctx(msg_root_id="test-root-123", cell=None):
    from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey

    msg = MagicMock()

    def _get_header(key, default=None):
        if key == MessageHeaderKey.MSG_ROOT_ID:
            return msg_root_id
        return default  # MSG_ROOT_TTL and others return None

    msg.get_header.side_effect = _get_header
    ctx = {
        fobs.FOBSContextKey.CELL: cell or MagicMock(),
        fobs.FOBSContextKey.MESSAGE: msg,
        fobs.FOBSContextKey.NUM_RECEIVERS: 1,
    }
    return ctx


class TestNoMsgRootSubscription:
    """_create_downloader() must never subscribe to msg_root."""

    def test_does_not_subscribe_to_msg_root_with_msg_root_id(self):
        """Even when msg_root_id is present, subscribe_to_msg_root is NOT called.

        Before the fix, _create_downloader() called subscribe_to_msg_root() which
        triggered delete_transaction() when msg_root was deleted — before async
        blob_cb finished secondary tensor downloads (RC12 Bug 1 race condition).
        """
        decomposer = _FakeDecomposer()
        ctx = _make_fobs_ctx(msg_root_id="some-root-id")

        with (
            patch("nvflare.fuel.utils.fobs.decomposers.via_downloader.ObjectDownloader") as MockOD,
            patch(
                "nvflare.fuel.utils.fobs.decomposers.via_downloader.subscribe_to_msg_root",
                side_effect=AssertionError("subscribe_to_msg_root must NOT be called"),
                create=True,
            ) as mock_sub,
        ):
            MockOD.return_value = MagicMock()
            decomposer._create_downloader(ctx)

        # If we reach here, subscribe_to_msg_root was not called (no AssertionError raised)
        # Also verify explicitly:
        mock_sub.assert_not_called()

    def test_does_not_subscribe_without_msg_root_id(self):
        """Without msg_root_id, subscribe_to_msg_root is also not called (baseline)."""
        decomposer = _FakeDecomposer()
        ctx = {
            fobs.FOBSContextKey.CELL: MagicMock(),
            fobs.FOBSContextKey.NUM_RECEIVERS: 1,
            # No MESSAGE header -> msg_root_id = None
        }

        with (
            patch("nvflare.fuel.utils.fobs.decomposers.via_downloader.ObjectDownloader") as MockOD,
            patch(
                "nvflare.fuel.utils.fobs.decomposers.via_downloader.subscribe_to_msg_root",
                side_effect=AssertionError("subscribe_to_msg_root must NOT be called"),
                create=True,
            ) as mock_sub,
        ):
            MockOD.return_value = MagicMock()
            decomposer._create_downloader(ctx)

        mock_sub.assert_not_called()

    def test_delete_download_tx_on_msg_root_method_removed(self):
        """_delete_download_tx_on_msg_root must NOT exist on ViaDownloaderDecomposer.

        The method was the msg_root deletion callback — removed with the fix.
        """
        decomposer = _FakeDecomposer()
        assert not hasattr(decomposer, "_delete_download_tx_on_msg_root"), (
            "_delete_download_tx_on_msg_root must be removed; "
            "it was the subscribe_to_msg_root callback that caused the ref-table race"
        )

    def test_download_complete_cb_still_wired(self):
        """Removing subscribe_to_msg_root must not affect DOWNLOAD_COMPLETE_CB wiring.

        DOWNLOAD_COMPLETE_CB (Fix 16) is unrelated to msg_root subscription — it
        is wired via fobs_ctx and must continue to function correctly.
        """
        sentinel_cb = MagicMock()
        ctx = {
            fobs.FOBSContextKey.CELL: MagicMock(),
            fobs.FOBSContextKey.NUM_RECEIVERS: 1,
            fobs.FOBSContextKey.DOWNLOAD_COMPLETE_CB: sentinel_cb,
        }

        with patch("nvflare.fuel.utils.fobs.decomposers.via_downloader.ObjectDownloader") as MockOD:
            MockOD.return_value = MagicMock()
            _FakeDecomposer()._create_downloader(ctx)

        _, kwargs = MockOD.call_args
        assert (
            kwargs.get("transaction_done_cb") is sentinel_cb
        ), "DOWNLOAD_COMPLETE_CB must still be wired as transaction_done_cb after Bug 1 fix"
