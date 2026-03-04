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

"""Unit tests for Bug 1 fix: deferred download transaction deletion.

Bug 1 — P2P broadcast download ref race condition:
    Root Cause: When a blob envelope is delivered, blob_cb is queued
    asynchronously on callback_thread_pool.  Meanwhile, the message root
    can be deleted, triggering _delete_download_tx_on_msg_root() which
    calls downloader.delete_transaction() immediately — removing refs
    from _ref_table.  When blob_cb later initiates secondary tensor
    downloads, _handle_download() returns "no ref found".

    Fix: _delete_download_tx_on_msg_root() now defers delete_transaction()
    by 30 seconds using threading.Timer, giving blob_cb time to complete
    secondary downloads.

CONTRACT:
- _delete_download_tx_on_msg_root() must NOT call delete_transaction() immediately
- _delete_download_tx_on_msg_root() must schedule a 30s deferred call via threading.Timer
- The Timer thread must be a daemon thread (so it doesn't block process exit)
- _deferred_delete_download_tx() must call downloader.delete_transaction()
"""

from unittest.mock import MagicMock, patch

from nvflare.fuel.utils.fobs.decomposers.via_downloader import ViaDownloaderDecomposer

# ---------------------------------------------------------------------------
# Minimal concrete subclass — only deletion methods are under test
# ---------------------------------------------------------------------------


class _FakeDecomposer(ViaDownloaderDecomposer):
    """Concrete stub of ViaDownloaderDecomposer for testing deferred deletion."""

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


# ---------------------------------------------------------------------------
# Bug 1: _delete_download_tx_on_msg_root deferred deletion
# ---------------------------------------------------------------------------


class TestDeferredDownloadTxDeletion:
    """_delete_download_tx_on_msg_root() must defer delete_transaction() by 30s."""

    def test_does_not_call_delete_transaction_immediately(self):
        """delete_transaction() must NOT be called synchronously."""
        decomposer = _FakeDecomposer()
        mock_downloader = MagicMock()

        with patch("nvflare.fuel.utils.fobs.decomposers.via_downloader.threading.Timer") as MockTimer:
            mock_timer_instance = MagicMock()
            MockTimer.return_value = mock_timer_instance

            decomposer._delete_download_tx_on_msg_root("msg-root-1", mock_downloader)

        mock_downloader.delete_transaction.assert_not_called()

    def test_schedules_timer_with_30s_delay(self):
        """A threading.Timer with 30.0s delay must be created."""
        decomposer = _FakeDecomposer()
        mock_downloader = MagicMock()

        with patch("nvflare.fuel.utils.fobs.decomposers.via_downloader.threading.Timer") as MockTimer:
            mock_timer_instance = MagicMock()
            MockTimer.return_value = mock_timer_instance

            decomposer._delete_download_tx_on_msg_root("msg-root-1", mock_downloader)

        MockTimer.assert_called_once()
        args, kwargs = MockTimer.call_args
        assert args[0] == 30.0, f"Timer delay must be 30.0s, got {args[0]}"

    def test_timer_is_daemon(self):
        """The Timer thread must be set as a daemon so it does not block process exit."""
        decomposer = _FakeDecomposer()
        mock_downloader = MagicMock()

        with patch("nvflare.fuel.utils.fobs.decomposers.via_downloader.threading.Timer") as MockTimer:
            mock_timer_instance = MagicMock()
            MockTimer.return_value = mock_timer_instance

            decomposer._delete_download_tx_on_msg_root("msg-root-1", mock_downloader)

        assert mock_timer_instance.daemon is True, "Timer must be a daemon thread"

    def test_timer_is_started(self):
        """The Timer must be started after creation."""
        decomposer = _FakeDecomposer()
        mock_downloader = MagicMock()

        with patch("nvflare.fuel.utils.fobs.decomposers.via_downloader.threading.Timer") as MockTimer:
            mock_timer_instance = MagicMock()
            MockTimer.return_value = mock_timer_instance

            decomposer._delete_download_tx_on_msg_root("msg-root-1", mock_downloader)

        mock_timer_instance.start.assert_called_once()

    def test_timer_target_is_deferred_delete(self):
        """Timer target must be _deferred_delete_download_tx with correct args."""
        decomposer = _FakeDecomposer()
        mock_downloader = MagicMock()

        with patch("nvflare.fuel.utils.fobs.decomposers.via_downloader.threading.Timer") as MockTimer:
            mock_timer_instance = MagicMock()
            MockTimer.return_value = mock_timer_instance

            decomposer._delete_download_tx_on_msg_root("msg-root-1", mock_downloader)

        args, kwargs = MockTimer.call_args
        # args = (30.0, target_fn, args=[msg_root_id, downloader])
        assert args[1] == decomposer._deferred_delete_download_tx
        assert kwargs.get("args") == ["msg-root-1", mock_downloader] or args[2] == ["msg-root-1", mock_downloader]

    def test_deferred_delete_calls_delete_transaction(self):
        """_deferred_delete_download_tx() must call downloader.delete_transaction()."""
        decomposer = _FakeDecomposer()
        mock_downloader = MagicMock()

        decomposer._deferred_delete_download_tx("msg-root-1", mock_downloader)

        mock_downloader.delete_transaction.assert_called_once()
