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

from unittest.mock import MagicMock

import pytest

from nvflare.private.fed.app import job_process_cleanup


def test_job_process_runtime_shutdown_order(monkeypatch):
    calls = []
    monkeypatch.setattr(job_process_cleanup, "shutdown_f3_streaming", lambda: calls.append("streaming"))
    monkeypatch.setattr(job_process_cleanup, "security_close", lambda: calls.append("security"))

    job_process_cleanup.shutdown_job_process_runtime(
        stop_command_admission=lambda: calls.append("admission"),
        wait_for_command_callbacks=lambda timeout: calls.append(("callbacks", timeout)) or True,
        stop_cell=lambda: calls.append("cell"),
        logger=MagicMock(),
        before_streaming_shutdown=lambda: calls.append("archive"),
    )

    assert calls == [
        "admission",
        ("callbacks", 0.0),
        "archive",
        "streaming",
        "cell",
        ("callbacks", job_process_cleanup._COMMAND_CALLBACK_DRAIN_TIMEOUT),
        "security",
    ]


@pytest.mark.parametrize("drained", (False, True))
def test_shutdown_without_publication_stops_transport_before_waiting(monkeypatch, drained):
    calls = []
    monkeypatch.setattr(job_process_cleanup, "shutdown_f3_streaming", lambda: calls.append("streaming"))
    monkeypatch.setattr(job_process_cleanup, "security_close", lambda: calls.append("security"))
    logger = MagicMock()

    def wait_for_callbacks(timeout):
        # A failed client's deferred cleanup has only MPM's short grace period.
        # Transport must already be stopped before any bounded callback wait.
        assert "streaming" in calls and "cell" in calls
        calls.append(("callbacks", timeout))
        return drained

    mark_incomplete = MagicMock()
    job_process_cleanup.shutdown_job_process_runtime(
        stop_command_admission=lambda: calls.append("admission"),
        wait_for_command_callbacks=wait_for_callbacks,
        stop_cell=lambda: calls.append("cell"),
        logger=logger,
        mark_callback_drain_incomplete=mark_incomplete,
    )

    assert calls == [
        "admission",
        "streaming",
        "cell",
        ("callbacks", job_process_cleanup._COMMAND_CALLBACK_DRAIN_TIMEOUT),
        "security",
    ]
    mark_incomplete.assert_not_called()
    assert logger.warning.call_count == (0 if drained else 1)


def test_shutdown_without_publication_continues_after_post_stop_wait_error(monkeypatch):
    calls = []
    monkeypatch.setattr(job_process_cleanup, "shutdown_f3_streaming", lambda: calls.append("streaming"))
    monkeypatch.setattr(job_process_cleanup, "security_close", lambda: calls.append("security"))

    def fail_wait(_timeout):
        calls.append("callbacks")
        raise RuntimeError("callback failure")

    logger = MagicMock()
    job_process_cleanup.shutdown_job_process_runtime(
        stop_command_admission=None,
        wait_for_command_callbacks=fail_wait,
        stop_cell=lambda: calls.append("cell"),
        logger=logger,
    )

    assert calls == ["streaming", "cell", "callbacks", "security"]
    logger.warning.assert_called_once()
    assert "before resource-statistics publication" not in logger.warning.call_args.args[0]


def test_job_process_runtime_attempts_every_stage_after_failure(monkeypatch):
    calls = []

    def fail_streaming():
        calls.append("streaming")
        raise RuntimeError("failed")

    monkeypatch.setattr(job_process_cleanup, "shutdown_f3_streaming", fail_streaming)
    monkeypatch.setattr(job_process_cleanup, "security_close", lambda: calls.append("security"))
    logger = MagicMock()

    job_process_cleanup.shutdown_job_process_runtime(
        stop_command_admission=lambda: calls.append("admission"),
        wait_for_command_callbacks=lambda _timeout: True,
        stop_cell=lambda: calls.append("cell"),
        logger=logger,
    )

    assert calls == ["admission", "streaming", "cell", "security"]
    logger.warning.assert_called_once()


@pytest.mark.parametrize("drained_after_stop", (False, True))
def test_publication_marks_callback_coverage_without_waiting(monkeypatch, drained_after_stop):
    calls = []
    drain_results = iter((False, drained_after_stop))
    monkeypatch.setattr(job_process_cleanup, "shutdown_f3_streaming", lambda: calls.append("streaming"))
    monkeypatch.setattr(job_process_cleanup, "security_close", lambda: calls.append("security"))
    logger = MagicMock()

    job_process_cleanup.shutdown_job_process_runtime(
        stop_command_admission=lambda: calls.append("admission"),
        wait_for_command_callbacks=lambda timeout: calls.append(("callbacks", timeout)) or next(drain_results),
        stop_cell=lambda: calls.append("cell"),
        logger=logger,
        before_streaming_shutdown=lambda: calls.append("archive"),
        mark_callback_drain_incomplete=lambda: calls.append("mark-incomplete"),
    )

    assert calls == [
        "admission",
        ("callbacks", 0.0),
        "mark-incomplete",
        "archive",
        "streaming",
        "cell",
        ("callbacks", job_process_cleanup._COMMAND_CALLBACK_DRAIN_TIMEOUT),
        "security",
    ]
    if drained_after_stop:
        logger.warning.assert_not_called()
    else:
        logger.warning.assert_called_once_with("timed out after 5.0 seconds waiting for command callbacks")


def test_publication_marks_incomplete_after_nonblocking_coverage_check_error(monkeypatch):
    calls = []
    attempts = 0
    monkeypatch.setattr(job_process_cleanup, "shutdown_f3_streaming", lambda: calls.append("streaming"))
    monkeypatch.setattr(job_process_cleanup, "security_close", lambda: calls.append("security"))
    logger = MagicMock()

    def wait_for_callbacks(timeout):
        nonlocal attempts
        attempts += 1
        calls.append(("callbacks", timeout))
        if attempts == 1:
            raise RuntimeError("still running")
        return True

    job_process_cleanup.shutdown_job_process_runtime(
        stop_command_admission=lambda: calls.append("admission"),
        wait_for_command_callbacks=wait_for_callbacks,
        stop_cell=lambda: calls.append("cell"),
        logger=logger,
        before_streaming_shutdown=lambda: calls.append("archive"),
        mark_callback_drain_incomplete=lambda: calls.append("mark-incomplete"),
    )

    assert calls == [
        "admission",
        ("callbacks", 0.0),
        "mark-incomplete",
        "archive",
        "streaming",
        "cell",
        ("callbacks", job_process_cleanup._COMMAND_CALLBACK_DRAIN_TIMEOUT),
        "security",
    ]
    logger.warning.assert_not_called()


def test_job_process_runtime_tears_down_after_archive_failure(monkeypatch):
    calls = []
    monkeypatch.setattr(job_process_cleanup, "shutdown_f3_streaming", lambda: calls.append("streaming"))
    monkeypatch.setattr(job_process_cleanup, "security_close", lambda: calls.append("security"))

    def fail_archive():
        calls.append("archive")
        raise RuntimeError("results upload failed")

    with pytest.raises(RuntimeError, match="results upload failed"):
        job_process_cleanup.shutdown_job_process_runtime(
            stop_command_admission=lambda: calls.append("admission"),
            wait_for_command_callbacks=lambda timeout: calls.append(("callbacks", timeout)) or True,
            stop_cell=lambda: calls.append("cell"),
            logger=MagicMock(),
            before_streaming_shutdown=fail_archive,
        )

    assert calls == [
        "admission",
        ("callbacks", 0.0),
        "archive",
        "streaming",
        "cell",
        ("callbacks", job_process_cleanup._COMMAND_CALLBACK_DRAIN_TIMEOUT),
        "security",
    ]


def test_publication_without_callback_tracker_does_not_mark_a_gap(monkeypatch):
    calls = []
    monkeypatch.setattr(job_process_cleanup, "shutdown_f3_streaming", lambda: calls.append("streaming"))
    monkeypatch.setattr(job_process_cleanup, "security_close", lambda: calls.append("security"))
    mark_incomplete = MagicMock()

    job_process_cleanup.shutdown_job_process_runtime(
        stop_command_admission=lambda: calls.append("admission"),
        wait_for_command_callbacks=None,
        stop_cell=lambda: calls.append("cell"),
        logger=MagicMock(),
        before_streaming_shutdown=lambda: calls.append("archive"),
        mark_callback_drain_incomplete=mark_incomplete,
    )

    assert calls == ["admission", "archive", "streaming", "cell", "security"]
    mark_incomplete.assert_not_called()


def test_failed_gap_marker_does_not_prevent_publication_or_cleanup(monkeypatch):
    calls = []
    monkeypatch.setattr(job_process_cleanup, "shutdown_f3_streaming", lambda: calls.append("streaming"))
    monkeypatch.setattr(job_process_cleanup, "security_close", lambda: calls.append("security"))
    logger = MagicMock()

    def mark_incomplete():
        calls.append("mark-incomplete")
        raise RuntimeError("optional coverage marker failed")

    job_process_cleanup.shutdown_job_process_runtime(
        stop_command_admission=lambda: calls.append("admission"),
        wait_for_command_callbacks=lambda timeout: calls.append(("callbacks", timeout)) or timeout > 0,
        stop_cell=lambda: calls.append("cell"),
        logger=logger,
        before_streaming_shutdown=lambda: calls.append("archive"),
        mark_callback_drain_incomplete=mark_incomplete,
    )

    assert calls == [
        "admission",
        ("callbacks", 0.0),
        "mark-incomplete",
        "archive",
        "streaming",
        "cell",
        ("callbacks", job_process_cleanup._COMMAND_CALLBACK_DRAIN_TIMEOUT),
        "security",
    ]
    logger.warning.assert_called_once()
