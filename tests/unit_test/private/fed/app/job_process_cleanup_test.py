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
    )

    assert calls == [
        "admission",
        "streaming",
        "cell",
        ("callbacks", job_process_cleanup._COMMAND_CALLBACK_DRAIN_TIMEOUT),
        "security",
    ]


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


def test_job_process_runtime_continues_after_callback_drain_timeout(monkeypatch):
    calls = []
    monkeypatch.setattr(job_process_cleanup, "shutdown_f3_streaming", lambda: calls.append("streaming"))
    monkeypatch.setattr(job_process_cleanup, "security_close", lambda: calls.append("security"))
    logger = MagicMock()

    job_process_cleanup.shutdown_job_process_runtime(
        stop_command_admission=lambda: calls.append("admission"),
        wait_for_command_callbacks=lambda _timeout: False,
        stop_cell=lambda: calls.append("cell"),
        logger=logger,
    )

    assert calls == ["admission", "streaming", "cell", "security"]
    logger.warning.assert_called_once_with("timed out after 5.0 seconds waiting for command callbacks")
