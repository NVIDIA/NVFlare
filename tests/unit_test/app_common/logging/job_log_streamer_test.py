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

from unittest.mock import Mock, patch

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ProcessType, ReservedKey, SiteType
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.logging.job_log_streamer import JobLogStreamer


def _allow(value: bool):
    return patch(
        "nvflare.app_common.logging.job_log_streamer.is_log_streaming_allowed",
        return_value=value,
    )


def test_handlers_registered_unconditionally():
    streamer = JobLogStreamer()
    handlers = streamer.get_event_handlers()
    assert EventType.START_RUN in handlers
    assert EventType.ABOUT_TO_END_RUN in handlers
    assert EventType.END_RUN in handlers


def test_on_job_started_is_noop_when_streaming_not_allowed():
    streamer = JobLogStreamer()
    fl_ctx = FLContext()
    with _allow(False), patch.object(streamer, "_find_log_path") as find_log_path:
        streamer._on_job_started(EventType.START_RUN, fl_ctx)
    find_log_path.assert_not_called()
    assert streamer._stream_thread is None


def test_on_job_started_starts_streaming_when_allowed(tmp_path):
    streamer = JobLogStreamer()
    fl_ctx = FLContext()
    with (
        _allow(True),
        patch.object(streamer, "_find_log_path", return_value=str(tmp_path / "log.txt")),
        patch("nvflare.app_common.logging.job_log_streamer.threading.Thread") as Thread,
        patch.object(fl_ctx, "get_engine"),
    ):
        streamer._on_job_started(EventType.START_RUN, fl_ctx)
    Thread.assert_called_once()


def test_system_injected_streamer_targets_parent_server(tmp_path):
    streamer = JobLogStreamer(target_parent_server=True)
    stream_fl_ctx = FLContext()
    stream_fl_ctx.put(key=ReservedKey.PROCESS_TYPE, value=ProcessType.CLIENT_JOB, private=True, sticky=False)
    engine = Mock()
    engine.new_context.return_value = stream_fl_ctx
    fl_ctx = FLContext()
    fl_ctx.put(key=ReservedKey.ENGINE, value=engine, private=True, sticky=False)
    fl_ctx.put(key=ReservedKey.RUN_NUM, value="job-1", private=False, sticky=False)
    fl_ctx.put(key=ReservedKey.IDENTITY_NAME, value="site-1", private=False, sticky=False)

    with (
        _allow(True),
        patch.object(streamer, "_find_log_path", return_value=str(tmp_path / "log.txt")),
        patch("nvflare.app_common.logging.job_log_streamer.threading.Thread") as Thread,
    ):
        streamer._on_job_started(EventType.START_RUN, fl_ctx)

    Thread.assert_called_once()
    assert stream_fl_ctx.get_process_type() == ProcessType.CLIENT_JOB
    assert Thread.call_args.kwargs["args"][4] == [SiteType.SERVER_PARENT]
