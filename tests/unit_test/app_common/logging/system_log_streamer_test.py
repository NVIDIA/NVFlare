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

import json
import os
from unittest.mock import Mock, patch

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import (
    FLContextKey,
    JobConstants,
    ProcessType,
    ReservedKey,
    StreamCtxKey,
    WorkspaceConstants,
)
from nvflare.apis.fl_context import FLContext
from nvflare.apis.workspace import Workspace
from nvflare.app_common.logging.constants import LIVE_LOG_TOPIC, Channels
from nvflare.app_common.logging.system_log_streamer import SystemLogStreamer


def _allow_streaming(value: bool):
    """Patch is_log_streaming_allowed inside system_log_streamer to return ``value``."""
    return patch(
        "nvflare.app_common.logging.system_log_streamer.is_log_streaming_allowed",
        return_value=value,
    )


class _ImmediateThread:
    def __init__(self, target, args=(), kwargs=None, daemon=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)


def _make_fl_ctx(tmp_path):
    os.makedirs(tmp_path / "startup", exist_ok=True)
    os.makedirs(tmp_path / "local", exist_ok=True)
    fl_ctx = FLContext()
    fl_ctx.put(key=ReservedKey.IDENTITY_NAME, value="site-1", private=True, sticky=False)
    fl_ctx.put(key=ReservedKey.WORKSPACE_ROOT, value=str(tmp_path), private=True, sticky=False)
    fl_ctx.put(key=ReservedKey.CURRENT_JOB_ID, value="job-1", private=True, sticky=False)
    fl_ctx.put(key=ReservedKey.PROCESS_TYPE, value=ProcessType.CLIENT_PARENT, private=True, sticky=False)
    engine = Mock()
    stream_fl_ctx = FLContext()
    engine.new_context.return_value = stream_fl_ctx
    fl_ctx.put(key=ReservedKey.ENGINE, value=engine, private=True, sticky=False)
    return fl_ctx


def test_system_log_streamer_uploads_completed_error_log_snapshot(tmp_path):
    os.makedirs(tmp_path / "startup", exist_ok=True)
    os.makedirs(tmp_path / "local", exist_ok=True)
    workspace = Workspace(root_dir=str(tmp_path), site_name="site-1")
    error_log_path = workspace.get_app_error_log_file_path("job-1")
    os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
    with open(error_log_path, "w") as f:
        f.write("boom\n")

    fl_ctx = _make_fl_ctx(tmp_path)
    streamer = SystemLogStreamer(log_file_name=WorkspaceConstants.ERROR_LOG_FILE_NAME)

    with (
        _allow_streaming(True),
        patch("nvflare.app_common.logging.system_log_streamer.threading.Thread", _ImmediateThread),
        patch("nvflare.app_common.logging.system_log_streamer.LogStreamer.stream_log") as stream_log,
    ):
        streamer._on_job_completed(EventType.JOB_COMPLETED, fl_ctx)

    stream_log.assert_called_once()
    kwargs = stream_log.call_args.kwargs
    assert kwargs["channel"] == Channels.LOG_STREAMING_CHANNEL
    assert kwargs["topic"] == LIVE_LOG_TOPIC
    assert kwargs["stream_ctx"] == {
        StreamCtxKey.CLIENT_NAME: "site-1",
        StreamCtxKey.JOB_ID: "job-1",
    }
    assert kwargs["targets"] == ["server"]
    assert kwargs["file_name"] == str(error_log_path)
    assert kwargs["fl_ctx"] is fl_ctx.get_engine().new_context.return_value
    assert kwargs["stop_event"].is_set()


def test_completed_upload_seeds_run_num_and_identity_on_fresh_context(tmp_path):
    """Regression for FLARE-2921: CLIENT_PARENT's engine.new_context() returns a
    fresh context with no per-job RUN_NUM / IDENTITY_NAME, so without explicit
    seeding the receiver gets back peer_ctx.get_job_id()=='' and routes the
    upload under the literal 'unknown' job id (StorageException). Verify that
    the upload thread's public context carries the real job_id and client_name."""
    os.makedirs(tmp_path / "startup", exist_ok=True)
    os.makedirs(tmp_path / "local", exist_ok=True)
    workspace = Workspace(root_dir=str(tmp_path), site_name="site-1")
    error_log_path = workspace.get_app_error_log_file_path("job-1")
    os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
    with open(error_log_path, "w") as f:
        f.write("boom\n")

    fl_ctx = _make_fl_ctx(tmp_path)
    stream_fl_ctx = fl_ctx.get_engine().new_context.return_value
    # Sanity: the fresh context starts with no job_id / identity_name, mirroring
    # the production CLIENT_PARENT engine.new_context() behavior.
    assert stream_fl_ctx.get_job_id(default="") == ""
    assert stream_fl_ctx.get_identity_name(default="") == ""
    assert ReservedKey.RUN_NUM not in stream_fl_ctx.get_all_public_props()
    assert ReservedKey.IDENTITY_NAME not in stream_fl_ctx.get_all_public_props()

    streamer = SystemLogStreamer(log_file_name=WorkspaceConstants.ERROR_LOG_FILE_NAME)
    with (
        _allow_streaming(True),
        patch("nvflare.app_common.logging.system_log_streamer.threading.Thread", _ImmediateThread),
        patch("nvflare.app_common.logging.system_log_streamer.LogStreamer.stream_log"),
    ):
        streamer._on_job_completed(EventType.JOB_COMPLETED, fl_ctx)

    assert stream_fl_ctx.get_job_id() == "job-1"
    assert stream_fl_ctx.get_identity_name() == "site-1"

    # AuxRunner forwards only public props. Rebuild the receiver-side peer ctx
    # from that public view to verify JobLogReceiver sees the real identity.
    peer_ctx = FLContext()
    peer_ctx.set_public_props(stream_fl_ctx.get_all_public_props())
    assert peer_ctx.get_job_id() == "job-1"
    assert peer_ctx.get_identity_name() == "site-1"


def test_system_log_streamer_skips_completed_upload_for_non_error_logs(tmp_path):
    fl_ctx = _make_fl_ctx(tmp_path)
    streamer = SystemLogStreamer(log_file_name=WorkspaceConstants.LOG_FILE_NAME)

    with patch("nvflare.app_common.logging.system_log_streamer.LogStreamer.stream_log") as stream_log:
        streamer._on_job_completed(EventType.JOB_COMPLETED, fl_ctx)

    stream_log.assert_not_called()


def test_on_job_completed_skips_upload_when_streaming_disabled(tmp_path):
    os.makedirs(tmp_path / "startup", exist_ok=True)
    os.makedirs(tmp_path / "local", exist_ok=True)
    workspace = Workspace(root_dir=str(tmp_path), site_name="site-1")
    error_log_path = workspace.get_app_error_log_file_path("job-1")
    os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
    with open(error_log_path, "w") as f:
        f.write("boom\n")

    fl_ctx = _make_fl_ctx(tmp_path)
    streamer = SystemLogStreamer(log_file_name=WorkspaceConstants.ERROR_LOG_FILE_NAME)

    with (
        _allow_streaming(False),
        patch("nvflare.app_common.logging.system_log_streamer.LogStreamer.stream_log") as stream_log,
    ):
        streamer._on_job_completed(EventType.JOB_COMPLETED, fl_ctx)

    stream_log.assert_not_called()


def _make_launch_fl_ctx(tmp_path, job_id="job-1", client_name="site-1"):
    fl_ctx = FLContext()
    fl_ctx.put(key=ReservedKey.IDENTITY_NAME, value=client_name, private=True, sticky=False)
    fl_ctx.put(key=FLContextKey.WORKSPACE_ROOT, value=str(tmp_path), private=True, sticky=False)
    fl_ctx.put(key=FLContextKey.JOB_META, value={"job_id": job_id}, private=True, sticky=False)
    return fl_ctx


def _write_job_config(tmp_path, components, job_id="job-1", client_name="site-1") -> str:
    os.makedirs(tmp_path / "startup", exist_ok=True)
    os.makedirs(tmp_path / "local", exist_ok=True)
    workspace = Workspace(root_dir=str(tmp_path), site_name=client_name)
    cfg_dir = workspace.get_app_config_dir(job_id)
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = os.path.join(cfg_dir, JobConstants.CLIENT_JOB_CONFIG)
    with open(cfg_path, "w") as f:
        json.dump({"format_version": 2, "components": components}, f)
    return cfg_path


def test_before_job_launch_strips_existing_streamer_when_disabled(tmp_path):
    cfg_path = _write_job_config(
        tmp_path,
        [
            {"id": "ls", "path": "nvflare.app_common.logging.job_log_streamer.JobLogStreamer"},
            {"id": "other", "path": "nvflare.something.Else"},
        ],
    )
    fl_ctx = _make_launch_fl_ctx(tmp_path)
    streamer = SystemLogStreamer()

    with _allow_streaming(False):
        streamer._on_before_job_launch(EventType.BEFORE_JOB_LAUNCH, fl_ctx)

    with open(cfg_path) as f:
        cfg = json.load(f)
    paths = [c["path"] for c in cfg["components"]]
    assert "nvflare.app_common.logging.job_log_streamer.JobLogStreamer" not in paths
    assert "nvflare.something.Else" in paths


def test_before_job_launch_skips_injection_when_disabled(tmp_path):
    cfg_path = _write_job_config(tmp_path, [{"id": "other", "path": "nvflare.something.Else"}])
    fl_ctx = _make_launch_fl_ctx(tmp_path)
    streamer = SystemLogStreamer()

    with _allow_streaming(False):
        streamer._on_before_job_launch(EventType.BEFORE_JOB_LAUNCH, fl_ctx)

    with open(cfg_path) as f:
        cfg = json.load(f)
    paths = [c["path"] for c in cfg["components"]]
    assert all("JobLogStreamer" not in p for p in paths)


def test_before_job_launch_injects_when_enabled(tmp_path):
    cfg_path = _write_job_config(tmp_path, [])
    fl_ctx = _make_launch_fl_ctx(tmp_path)
    streamer = SystemLogStreamer()

    with _allow_streaming(True):
        streamer._on_before_job_launch(EventType.BEFORE_JOB_LAUNCH, fl_ctx)

    with open(cfg_path) as f:
        cfg = json.load(f)
    paths = [c["path"] for c in cfg["components"]]
    assert "nvflare.app_common.logging.job_log_streamer.JobLogStreamer" in paths
