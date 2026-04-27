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

import pytest

from nvflare.apis.fl_constant import ReservedKey, ReturnCode, StreamCtxKey, WorkspaceConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.storage import DataTypes, StorageSpec
from nvflare.apis.streaming import StreamContextKey
from nvflare.app_common.logging.job_log_receiver import JobLogReceiver
from nvflare.app_common.streamers.log_streamer import KEY_FILE_NAME


def _allowed_client(name: str = "trusted_client", allow: bool = True):
    client = Mock()
    client.get_site_config.return_value = {"allow_log_streaming": allow} if allow is not None else None
    return client


@pytest.mark.parametrize(
    "file_name,expected_data_type",
    [
        (WorkspaceConstants.ERROR_LOG_FILE_NAME, DataTypes.ERRORLOG.value),
        (WorkspaceConstants.LOG_FILE_NAME, f"{DataTypes.LOG.value}_{WorkspaceConstants.LOG_FILE_NAME}"),
        ("metrics.out", f"{DataTypes.LOG.value}_metrics.out"),
    ],
)
def test_job_log_receiver_uses_trusted_peer_identity_for_storage(tmp_path, file_name, expected_data_type):
    receiver = JobLogReceiver(dest_dir=str(tmp_path))

    peer_ctx = FLContext()
    peer_ctx.put(key=ReservedKey.IDENTITY_NAME, value="trusted_client", private=True, sticky=False)
    peer_ctx.put(key=ReservedKey.RUN_NUM, value="trusted_job", private=True, sticky=False)

    job_manager = Mock()
    engine = Mock()
    engine.get_component.return_value = job_manager
    engine.get_client_from_name.return_value = _allowed_client()

    fl_ctx = FLContext()
    fl_ctx.put(key=ReservedKey.IDENTITY_NAME, value="server", private=True, sticky=False)
    fl_ctx.put(key=ReservedKey.RUN_NUM, value="server_job", private=True, sticky=False)
    fl_ctx.put(key=ReservedKey.ENGINE, value=engine, private=True, sticky=False)
    fl_ctx.set_peer_context(peer_ctx)

    stream_ctx = {
        StreamCtxKey.CLIENT_NAME: "../../forged_client",
        StreamCtxKey.JOB_ID: "../../forged_job",
        KEY_FILE_NAME: file_name,
    }
    stream_ctx[StreamContextKey.RC] = ReturnCode.OK

    receiver._on_chunk_received(b"log line\n", stream_ctx, fl_ctx)
    receiver._on_stream_done(stream_ctx, fl_ctx)

    expected_path = tmp_path / "trusted_job" / "trusted_client" / file_name
    assert expected_path.exists()
    assert expected_path.read_bytes() == b"log line\n"
    job_manager.set_client_data.assert_called_once_with(
        "trusted_job",
        str(expected_path),
        "trusted_client",
        expected_data_type,
        fl_ctx,
    )
    assert StorageSpec.is_valid_component(f"{expected_data_type}_trusted_client")


def _make_recv_fl_ctx(client_name="trusted_client", site_allows: bool = True):
    peer_ctx = FLContext()
    peer_ctx.put(key=ReservedKey.IDENTITY_NAME, value=client_name, private=True, sticky=False)
    peer_ctx.put(key=ReservedKey.RUN_NUM, value="trusted_job", private=True, sticky=False)

    engine = Mock()
    engine.get_component.return_value = Mock()
    if site_allows is None:
        engine.get_client_from_name.return_value = None
    else:
        engine.get_client_from_name.return_value = _allowed_client(client_name, allow=site_allows)

    fl_ctx = FLContext()
    fl_ctx.put(key=ReservedKey.IDENTITY_NAME, value="server", private=True, sticky=False)
    fl_ctx.put(key=ReservedKey.ENGINE, value=engine, private=True, sticky=False)
    fl_ctx.set_peer_context(peer_ctx)
    return fl_ctx


def test_job_log_receiver_logs_error_once_when_site_does_not_allow(tmp_path):
    receiver = JobLogReceiver(dest_dir=str(tmp_path))
    fl_ctx = _make_recv_fl_ctx(site_allows=False)
    stream_ctx = {
        StreamCtxKey.CLIENT_NAME: "trusted_client",
        StreamCtxKey.JOB_ID: "trusted_job",
        KEY_FILE_NAME: WorkspaceConstants.LOG_FILE_NAME,
        StreamContextKey.RC: ReturnCode.OK,
    }

    with patch.object(receiver, "log_error") as log_error:
        receiver._on_chunk_received(b"a\n", stream_ctx, fl_ctx)
        receiver._on_chunk_received(b"b\n", stream_ctx, fl_ctx)

    assert log_error.call_count == 1
    assert "allow_log_streaming" in log_error.call_args.args[1]


def test_job_log_receiver_logs_error_once_when_client_not_registered(tmp_path):
    receiver = JobLogReceiver(dest_dir=str(tmp_path))
    fl_ctx = _make_recv_fl_ctx(site_allows=None)
    stream_ctx = {
        StreamCtxKey.CLIENT_NAME: "trusted_client",
        StreamCtxKey.JOB_ID: "trusted_job",
        KEY_FILE_NAME: WorkspaceConstants.LOG_FILE_NAME,
        StreamContextKey.RC: ReturnCode.OK,
    }

    with patch.object(receiver, "log_error") as log_error:
        receiver._on_chunk_received(b"a\n", stream_ctx, fl_ctx)
        receiver._on_chunk_received(b"b\n", stream_ctx, fl_ctx)

    assert log_error.call_count == 1


def test_job_log_receiver_does_not_warn_when_site_allows(tmp_path):
    receiver = JobLogReceiver(dest_dir=str(tmp_path))
    fl_ctx = _make_recv_fl_ctx(site_allows=True)
    stream_ctx = {
        StreamCtxKey.CLIENT_NAME: "trusted_client",
        StreamCtxKey.JOB_ID: "trusted_job",
        KEY_FILE_NAME: WorkspaceConstants.LOG_FILE_NAME,
        StreamContextKey.RC: ReturnCode.OK,
    }

    with patch.object(receiver, "log_error") as log_error:
        receiver._on_chunk_received(b"a\n", stream_ctx, fl_ctx)

    log_error.assert_not_called()
