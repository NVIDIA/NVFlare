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

from unittest.mock import Mock

from nvflare.apis.fl_constant import ReservedKey, ReturnCode, StreamCtxKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.streaming import StreamContextKey
from nvflare.app_common.logging.job_log_receiver import JobLogReceiver
from nvflare.app_common.streamers.log_streamer import KEY_FILE_NAME


def test_job_log_receiver_uses_trusted_peer_identity_for_storage(tmp_path):
    receiver = JobLogReceiver(dest_dir=str(tmp_path))

    peer_ctx = FLContext()
    peer_ctx.put(key=ReservedKey.IDENTITY_NAME, value="trusted_client", private=True, sticky=False)
    peer_ctx.put(key=ReservedKey.RUN_NUM, value="trusted_job", private=True, sticky=False)

    job_manager = Mock()
    engine = Mock()
    engine.get_component.return_value = job_manager

    fl_ctx = FLContext()
    fl_ctx.put(key=ReservedKey.IDENTITY_NAME, value="server", private=True, sticky=False)
    fl_ctx.put(key=ReservedKey.RUN_NUM, value="server_job", private=True, sticky=False)
    fl_ctx.put(key=ReservedKey.ENGINE, value=engine, private=True, sticky=False)
    fl_ctx.set_peer_context(peer_ctx)

    stream_ctx = {
        StreamCtxKey.CLIENT_NAME: "../../forged_client",
        StreamCtxKey.JOB_ID: "../../forged_job",
        KEY_FILE_NAME: "live.log",
    }
    stream_ctx[StreamContextKey.RC] = ReturnCode.OK

    receiver._on_chunk_received(b"log line\n", stream_ctx, fl_ctx)
    receiver._on_stream_done(stream_ctx, fl_ctx)

    expected_path = tmp_path / "trusted_job" / "trusted_client" / "live.log"
    assert expected_path.exists()
    assert expected_path.read_bytes() == b"log line\n"
    job_manager.set_client_data.assert_called_once_with(
        "trusted_job",
        str(expected_path),
        "trusted_client",
        "live.log",
        fl_ctx,
    )
