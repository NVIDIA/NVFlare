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

from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_constant import ReturnCode as FLReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.storage import DataTypes
from nvflare.apis.streaming import StreamContextKey
from nvflare.app_common.logging.job_log_receiver import JobLogReceiver
from nvflare.app_common.streamers.log_streamer import KEY_FILE_NAME
from nvflare.fuel.f3.cellnet.core_cell import ReturnCode
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.private.aux_runner import AuxRunner
from nvflare.private.defs import CellMessageHeaderKeys, new_cell_message
from nvflare.private.fed.server.server_command_agent import ServerCommandAgent


class TestAuxCommunicateAuthCheck:
    """Verify that aux_communicate returns early when authentication fails."""

    def test_dispatch_not_called_on_auth_failure(self):
        mock_engine = MagicMock()
        mock_fl_ctx = MagicMock()
        mock_engine.new_context.return_value.__enter__ = MagicMock(return_value=mock_fl_ctx)
        mock_engine.new_context.return_value.__exit__ = MagicMock(return_value=False)
        mock_fl_ctx.get_engine.return_value = mock_engine

        # authentication_check returns an error string
        mock_engine.server.server_state.aux_communicate.return_value = "state_ok"
        mock_engine.server.authentication_check.return_value = "auth_failed"

        agent = ServerCommandAgent(engine=mock_engine, cell=MagicMock())

        request = CellMessage()
        request.payload = {"test": "data"}
        request.set_header(MessageHeaderKey.TOPIC, "test_topic")
        result = agent.aux_communicate(request)

        assert not mock_engine.dispatch.called, "engine.dispatch should NOT be called when authentication fails"
        mock_fl_ctx.set_prop.assert_not_called()
        assert result is not None, "aux_communicate should return an error reply, not None"
        assert result.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.AUTHENTICATION_ERROR


def test_stopped_server_command_agent_rejects_requests_without_engine_access():
    engine = MagicMock()
    agent = ServerCommandAgent(engine=engine, cell=MagicMock())
    agent.shutdown()
    request = CellMessage()

    execute_reply = agent.execute_command(request)
    aux_reply = agent.aux_communicate(request)

    assert execute_reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.SERVICE_UNAVAILABLE
    assert aux_reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.SERVICE_UNAVAILABLE
    engine.new_context.assert_not_called()


@pytest.mark.parametrize("file_name", ["log.txt", "log.json", "external.log"])
@pytest.mark.parametrize(
    "sender,claimed_sender,accepted",
    [("site-1", "site-1", True), ("site-1", "another-client", False), (None, "site-1", False)],
)
def test_job_aux_dispatch_preserves_log_stream_identity(tmp_path, file_name, sender, claimed_sender, accepted):
    engine = MagicMock()
    engine.server.authentication_check.return_value = None
    engine.get_client_from_name.return_value = None
    engine.get_workspace.return_value = None

    def new_context():
        fl_ctx = FLContext()
        fl_ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=False)
        return fl_ctx

    engine.new_context.side_effect = new_context
    aux_runner = AuxRunner(engine)
    engine.dispatch.side_effect = aux_runner.dispatch
    receiver = JobLogReceiver(dest_dir=str(tmp_path))
    stream_ctx = {KEY_FILE_NAME: file_name, StreamContextKey.RC: FLReturnCode.OK}

    def receive_log(topic, request, fl_ctx):
        if request["eof"]:
            receiver._on_stream_done(stream_ctx, fl_ctx)
        else:
            receiver._on_chunk_received(request["data"], stream_ctx, fl_ctx)
        return make_reply(FLReturnCode.OK)

    aux_runner.register_aux_message_handler("test-log", receive_log)
    agent = ServerCommandAgent(engine=engine, cell=MagicMock())

    def send(data=b"", eof=False):
        payload = Shareable()
        payload.update(data=data, eof=eof)
        payload.set_peer_props(
            {
                ReservedKey.IDENTITY_NAME: claimed_sender,
                ReservedKey.RUN_NUM: "job-1",
                FLContextKey.CLIENT_NAME: "forged-client",
            }
        )
        return agent.aux_communicate(
            new_cell_message({MessageHeaderKey.TOPIC: "test-log", CellMessageHeaderKeys.CLIENT_NAME: sender}, payload)
        )

    try:
        reply = send(b"first chunk\n")
        if not accepted:
            assert reply.payload.get_return_code() == FLReturnCode.HANDLER_EXCEPTION
            engine.get_component.return_value.set_client_data.assert_not_called()
            assert list(tmp_path.iterdir()) == []
            return

        assert reply.payload.get_return_code() == FLReturnCode.OK
        assert send(b"second chunk\n").payload.get_return_code() == FLReturnCode.OK
        assert send(eof=True).payload.get_return_code() == FLReturnCode.OK
        path = tmp_path / "job-1" / "site-1" / file_name
        assert path.read_bytes() == b"first chunk\nsecond chunk\n"
        stored = engine.get_component.return_value.set_client_data
        stored.assert_called_once()
        assert stored.call_args.args[:4] == ("job-1", str(path), "site-1", f"{DataTypes.LOG.value}_{file_name}")
        assert not receiver._stream_files
    finally:
        for state in receiver._stream_files.values():
            state["file"].close()
