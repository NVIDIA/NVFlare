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

from collections import OrderedDict, UserList
from unittest.mock import Mock, patch

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ProcessType, ReservedKey
from nvflare.apis.fl_context import FLContextManager
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.streaming import StreamableEngine, StreamContextKey
from nvflare.app_common.streamers.container_retriever import ContainerRetriever
from nvflare.app_common.streamers.container_streamer import ContainerStreamer
from nvflare.app_common.streamers.object_retriever import RETRIEVER_TX_ID
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.message import Message
from nvflare.private.aux_runner import AuxMsgTarget, AuxRunner
from nvflare.private.fed.client.client_run_manager import ClientRunManager
from nvflare.private.fed.client.command_agent import CommandAgent
from nvflare.private.fed.task_worker.runtime import TaskRuntime
from nvflare.private.stream_runner import TOPIC_STREAM_ABORT, TOPIC_STREAM_REQUEST, HeaderKey, ObjectStreamer


@pytest.fixture
def container_stream():
    aux = Mock()
    streamer = ObjectStreamer(aux)
    engine = Mock(spec=StreamableEngine)
    engine.register_stream_processing.side_effect = streamer.register_stream_processing
    engine.stream_objects.side_effect = streamer.stream
    fl_ctx = FLContextManager(engine=engine).new_context()
    stream_done = Mock(return_value=None)
    ContainerStreamer.register_stream_processing(fl_ctx, "container", "data", stream_done_cb=stream_done)

    def dispatch(topic, request, **kwargs):
        if topic == TOPIC_STREAM_ABORT:
            streamer._handle_abort(topic, request, fl_ctx)
            return {}
        assert topic == TOPIC_STREAM_REQUEST
        return {"receiver": streamer._handle_request(topic, request, fl_ctx)}

    aux.send_aux_request.side_effect = dispatch
    try:
        yield fl_ctx, streamer, stream_done
    finally:
        streamer.shutdown()


@pytest.mark.parametrize(
    "container, wire_type",
    [
        ({"model": 1, "metrics": 2}, "dict"),
        ([3, 1], "list"),
        ({1, 3}, "set"),
        (OrderedDict([("z", 1), ("a", 2)]), "collections.OrderedDict"),
    ],
)
def test_supported_containers_keep_wire_name_and_round_trip(container_stream, container, wire_type):
    fl_ctx, streamer, stream_done = container_stream
    stream_ctx = {"request": "retrieve"}
    rc, success = ContainerStreamer.stream_container("container", "data", stream_ctx, ["receiver"], container, fl_ctx)

    assert rc == ReturnCode.OK
    assert success is True
    assert stream_ctx["ContainerStreamer.type"] == wire_type
    stream_done.assert_called_once()
    result = ContainerStreamer.get_result(stream_done.call_args.args[0])
    assert type(result) is type(container)
    assert result == container
    assert not streamer.tx_table


@pytest.mark.parametrize("wire_type", ["example_container.CustomContainer", "tuple", "", None, []])
def test_unsupported_container_metadata_is_rejected_before_import(container_stream, wire_type):
    fl_ctx, streamer, stream_done = container_stream
    request = Shareable()
    request.set_header(HeaderKey.TX_ID, "container-request")
    request.set_header(HeaderKey.SEQ, 0)
    request.set_header(HeaderKey.TOPIC, "data")
    request.set_header(HeaderKey.CHANNEL, "container")
    request.set_header(HeaderKey.CTX, {"ContainerStreamer.type": wire_type, "ContainerStreamer.size": 1})

    with patch("nvflare.fuel.utils.class_loader.importlib.import_module", side_effect=ImportError) as import_module:
        reply = streamer._handle_request(TOPIC_STREAM_REQUEST, request, fl_ctx)

    assert reply.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    import_module.assert_not_called()
    stream_done.assert_not_called()
    assert not streamer.tx_table


def test_unsupported_container_fails_before_sending(container_stream):
    fl_ctx, _, _ = container_stream

    with pytest.raises(ValueError, match="Unsupported container type"):
        ContainerStreamer.stream_container("container", "data", {}, ["receiver"], UserList([1]), fl_ctx)

    fl_ctx.get_engine().stream_objects.assert_not_called()


def test_retriever_rejects_unsupported_container_at_registration():
    retriever = ContainerRetriever()
    container = [1]
    retriever.add_container("model", container)

    with pytest.raises(ValueError, match="Unsupported container type"):
        retriever.add_container("model", UserList([2]))

    assert retriever.containers["model"] is container


@pytest.mark.parametrize(
    "process_type, custom_route, expected_origin",
    [
        (ProcessType.CLIENT_JOB, False, "relay.source.job"),
        (ProcessType.SERVER_PARENT, False, "relay.source"),
        (ProcessType.SERVER_PARENT, True, "relay.source.job"),
    ],
)
def test_legacy_container_rejection_completes_only_its_retrieval_from_the_expected_peer(
    process_type, custom_route, expected_origin
):
    engine = Mock(spec=ClientRunManager)
    if custom_route:
        del engine._get_aux_msg_target
        engine.get_stream_peer_origin.return_value = expected_origin
    else:
        engine._get_aux_msg_target.return_value = AuxMsgTarget("source", "relay.source")
        engine.get_stream_peer_origin.side_effect = lambda site, ctx: ClientRunManager.get_stream_peer_origin(
            engine, site, ctx
        )
    context_manager = FLContextManager(
        engine=engine,
        identity_name="receiver",
        job_id="job",
        private_stickers={FLContextKey.PROCESS_TYPE: process_type},
    )
    engine.new_context.side_effect = context_manager.new_context
    aux = AuxRunner(engine)
    engine.dispatch.side_effect = aux.dispatch
    engine.register_aux_message_handler.side_effect = aux.register_aux_message_handler
    streamer = ObjectStreamer(aux)
    engine.register_stream_processing.side_effect = streamer.register_stream_processing
    agent = CommandAgent(Mock())
    agent.engine = engine
    retriever = ContainerRetriever()
    fl_ctx = engine.new_context()
    stream_replies = []
    waiter_ready = []

    def send_legacy_container(origin, stream_id, retrieval_id):
        # An older sender acknowledges the retrieval and describes its mapping
        # subclass using the original first-chunk headers.
        stream_request = Shareable({"ContainerStreamer.entry": ("key", 1), "ContainerStreamer.last": True})
        stream_request.set_header(HeaderKey.TX_ID, stream_id)
        stream_request.set_header(HeaderKey.SEQ, 0)
        stream_request.set_header(HeaderKey.CHANNEL, retriever.stream_channel)
        stream_request.set_header(HeaderKey.TOPIC, retriever.topic)
        stream_request.set_header(
            HeaderKey.CTX,
            {
                RETRIEVER_TX_ID: retrieval_id,
                "ContainerStreamer.type": "collections.defaultdict",
                "ContainerStreamer.size": 1,
            },
        )
        # The wrong sender also claims to be the expected site in the payload.
        stream_request.set_peer_props({ReservedKey.IDENTITY_NAME: "source"})
        message = Message(
            {MessageHeaderKey.TOPIC: TOPIC_STREAM_REQUEST, MessageHeaderKey.ORIGIN: origin}, stream_request
        )
        stream_replies.append(agent.aux_communication(message).payload.get_return_code())

    def send_aux_request(*, request, targets, **kwargs):
        assert targets == ["source"]
        retrieval_id = request[RETRIEVER_TX_ID]
        waiter = retriever.tx_table[retrieval_id]
        wrong_origin = "relay.source" if expected_origin.endswith(".job") else "relay.source.job"
        for origin, stream_id, claimed_retrieval in (
            (expected_origin, "wrong-request", "different-retrieval"),
            ("relay.other.job", "wrong-peer", retrieval_id),
            (wrong_origin, "wrong-process-scope", retrieval_id),
            (expected_origin, "matching", retrieval_id),
        ):
            send_legacy_container(origin, stream_id, claimed_retrieval)
            waiter_ready.append(waiter.is_set())
        return {"source": make_reply(ReturnCode.OK)}

    engine.send_aux_request.side_effect = send_aux_request
    try:
        retriever.handle_event(EventType.START_RUN, fl_ctx)
        result = retriever.retrieve_container("source", fl_ctx, timeout=0.5, name="model")
        assert result == (ReturnCode.UNSUPPORTED_CONTAINER_TYPE, None)
        assert waiter_ready == [False, False, False, True]
        assert stream_replies == [ReturnCode.EXECUTION_EXCEPTION] * 4
        assert not retriever.tx_table
        assert not streamer.tx_table
    finally:
        streamer.shutdown()


@pytest.mark.parametrize("route_resolution_fails", [False, True])
def test_container_retrieval_can_complete_without_origin_resolution(route_resolution_fails):
    methods = ["send_aux_request", "fire_event"]
    if route_resolution_fails:
        methods.append("get_stream_peer_origin")
    engine = Mock(spec=methods)
    if route_resolution_fails:
        engine.get_stream_peer_origin.side_effect = RuntimeError("origin unavailable")
    fl_ctx = FLContextManager(engine=engine, job_id="job").new_context()
    retriever = ContainerRetriever()

    def send(*, request, **kwargs):
        stream_ctx = {
            RETRIEVER_TX_ID: request[RETRIEVER_TX_ID],
            StreamContextKey.RC: ReturnCode.OK,
            "ContainerStreamer.result": {"key": 1},
        }
        retriever._handle_stream_done(stream_ctx, fl_ctx)
        return {"source": make_reply(ReturnCode.OK)}

    engine.send_aux_request.side_effect = send
    assert retriever.retrieve_container("source", fl_ctx, timeout=0.5, name="model") == (ReturnCode.OK, {"key": 1})
    engine.send_aux_request.assert_called_once()


def test_container_retrieval_uses_task_workers_existing_service_rejection(caplog):
    runtime = TaskRuntime(None, "receiver", "job")
    retriever = ContainerRetriever()

    rc, data = retriever.retrieve_container("source", runtime.new_context(), timeout=0.5, name="model")

    assert (rc, data) == (ReturnCode.EXECUTION_EXCEPTION, None)
    assert "auxiliary federation requests" in caplog.text
    assert not retriever.tx_table
