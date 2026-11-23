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

from nvflare.apis.fl_context import FLContextManager
from nvflare.apis.shareable import ReturnCode, Shareable
from nvflare.apis.streaming import StreamableEngine
from nvflare.app_common.streamers.container_retriever import ContainerRetriever
from nvflare.app_common.streamers.container_streamer import ContainerStreamer
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
