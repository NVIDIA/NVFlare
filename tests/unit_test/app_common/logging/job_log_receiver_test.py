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

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import (
    FLContextKey,
    ProcessType,
    ReservedKey,
    ReturnCode,
    StreamCtxKey,
    SystemComponents,
    WorkspaceConstants,
)
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.storage import DataTypes, StorageSpec
from nvflare.apis.streaming import StreamableEngine, StreamContextKey
from nvflare.app_common.logging.constants import LIVE_LOG_TOPIC, Channels
from nvflare.app_common.logging.job_log_receiver import JobLogReceiver
from nvflare.app_common.streamers.log_streamer import KEY_DATA, KEY_DATA_SIZE, KEY_EOF, KEY_FILE_NAME, KEY_HEARTBEAT
from nvflare.private.event import fire_event
from nvflare.private.fed.tbi import TBI
from nvflare.private.stream_runner import TOPIC_STREAM_REQUEST, HeaderKey, ObjectStreamer


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
    with patch.object(receiver, "log_info") as log_info:
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
    log_info.assert_called_once_with(
        fl_ctx,
        f"Saved live log '{file_name}' from trusted_client for job trusted_job",
    )


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


def test_job_log_receiver_does_not_warn_when_client_not_registered(tmp_path):
    # Default-allow: an unknown client does not produce a security error;
    # only an explicit allow_log_streaming=False in site_config does.
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

    log_error.assert_not_called()


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


def test_job_log_receiver_does_not_warn_for_empty_error_log_stream(tmp_path):
    receiver = JobLogReceiver(dest_dir=str(tmp_path))
    fl_ctx = _make_recv_fl_ctx()
    stream_ctx = {
        KEY_FILE_NAME: WorkspaceConstants.ERROR_LOG_FILE_NAME,
        StreamContextKey.RC: ReturnCode.OK,
    }

    with patch.object(receiver, "log_warning") as log_warning:
        receiver._on_stream_done(stream_ctx, fl_ctx)

    log_warning.assert_not_called()


def test_job_log_receiver_warns_for_empty_regular_log_stream(tmp_path):
    receiver = JobLogReceiver(dest_dir=str(tmp_path))
    fl_ctx = _make_recv_fl_ctx()
    stream_ctx = {
        KEY_FILE_NAME: WorkspaceConstants.LOG_FILE_NAME,
        StreamContextKey.RC: ReturnCode.OK,
    }

    with patch.object(receiver, "log_warning") as log_warning:
        receiver._on_stream_done(stream_ctx, fl_ctx)

    log_warning.assert_called_once()
    assert "No log data received from trusted_client for job trusted_job" in log_warning.call_args.args[1]


def test_job_log_receiver_does_not_log_saved_when_storage_fails(tmp_path):
    receiver = JobLogReceiver(dest_dir=str(tmp_path))
    fl_ctx = _make_recv_fl_ctx()
    job_manager = fl_ctx.get_engine().get_component(SystemComponents.JOB_MANAGER)
    job_manager.set_client_data.side_effect = RuntimeError("storage failed")
    stream_ctx = {
        KEY_FILE_NAME: WorkspaceConstants.ERROR_LOG_FILE_NAME,
        StreamContextKey.RC: ReturnCode.OK,
    }

    receiver._on_chunk_received(b"error\n", stream_ctx, fl_ctx)
    with patch.object(receiver, "log_info") as log_info:
        with pytest.raises(RuntimeError, match="storage failed"):
            receiver._on_stream_done(stream_ctx, fl_ctx)

    log_info.assert_not_called()


def test_job_log_receiver_delays_end_run_until_all_active_streams_finish():
    receiver = JobLogReceiver()
    stream_ctx_1 = {
        KEY_FILE_NAME: WorkspaceConstants.ERROR_LOG_FILE_NAME,
        StreamContextKey.RC: ReturnCode.OK,
    }
    stream_ctx_2 = {
        KEY_FILE_NAME: WorkspaceConstants.ERROR_LOG_FILE_NAME,
        StreamContextKey.RC: ReturnCode.OK,
    }

    receiver._on_stream_started(stream_ctx_1, FLContext())
    receiver._on_stream_started(stream_ctx_2, FLContext())

    waiting_ctx = FLContext()
    receiver._check_end_run_readiness(EventType.CHECK_END_RUN_READINESS, waiting_ctx)
    assert waiting_ctx.get_prop(FLContextKey.NOT_READY_TO_END_RUN) is True

    receiver._on_stream_done(stream_ctx_1, _make_recv_fl_ctx())

    still_waiting_ctx = FLContext()
    receiver._check_end_run_readiness(EventType.CHECK_END_RUN_READINESS, still_waiting_ctx)
    assert still_waiting_ctx.get_prop(FLContextKey.NOT_READY_TO_END_RUN) is True

    receiver._on_stream_done(stream_ctx_2, _make_recv_fl_ctx())

    ready_ctx = FLContext()
    receiver._check_end_run_readiness(EventType.CHECK_END_RUN_READINESS, ready_ctx)
    assert ready_ctx.get_prop(FLContextKey.NOT_READY_TO_END_RUN, False) is False


def test_job_log_receiver_releases_readiness_when_stream_finalization_fails(tmp_path):
    receiver = JobLogReceiver(dest_dir=str(tmp_path))
    fl_ctx = _make_recv_fl_ctx()
    job_manager = fl_ctx.get_engine().get_component(SystemComponents.JOB_MANAGER)
    job_manager.set_client_data.side_effect = RuntimeError("storage failed")
    stream_ctx = {
        KEY_FILE_NAME: WorkspaceConstants.LOG_FILE_NAME,
        StreamContextKey.RC: ReturnCode.OK,
    }

    receiver._on_stream_started(stream_ctx, fl_ctx)
    receiver._on_chunk_received(b"log line\n", stream_ctx, fl_ctx)
    with pytest.raises(RuntimeError, match="storage failed"):
        receiver._on_stream_done(stream_ctx, fl_ctx)

    ready_ctx = FLContext()
    receiver._check_end_run_readiness(EventType.CHECK_END_RUN_READINESS, ready_ctx)
    assert ready_ctx.get_prop(FLContextKey.NOT_READY_TO_END_RUN, False) is False


@pytest.fixture
def log_transports(tmp_path):
    """Use the real receiving protocol with separate parent/job receiver instances."""
    transports = []

    def create(process_type=ProcessType.SERVER_JOB):
        receiver = JobLogReceiver(dest_dir=str(tmp_path / str(len(transports))), idle_timeout=0)
        streamer = ObjectStreamer(Mock())
        transports.append(streamer)
        fl_ctx = _make_recv_fl_ctx()
        engine = Mock(spec=StreamableEngine)
        engine.get_component = Mock(return_value=Mock())
        engine.get_client_from_name = Mock(return_value=_allowed_client())
        engine.fire_event = lambda event, ctx: fire_event(event, [receiver], ctx)
        engine.register_stream_processing.side_effect = streamer.register_stream_processing
        fl_ctx.put(key=ReservedKey.ENGINE, value=engine, private=True, sticky=False)
        fl_ctx.put(key=ReservedKey.PROCESS_TYPE, value=process_type, private=True, sticky=False)
        event = EventType.SYSTEM_START if process_type == ProcessType.SERVER_PARENT else EventType.START_RUN
        fire_event(event, [receiver], fl_ctx)
        return receiver, streamer, fl_ctx

    yield create

    for streamer in transports:
        for tx_id in list(streamer.tx_table):
            streamer._end_tx(tx_id, ReturnCode.TASK_ABORTED, streamer.tx_table[tx_id].consumer._fl_ctx)
        streamer.shutdown()


def _log_request(tx_id, seq=0, data=b"", eof=False, heartbeat=False, stream_ctx=None):
    request = Shareable()
    request[KEY_DATA] = data
    request[KEY_DATA_SIZE] = len(data)
    request[KEY_EOF] = eof
    request[KEY_HEARTBEAT] = heartbeat
    request.set_header(HeaderKey.TX_ID, tx_id)
    request.set_header(HeaderKey.SEQ, seq)
    request.set_header(HeaderKey.CHANNEL, Channels.LOG_STREAMING_CHANNEL)
    request.set_header(HeaderKey.TOPIC, LIVE_LOG_TOPIC)
    if seq == 0:
        request.set_header(HeaderKey.CTX, stream_ctx or {KEY_FILE_NAME: "log.json"})
    return request


def _ready_to_end(receiver):
    fl_ctx = FLContext()
    fire_event(EventType.CHECK_END_RUN_READINESS, [receiver], fl_ctx)
    return not fl_ctx.get_prop(FLContextKey.NOT_READY_TO_END_RUN, False)


def test_sender_marker_cannot_hide_another_active_stream(log_transports):
    receiver, streamer, fl_ctx = log_transports()
    forged_ctx = {KEY_FILE_NAME: "log.json", "JobLogReceiver.stream_active": True}
    reply = streamer._handle_request(
        TOPIC_STREAM_REQUEST, _log_request("forged", heartbeat=True, stream_ctx=forged_ctx), fl_ctx
    )
    assert reply.get_return_code() == ReturnCode.OK
    reply = streamer._handle_request(TOPIC_STREAM_REQUEST, _log_request("normal", heartbeat=True), fl_ctx)
    assert reply.get_return_code() == ReturnCode.OK

    streamer._handle_request(TOPIC_STREAM_REQUEST, _log_request("forged", seq=1, eof=True), fl_ctx)
    assert not _ready_to_end(receiver)

    streamer._handle_request(TOPIC_STREAM_REQUEST, _log_request("normal", seq=1, eof=True), fl_ctx)
    assert _ready_to_end(receiver)


def test_ready_transition_rejects_a_racing_first_message(log_transports):
    receiver, streamer, fl_ctx = log_transports()
    entered = threading.Event()
    resume = threading.Event()
    on_started = receiver._on_stream_started

    def delayed_start(stream_ctx, callback_fl_ctx):
        entered.set()
        assert resume.wait(timeout=5)
        return on_started(stream_ctx, callback_fl_ctx)

    with patch.object(receiver, "_on_stream_started", side_effect=delayed_start):
        fire_event(EventType.START_RUN, [receiver], fl_ctx)
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(
                streamer._handle_request, TOPIC_STREAM_REQUEST, _log_request("late", data=b"late log\n"), fl_ctx
            )
            try:
                assert entered.wait(timeout=5)
                assert _ready_to_end(receiver)
            finally:
                resume.set()
            assert pending.result(timeout=5).get_return_code() == ReturnCode.EXECUTION_EXCEPTION

    assert not streamer.tx_table
    fl_ctx.get_engine().get_component.return_value.set_client_data.assert_not_called()


def test_accepted_stream_blocks_readiness_until_eof_is_persisted(log_transports):
    receiver, streamer, fl_ctx = log_transports()
    job_manager = fl_ctx.get_engine().get_component.return_value
    during_storage = []
    job_manager.set_client_data.side_effect = lambda *args: during_storage.append(_ready_to_end(receiver))

    reply = streamer._handle_request(TOPIC_STREAM_REQUEST, _log_request("active", data=b"before stop\n"), fl_ctx)
    assert reply.get_return_code() == ReturnCode.OK
    assert not _ready_to_end(receiver)
    reply = streamer._handle_request(
        TOPIC_STREAM_REQUEST, _log_request("active", seq=1, data=b"final bytes\n", eof=True), fl_ctx
    )

    assert reply.get_return_code() == ReturnCode.OK
    assert during_storage == [False]
    assert _ready_to_end(receiver)
    job_manager.set_client_data.assert_called_once()
    with open(job_manager.set_client_data.call_args.args[1], "rb") as log_file:
        assert log_file.read() == b"before stop\nfinal bytes\n"


def test_readiness_timeout_still_bounds_a_stalled_stream(log_transports):
    receiver, streamer, fl_ctx = log_transports()
    streamer._handle_request(TOPIC_STREAM_REQUEST, _log_request("stalled", data=b"partial\n"), fl_ctx)
    runner = TBI()

    with (
        patch.object(runner, "get_positive_float_var", side_effect=[5.0, 0.5]),
        patch("nvflare.private.fed.tbi.time.time", side_effect=[0.0, 1.0, 6.0]),
        patch("nvflare.private.fed.tbi.time.sleep") as sleep,
        patch.object(runner, "log_warning") as warning,
    ):
        runner.check_end_run_readiness(fl_ctx)

    sleep.assert_called_once_with(0.5)
    warning.assert_called_once_with(fl_ctx, "quit waiting for component ready-to-end-run after 5.0 seconds")
    fire_event(EventType.END_RUN, [receiver], fl_ctx)
    reply = streamer._handle_request(TOPIC_STREAM_REQUEST, _log_request("late", data=b"late\n"), fl_ctx)
    assert reply.get_return_code() == ReturnCode.EXECUTION_EXCEPTION


def test_start_run_reopens_admission_after_previous_run(log_transports):
    receiver, streamer, fl_ctx = log_transports()
    assert _ready_to_end(receiver)
    fire_event(EventType.END_RUN, [receiver], fl_ctx)
    fire_event(EventType.START_RUN, [receiver], fl_ctx)

    reply = streamer._handle_request(TOPIC_STREAM_REQUEST, _log_request("new-run", heartbeat=True), fl_ctx)
    assert reply.get_return_code() == ReturnCode.OK
    assert not _ready_to_end(receiver)


def test_parent_receiver_persists_after_job_receiver_has_shut_down(log_transports):
    _, parent_streamer, parent_ctx = log_transports(ProcessType.SERVER_PARENT)
    job_receiver, job_streamer, job_ctx = log_transports(ProcessType.SERVER_JOB)
    parent_streamer._handle_request(
        TOPIC_STREAM_REQUEST, _log_request("parent-stream", data=b"before job shutdown\n"), parent_ctx
    )

    assert _ready_to_end(job_receiver)
    fire_event(EventType.END_RUN, [job_receiver], job_ctx)
    job_streamer.shutdown()

    reply = parent_streamer._handle_request(
        TOPIC_STREAM_REQUEST,
        _log_request("parent-stream", seq=1, data=b"after job shutdown\n", eof=True),
        parent_ctx,
    )
    assert reply.get_return_code() == ReturnCode.OK
    job_manager = parent_ctx.get_engine().get_component.return_value
    job_manager.set_client_data.assert_called_once()
    assert job_manager.set_client_data.call_args.args[0] == "trusted_job"
    with open(job_manager.set_client_data.call_args.args[1], "rb") as log_file:
        assert log_file.read() == b"before job shutdown\nafter job shutdown\n"


@pytest.mark.parametrize(
    "event_type",
    [EventType.SYSTEM_START, EventType.ABOUT_TO_START_RUN, EventType.START_RUN],
)
def test_register_fires_on_run_lifecycle_events(event_type):
    # Each trigger event must call register_stream_processing so the receiver
    # gets wired up wherever it happens to live (parent process via
    # SYSTEM_START, in-process job via ABOUT_TO_START_RUN, or per-job
    # subprocess via START_RUN — the only event that fires server-side in
    # ServerRunner.run).
    receiver = JobLogReceiver()
    fl_ctx = FLContext()

    with patch("nvflare.app_common.logging.job_log_receiver.LogStreamer.register_stream_processing") as mock_register:
        receiver._register(event_type, fl_ctx)

    mock_register.assert_called_once()
    kwargs = mock_register.call_args.kwargs
    assert kwargs["channel"] == Channels.LOG_STREAMING_CHANNEL
    assert kwargs["topic"] == LIVE_LOG_TOPIC
    assert kwargs["stream_started_cb"] == receiver._on_stream_started
    assert kwargs["stream_done_cb"] == receiver._on_stream_done


def test_register_reregisters_on_each_event():
    # In a per-job server subprocess the active ObjectStreamer is replaced
    # whenever the engine swaps in a new RunManager. The receiver must
    # re-register against the new streamer rather than skip on a "registered
    # once" latch; otherwise the new run_manager's registry is empty and the
    # first incoming chunk fails with "no stream processing info registered
    # for log_streaming:live_log".
    receiver = JobLogReceiver()
    fl_ctx = FLContext()

    with patch("nvflare.app_common.logging.job_log_receiver.LogStreamer.register_stream_processing") as mock_register:
        receiver._register(EventType.SYSTEM_START, fl_ctx)
        receiver._register(EventType.START_RUN, fl_ctx)
        receiver._register(EventType.START_RUN, fl_ctx)

    assert mock_register.call_count == 3


def test_register_event_handlers_cover_server_subprocess_path():
    # Guard against an accidental regression of PR #4558: dropping START_RUN
    # from the handler list left server-job subprocesses (where
    # ABOUT_TO_START_RUN never fires and SYSTEM_START is only fired by the
    # parent's server_deployer) with no event that would call _register.
    receiver = JobLogReceiver()

    registered_events = {
        event_type
        for event_type, entries in receiver.get_event_handlers().items()
        if any(handler == receiver._register for handler, _ in entries)
    }

    assert EventType.START_RUN in registered_events
    assert EventType.SYSTEM_START in registered_events
    assert EventType.ABOUT_TO_START_RUN in registered_events


def test_register_event_handlers_cover_end_run_readiness():
    receiver = JobLogReceiver()

    registered_events = {
        event_type
        for event_type, entries in receiver.get_event_handlers().items()
        if any(handler == receiver._check_end_run_readiness for handler, _ in entries)
    }

    assert EventType.CHECK_END_RUN_READINESS in registered_events
