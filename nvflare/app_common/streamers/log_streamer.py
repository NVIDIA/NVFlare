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
"""
LogStreamer — live-tailing streamer for growing log files
=========================================================

LogStreamer differs from FileStreamer in three fundamental ways:

1. **Growing file (live tail)**
   FileStreamer treats a file as a static snapshot: it opens the file, reads
   it to the end, and sends EOF.  LogStreamer instead *tails* the file — it
   keeps reading as new bytes are appended, blocking between polls until the
   caller sets ``stop_event`` and all buffered bytes have been flushed.  It
   also survives log rotation: when the inode changes (or the file shrinks
   below the current read position) it closes the stale handle and reopens
   the file at the new path from the beginning.

2. **Dead-sender detection via liveness heartbeats**
   Because the stream may be idle for extended periods (no new log lines),
   the receiver cannot distinguish "nothing to send yet" from "sender has
   crashed".  LogStreamer solves this with periodic *heartbeat* messages: if
   no data chunk has been sent for ``liveness_interval`` seconds the producer
   emits a zero-payload heartbeat.  The consumer resets its idle clock on
   every message — data or heartbeat — so the idle timer only counts genuine
   silence on the network.

3. **Automatic stream closure on idle timeout**
   The receiver runs a background watchdog thread.  If no message of any
   kind (data or heartbeat) arrives for ``idle_timeout`` seconds the watchdog
   concludes the sender is unreachable, closes the stream via the engine's
   ``END_STREAM`` hook, and fires ``stream_done_cb`` with
   ``StreamContextKey.RC = ReturnCode.TIMEOUT``.  To avoid spurious timeouts,
   ``liveness_interval`` must be strictly less than ``idle_timeout``.  This
   can be validated only when the caller knows both values; otherwise ensure
   they are consistent at deployment.

Relationship between the two parameters
----------------------------------------
- ``liveness_interval``: how often the *sender* emits a heartbeat when idle
- ``idle_timeout``: how long the *receiver* waits before declaring the sender dead

Rule: ``liveness_interval < idle_timeout``.  With the defaults (10 s and
30 s) a healthy sender heartbeats every 10 s, so the receiver's 30 s timer
is always reset well before it fires.
"""
import os
import threading
import time
from typing import List, Tuple

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.streaming import ConsumerFactory, ObjectConsumer, StreamableEngine, StreamContext, StreamContextKey
from nvflare.fuel.utils.validation_utils import check_non_negative_number, check_positive_int, check_positive_number

from .streamer_base import (  # noqa: F401
    KEY_DATA,
    KEY_DATA_SIZE,
    KEY_EOF,
    KEY_FILE_NAME,
    KEY_HEARTBEAT,
    KEY_STREAM_DONE_CB,
    BaseChunkConsumer,
    BaseChunkProducer,
    StreamerBase,
)


def _make_once(fn):
    """Return a thread-safe wrapper that calls *fn* at most once across all callers.

    Used so that ``stream_done_cb`` is executed exactly once regardless of whether
    the idle-timeout watchdog or the normal engine completion path fires first.
    """
    lock = threading.Lock()
    called = [False]

    def wrapper(*args, **kwargs):
        with lock:
            if called[0]:
                return
            called[0] = True
        return fn(*args, **kwargs)

    return wrapper


class _LogChunkConsumer(BaseChunkConsumer):
    def __init__(
        self,
        stream_ctx: StreamContext,
        chunk_received_cb,
        idle_timeout: float,
        cb_kwargs: dict,
        fl_ctx: FLContext = None,
    ):
        super().__init__()
        self._chunk_received_cb = chunk_received_cb
        self._idle_timeout = idle_timeout
        self._cb_kwargs = cb_kwargs
        self._stream_ctx = stream_ctx
        self._fl_ctx = fl_ctx  # updated on each consume(); seeded from get_consumer()
        self._last_received_time = time.time()
        self._done = threading.Event()

        if idle_timeout > 0:
            t = threading.Thread(target=self._watchdog, daemon=True)
            t.start()

    def _watchdog(self):
        """Background thread: end this stream when it goes idle."""
        poll = min(1.0, self._idle_timeout / 3)
        while not self._done.wait(timeout=poll):
            elapsed = time.time() - self._last_received_time
            if elapsed >= self._idle_timeout:
                self.logger.warning(f"log stream idle for {elapsed:.1f}s (threshold {self._idle_timeout}s) — closing")
                self._done.set()
                end_stream = self._stream_ctx.get(StreamContextKey.END_STREAM)
                if callable(end_stream):
                    # Use the latest FLContext we have seen. It may still be None if
                    # the sender dies immediately after opening the stream, but ending
                    # the transport still lets stream_runner clean up its tx_table.
                    end_stream(ReturnCode.TIMEOUT, self._fl_ctx)
                else:
                    self.logger.error("missing end_stream hook in stream context")
                return

    def consume(
        self,
        shareable: Shareable,
        stream_ctx: StreamContext,
        fl_ctx: FLContext,
    ) -> Tuple[bool, Shareable]:
        self._last_received_time = time.time()
        self._fl_ctx = fl_ctx

        # If the watchdog already fired, reject any late-arriving chunks from the
        # sender after this stream has already been closed locally.
        if self._done.is_set():
            return False, make_reply(ReturnCode.TASK_ABORTED)

        # Heartbeat: sender is alive but idle — update the liveness timestamp and continue.
        if shareable.get(KEY_HEARTBEAT):
            return True, make_reply(ReturnCode.OK)

        data = shareable.get(KEY_DATA)
        data_size = shareable.get(KEY_DATA_SIZE)
        self._validate_chunk(data, data_size)

        if data and self._chunk_received_cb:
            self._chunk_received_cb(data, stream_ctx, fl_ctx, **self._cb_kwargs)

        eof = shareable.get(KEY_EOF)
        if eof:
            return False, make_reply(ReturnCode.OK)
        return True, make_reply(ReturnCode.OK)

    def finalize(self, stream_ctx: StreamContext, fl_ctx: FLContext):
        # Stop the watchdog — the stream ended normally via EOF or engine abort.
        self._done.set()


class LogChunkConsumerFactory(ConsumerFactory):
    def __init__(self, chunk_received_cb, idle_timeout: float, stream_done_cb, cb_kwargs: dict):
        self._chunk_received_cb = chunk_received_cb
        self._idle_timeout = idle_timeout
        self._stream_done_cb = stream_done_cb
        self._cb_kwargs = cb_kwargs

    def get_consumer(self, stream_ctx: StreamContext, fl_ctx: FLContext) -> ObjectConsumer:
        if self._stream_done_cb:
            stream_ctx[KEY_STREAM_DONE_CB] = _make_once(self._stream_done_cb)
        return _LogChunkConsumer(
            stream_ctx=stream_ctx,
            chunk_received_cb=self._chunk_received_cb,
            idle_timeout=self._idle_timeout,
            cb_kwargs=self._cb_kwargs,
            fl_ctx=fl_ctx,
        )


def dispatch_stream_done(stream_ctx: StreamContext, fl_ctx: FLContext, **kwargs):
    stream_done_cb = stream_ctx.get(KEY_STREAM_DONE_CB)
    if stream_done_cb:
        return stream_done_cb(stream_ctx, fl_ctx, **kwargs)
    return None


class _LogTailProducer(BaseChunkProducer):
    def __init__(
        self,
        file_name: str,
        chunk_size: int,
        chunk_timeout: float,
        poll_interval: float,
        stop_event: threading.Event,
        liveness_interval: float,
    ):
        super().__init__()
        self.file_name = file_name
        self.chunk_size = chunk_size
        self.chunk_timeout = chunk_timeout
        self.poll_interval = poll_interval
        self.stop_event = stop_event
        self._liveness_interval = liveness_interval
        self._last_send_time = time.time()
        self.file = None
        self.inode = None
        self._draining = False  # True while doing the post-stop drain retry
        # Bootstrap-tolerance bookkeeping: at job startup the receiver may not
        # have registered its handler yet when the first chunk arrives, in
        # which case the server replies EXECUTION_EXCEPTION and (without this
        # tolerance) the stream tears down with a misleading ERROR log line
        # and no further bytes are streamed. Until we see the first OK reply
        # we treat an EXECUTION_EXCEPTION as a transient miss, roll the file
        # offset back, and let produce() re-emit the same bytes.
        self._first_ok_received = False
        self._last_data_offset = None
        self._last_was_data = False
        self._open_file()

    def _open_file(self):
        try:
            self.file = open(self.file_name, "rb")
            self.inode = os.stat(self.file_name).st_ino
            self.logger.debug(f"opened log file {self.file_name} inode={self.inode}")
        except OSError as e:
            self.logger.warning(f"cannot open log file {self.file_name}: {e}")
            self.file = None
            self.inode = None

    def _check_rotation(self) -> bool:
        """Return True if the log file has been rotated (inode change or truncation)."""
        try:
            stat = os.stat(self.file_name)
        except OSError:
            # File temporarily absent during a rotation in progress — don't switch yet.
            return False

        if stat.st_ino != self.inode:
            self.logger.info(f"log rotation detected (inode change): {self.file_name}")
            return True

        if stat.st_size < self.file.tell():
            self.logger.info(f"log rotation detected (truncation): {self.file_name}")
            return True

        return False

    def produce(
        self,
        stream_ctx: StreamContext,
        fl_ctx: FLContext,
    ) -> Tuple[Shareable, float]:
        abort_signal = fl_ctx.get_run_abort_signal()

        while True:
            # Detect and handle log rotation: close stale handle, reopen at new path.
            if self.file and self._check_rotation():
                self.file.close()
                self.file = None
                self._open_file()
            elif not self.file and os.path.exists(self.file_name):
                # File appeared (or reappeared after rotation); open it now.
                self._open_file()

            # Read the next chunk from the current position.
            if self.file:
                # Record the pre-read offset so process_replies() can roll back
                # if the first send hits a bootstrap miss on the receiver.
                if not self._first_ok_received:
                    try:
                        self._last_data_offset = self.file.tell()
                    except OSError:
                        self._last_data_offset = None
                chunk = self.file.read(self.chunk_size)
                if chunk:
                    self._draining = False  # new data arrived — reset drain state
                    self._last_send_time = time.time()
                    self._last_was_data = True
                    result = Shareable()
                    result[KEY_DATA] = chunk
                    result[KEY_DATA_SIZE] = len(chunk)
                    result[KEY_EOF] = False
                    return result, self.chunk_timeout

            # No new data — drain completely before honouring stop/abort so we
            # never drop bytes written just before the signal fires.
            # NOTE: abort_signal is intentionally checked here (after the read)
            # rather than at the top of the loop because in the simulator the
            # abort signal is triggered *before* ABOUT_TO_END_RUN fires, so
            # checking it first causes the producer to exit immediately and
            # lose any log bytes still buffered in the file.
            if self.stop_event.is_set() or (abort_signal and abort_signal.triggered):
                if not self._draining:
                    # First empty read after stop — sleep one more interval and retry
                    # to capture log bytes written by cleanup / END_RUN handlers that
                    # run just after the stop signal fires.
                    self._draining = True
                    time.sleep(self.poll_interval)
                    continue
                # Second consecutive empty read after stop — nothing more to drain.
                self.eof = True
                break

            # Send a heartbeat if we have been idle longer than liveness_interval.
            # This lets the receiver distinguish "no new logs yet" from "sender died".
            if time.time() - self._last_send_time >= self._liveness_interval:
                self._last_send_time = time.time()
                self._last_was_data = False
                hb = Shareable()
                hb[KEY_HEARTBEAT] = True
                hb[KEY_DATA] = None
                hb[KEY_DATA_SIZE] = 0
                hb[KEY_EOF] = False
                return hb, self.chunk_timeout

            time.sleep(self.poll_interval)

        # Signal end-of-stream to receivers.
        result = Shareable()
        result[KEY_DATA] = None
        result[KEY_DATA_SIZE] = 0
        result[KEY_EOF] = True
        return result, self.chunk_timeout

    def close(self):
        if self.file:
            self.file.close()
            self.file = None

    def process_replies(self, replies, stream_ctx, fl_ctx):
        # Tolerate exactly one bootstrap miss at job startup. The receiver
        # registers its handler on START_RUN in the server's job subprocess;
        # if the client's first chunk arrives before that handler is wired
        # up, the server replies EXECUTION_EXCEPTION. Without this guard the
        # base class logs an ERROR and tears the stream down for the rest of
        # the job. Instead, until we see the first OK reply, treat a uniform
        # EXECUTION_EXCEPTION as transient: roll the file offset back so the
        # bytes are re-emitted on the next produce(), log at debug, and keep
        # the stream alive. After the first OK reply, fall through to the
        # base behavior so genuine receiver failures are still surfaced.
        if not self._first_ok_received and replies:
            transient = all(
                reply.get_return_code(ReturnCode.OK) == ReturnCode.EXECUTION_EXCEPTION for reply in replies.values()
            )
            if transient:
                if self._last_was_data and self.file is not None and self._last_data_offset is not None:
                    try:
                        self.file.seek(self._last_data_offset)
                    except OSError as e:
                        self.logger.warning(f"could not seek back after bootstrap miss: {e}")
                self.logger.debug(
                    f"transient bootstrap miss on first chunk to {list(replies)}; rolling back and retrying"
                )
                return None  # keep producing — receiver should be ready by next send

        if any(reply.get_return_code(ReturnCode.OK) == ReturnCode.OK for reply in replies.values()):
            self._first_ok_received = True

        return super().process_replies(replies, stream_ctx, fl_ctx)


class LogStreamer(StreamerBase):
    @staticmethod
    def register_stream_processing(
        fl_ctx: FLContext,
        channel: str,
        topic: str,
        chunk_received_cb=None,
        stream_done_cb=None,
        idle_timeout: float = 30.0,
        **cb_kwargs,
    ):
        """Register for live log stream processing on the receiving side.

        Args:
            fl_ctx: the FLContext object
            channel: the app channel
            topic: the app topic
            chunk_received_cb: called for each received data chunk (heartbeats are
                silently absorbed and never forwarded):
                ``chunk_received_cb(data: bytes, stream_ctx: StreamContext, fl_ctx: FLContext, **cb_kwargs)``
            stream_done_cb: called when the stream ends (normal EOF, engine abort, or
                idle timeout); follows ``stream_done_cb_signature`` in
                ``nvflare.apis.streaming``.  The ``stream_ctx`` passed to this callback
                will contain ``StreamContextKey.RC = ReturnCode.TIMEOUT`` when the call
                is triggered by the idle-timeout watchdog.
            idle_timeout: seconds without any message (data or heartbeat) before the
                receiver declares the sender dead and closes the stream (default 30.0).
                Set to 0 to disable.
            **cb_kwargs: kwargs forwarded to both callbacks

        Returns: None

        Notes:
            ``stream_done_cb`` is guaranteed to be called at most once per stream even
            when both the idle-timeout path and the normal engine completion path race.
        """
        engine = fl_ctx.get_engine()
        if not isinstance(engine, StreamableEngine):
            raise RuntimeError(f"engine must be StreamableEngine but got {type(engine)}")

        engine.register_stream_processing(
            channel=channel,
            topic=topic,
            factory=LogChunkConsumerFactory(chunk_received_cb, idle_timeout, stream_done_cb, cb_kwargs),
            stream_done_cb=dispatch_stream_done if stream_done_cb else None,
            **cb_kwargs,
        )

    @staticmethod
    def stream_log(
        channel: str,
        topic: str,
        stream_ctx: StreamContext,
        targets: List[str],
        file_name: str,
        fl_ctx: FLContext,
        stop_event: threading.Event = None,
        poll_interval: float = 0.5,
        liveness_interval: float = 10.0,
        idle_timeout: float = None,
        chunk_size: int = None,
        chunk_timeout: float = None,
        optional: bool = False,
        secure: bool = False,
    ):
        """Tail and stream a live log file to one or more targets.

        Continuously reads new data appended to *file_name* (including across log
        rotations) and streams it to *targets*.  Blocks until *stop_event* is set
        **and** all buffered data has been flushed, or until the run is aborted.

        **Log rotation** is detected by comparing the file's inode after each poll.
        When a rotation is found the old file handle is closed and the new file is
        opened from the beginning.  Truncation (size decreased below the current read
        position) is treated the same way.

        **Liveness heartbeats** — when no new data has been sent for *liveness_interval*
        seconds a lightweight heartbeat message (no payload) is sent to each target.
        This lets the receiver distinguish "log is quiet" from "sender process died".
        The receiver's idle-timeout clock is reset on every message, including
        heartbeats, so the timeout only fires when the sender is genuinely unreachable.

        If the file does not exist when ``stream_log`` is called the producer waits,
        polling every *poll_interval* seconds until it appears.

        Args:
            channel: the app channel
            topic: the app topic
            stream_ctx: context data for this stream
            targets: receiving site names
            file_name: full path of the log file to tail
            fl_ctx: a FLContext object
            stop_event: a :class:`threading.Event` used to signal the streamer to
                finish.  When set, the streamer drains any remaining unread bytes and
                then sends EOF.  If *None* a new Event is created; the only way to stop
                in that case is via the run abort signal.
            poll_interval: seconds to wait between polls when no new data is available
                (default 0.5)
            liveness_interval: seconds of log silence before sending a heartbeat to
                receivers (default 10.0)
            idle_timeout: optional receiver idle-timeout value used only for local
                validation. When provided and greater than zero,
                ``liveness_interval`` must be strictly less than ``idle_timeout``.
            chunk_size: bytes per chunk; defaults to 64 KB
            chunk_timeout: per-chunk send timeout in seconds; defaults to 5.0
            optional: whether the stream is optional
            secure: whether P2P security is required

        Returns: result from ``engine.stream_objects`` — same shape as
            :meth:`FileStreamer.stream_file`
        """
        if not chunk_size:
            chunk_size = 64 * 1024
        check_positive_int("chunk_size", chunk_size)

        if not chunk_timeout:
            chunk_timeout = 5.0
        check_positive_number("chunk_timeout", chunk_timeout)

        check_positive_number("poll_interval", poll_interval)
        check_positive_number("liveness_interval", liveness_interval)
        if idle_timeout is not None:
            check_non_negative_number("idle_timeout", idle_timeout)
            if idle_timeout > 0 and liveness_interval >= idle_timeout:
                raise ValueError(
                    f"liveness_interval ({liveness_interval}s) must be less than idle_timeout ({idle_timeout}s)"
                )

        if stop_event is None:
            stop_event = threading.Event()

        engine = fl_ctx.get_engine()
        if not isinstance(engine, StreamableEngine):
            raise RuntimeError(f"engine must be StreamableEngine but got {type(engine)}")

        if not stream_ctx:
            stream_ctx = {}
        stream_ctx[KEY_FILE_NAME] = os.path.basename(file_name)

        producer = _LogTailProducer(file_name, chunk_size, chunk_timeout, poll_interval, stop_event, liveness_interval)
        try:
            return engine.stream_objects(
                channel=channel,
                topic=topic,
                stream_ctx=stream_ctx,
                targets=targets,
                producer=producer,
                fl_ctx=fl_ctx,
                optional=optional,
                secure=secure,
            )
        finally:
            producer.close()

    @staticmethod
    def get_file_name(stream_ctx: StreamContext):
        """Get the source log file's base name from the stream context.

        Intended for use inside ``chunk_received_cb`` or ``stream_done_cb`` on the
        receiving side.

        Args:
            stream_ctx: the stream context

        Returns: file base name string, or None
        """
        return stream_ctx.get(KEY_FILE_NAME)
