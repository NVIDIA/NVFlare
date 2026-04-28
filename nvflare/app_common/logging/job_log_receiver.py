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
import os
import shutil
import tempfile

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReturnCode, SystemComponents, WorkspaceConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.storage import DataTypes
from nvflare.apis.streaming import StreamContext
from nvflare.app_common.logging.constants import ALLOW_LOG_STREAMING_VAR, LIVE_LOG_TOPIC, Channels
from nvflare.app_common.streamers.log_streamer import LogStreamer
from nvflare.widgets.widget import Widget

# Keys for per-stream state stored in StreamContext
_KEY_RECV_FILE = "JobLogReceiver.recv_file"
_KEY_RECV_PATH = "JobLogReceiver.recv_path"


class JobLogReceiver(Widget):
    """Receives live log data streamed by :class:`JobLogStreamer`.

    ``JobLogReceiver`` accepts a live stream: each chunk is written directly to
    its final file as it arrives so that the log can be followed with
    ``tail -f`` on the server while the job runs. When the stream closes
    (normal EOF, job abort, or idle timeout) the file is handed to the job
    manager for storage.

    The destination file is written to ``{dest_dir}/{job_id}/{client_name}/{log_file_name}``,
    making it easy to locate and tail during a run.

    This widget can be placed in either of two locations:

    **Job-level configuration** (``config_fed_server.json``)
        Add it via ``job.to_server(JobLogReceiver())`` in the Job API, or
        declare it in the job's server config.  In this mode the handler is
        registered on ``START_RUN``, which fires when the job begins.  The
        widget is only active for that specific job.

    **System-level resources** (``resources.json`` on the server)
        Declare it as a system component so it is instantiated when the server
        process starts.  In this mode the handler is registered on
        ``SYSTEM_START`` and remains active for every job that runs on that
        server for the lifetime of the process.

    Regardless of placement, the stream handler is registered exactly once.

    Args:
        dest_dir: directory where incoming log files are written.
            Defaults to the system temporary directory.
        idle_timeout: seconds without any message (data or heartbeat) before
            the receiver declares the sender dead and closes the stream
            (default 30.0).  Set to 0 to disable.
    """

    def __init__(self, dest_dir: str = None, idle_timeout: float = 30.0):
        super().__init__()
        self._dest_dir = dest_dir
        self._idle_timeout = idle_timeout
        self._registered = False
        # Tracks (client, job_id) pairs we've already logged a "site does not allow
        # streaming" error for, so the warning is emitted at most once per job.
        self._unauthorized_logged: set = set()
        self.register_event_handler([EventType.SYSTEM_START, EventType.START_RUN], self._register)

    def _effective_dest_dir(self) -> str:
        return self._dest_dir or tempfile.gettempdir()

    @staticmethod
    def _sanitize_path_component(name: str) -> str:
        """Strip path separators and traversal sequences from a single path component."""
        # Use only the base name to prevent directory traversal via '/' or '..'
        return os.path.basename(name) if name else "unknown"

    @classmethod
    def _storage_data_type(cls, log_file_name: str) -> str:
        log_file_name = cls._sanitize_path_component(log_file_name) or WorkspaceConstants.LOG_FILE_NAME
        if log_file_name == WorkspaceConstants.ERROR_LOG_FILE_NAME:
            return DataTypes.ERRORLOG.value
        return f"{DataTypes.LOG.value}_{log_file_name}"

    def _get_trusted_stream_identity(self, fl_ctx: FLContext):
        peer_ctx = fl_ctx.get_peer_context()
        if peer_ctx is None:
            return "unknown", "unknown"

        client = self._sanitize_path_component(peer_ctx.get_identity_name(default="unknown"))
        job_id = self._sanitize_path_component(peer_ctx.get_job_id(default="unknown"))
        return client, job_id

    def _is_site_allowed_to_stream(self, client_name: str, fl_ctx: FLContext) -> bool:
        """Look up the registered Client and check its site_config (forwarded
        from the client's resources.json during registration).

        Default-allow: only an explicit ``allow_log_streaming=False`` in the
        site_config disables streaming. Missing fields, unknown clients, and
        unavailable engine all resolve to allowed.
        """
        engine = fl_ctx.get_engine()
        if engine is None:
            return True
        getter = getattr(engine, "get_client_from_name", None)
        if getter is None:
            return True
        client = getter(client_name)
        if client is None:
            return True
        site_config = client.get_site_config() or {}
        return bool(site_config.get(ALLOW_LOG_STREAMING_VAR, True))

    def _on_chunk_received(self, data: bytes, stream_ctx: StreamContext, fl_ctx: FLContext):
        f = stream_ctx.get(_KEY_RECV_FILE)
        if f is None:
            client, job_id = self._get_trusted_stream_identity(fl_ctx)
            if not self._is_site_allowed_to_stream(client, fl_ctx):
                # Drop the chunk: the site has explicitly disabled streaming.
                # Log once per (client, job_id) so the operator can see it
                # without flooding the server log.
                key = (client, job_id)
                if key not in self._unauthorized_logged:
                    self._unauthorized_logged.add(key)
                    self.log_error(
                        fl_ctx,
                        f"Dropping live log chunk from {client} for job {job_id}: site has "
                        f"'{ALLOW_LOG_STREAMING_VAR}' disabled in its resources.json",
                    )
                return
            log_file_name = self._sanitize_path_component(LogStreamer.get_file_name(stream_ctx) or "log.txt")
            path = os.path.join(self._effective_dest_dir(), job_id, client, log_file_name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.log_debug(fl_ctx, f"Opening log file for {client} job {job_id}: {path}")
            f = open(path, "wb")
            stream_ctx[_KEY_RECV_FILE] = f
            stream_ctx[_KEY_RECV_PATH] = path
        f.write(data)
        f.flush()

    def _on_stream_done(self, stream_ctx: StreamContext, fl_ctx: FLContext):
        f = stream_ctx.get(_KEY_RECV_FILE)
        if f is not None:
            f.close()
            stream_ctx[_KEY_RECV_FILE] = None

        rc = LogStreamer.get_rc(stream_ctx)
        client, job_id = self._get_trusted_stream_identity(fl_ctx)

        if rc != ReturnCode.OK:
            file_path = stream_ctx.get(_KEY_RECV_PATH)
            self.log_warning(
                fl_ctx,
                f"Live log stream from {client} job {job_id} ended with rc={rc}; partial log retained at {file_path}",
            )
            return

        file_path = stream_ctx.get(_KEY_RECV_PATH)
        if not file_path:
            self.log_warning(fl_ctx, f"No log data received from {client} for job {job_id}")
            return

        log_type = LogStreamer.get_file_name(stream_ctx)
        engine = fl_ctx.get_engine()
        job_manager = engine.get_component(SystemComponents.JOB_MANAGER)
        if job_manager is None:
            # No job manager (e.g. simulator): move file from temp staging dir to the
            # job's workspace run directory so it lives alongside other job artifacts.
            if self._dest_dir is None:
                workspace = getattr(engine, "get_workspace", lambda: None)()
                if workspace is not None:
                    dest_path = os.path.join(workspace.get_run_dir(job_id), client, log_type)
                    try:
                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        shutil.move(file_path, dest_path)
                        self.log_info(fl_ctx, f"Saved live log '{log_type}' from {client} to {dest_path}")
                        return
                    except Exception:
                        self.log_exception(fl_ctx, f"Failed to move live log to workspace; retained at {file_path}")
                        return
            self.log_info(fl_ctx, f"Live log '{log_type}' from {client} retained at {file_path}")
            return
        data_type = self._storage_data_type(log_type)
        self.log_info(
            fl_ctx, f"Saving live log '{log_type}' as '{data_type}' from {client} for job {job_id}: {file_path}"
        )
        job_manager.set_client_data(job_id, file_path, client, data_type, fl_ctx)

    def _register(self, event_type: str, fl_ctx: FLContext):
        if self._registered:
            return
        self._registered = True
        LogStreamer.register_stream_processing(
            fl_ctx,
            channel=Channels.LOG_STREAMING_CHANNEL,
            topic=LIVE_LOG_TOPIC,
            chunk_received_cb=self._on_chunk_received,
            stream_done_cb=self._on_stream_done,
            idle_timeout=self._idle_timeout,
        )
