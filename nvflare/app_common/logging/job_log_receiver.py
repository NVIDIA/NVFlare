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
import stat
import tempfile
import threading
from collections import OrderedDict
from contextlib import nullcontext

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReturnCode, SystemComponents, WorkspaceConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.storage import DataTypes
from nvflare.apis.streaming import StreamContext
from nvflare.app_common.logging.constants import (
    ALLOW_LOG_STREAMING_VAR,
    LIVE_LOG_TOPIC,
    Channels,
    client_log_path,
    client_log_staging_root,
)
from nvflare.app_common.logging.log_staging import ClientLogStaging, lock_file, private_directory
from nvflare.app_common.streamers.log_streamer import LogStreamer
from nvflare.widgets.widget import Widget

# Cap on the number of (client, job_id) pairs we remember for "log once"
# tracking. The receiver is a long-lived server-side widget; without a cap,
# the set would grow without bound across job churn. When the cap is hit,
# the oldest entry is evicted (FIFO), at the cost of possibly re-logging a
# repeat violation from a long-quiescent job.
_UNAUTHORIZED_LOG_CAP = 4096


class JobLogReceiver(Widget):
    """Receives live log data streamed by :class:`JobLogStreamer`.

    ``JobLogReceiver`` accepts a live stream: each chunk is written directly to
    a staging file as it arrives so that the log can be followed with
    ``tail -f`` on the server while the job runs. Successful streams are handed
    to the job manager for storage. Interrupted streams retain partial logs;
    explicit job deletion removes managed logs after active streams drain.
    The storage backend may consume the staging file.

    By default the staging file is written to
    ``{workspace.get_log_root()}/.client_log_streams/{job_id}/{client_name}/{log_file_name}``,
    outside per-job cleanup. The administrator's live-log reader uses this location. An explicit ``dest_dir``
    instead selects ``{dest_dir}/{job_id}/{client_name}/{log_file_name}``.

    This widget can be placed in either of two locations:

    **Job-level configuration** (``config_fed_server.json``)
        Add it via ``job.to_server(JobLogReceiver())`` in the Job API, or
        declare it in the job's server config.  In this mode the handler is
        registered on ``START_RUN`` (and on ``ABOUT_TO_START_RUN`` where
        available) so that the stream handler is wired up before any client
        chunk can arrive.  The widget is only active for that specific job.

    **System-level resources** (``resources.json`` on the server)
        Declare it as a system component so it is instantiated when the server
        process starts.  In the long-lived parent process the handler is
        registered on ``SYSTEM_START`` and remains active across jobs.  In
        per-job server subprocesses (where ``SYSTEM_START`` does not fire) the
        same instance re-registers on every ``START_RUN`` against the new
        job-scoped ``ObjectStreamer``.

    The stream handler may be (re-)registered on every triggering event;
    ``registry.set`` is idempotent for the same channel/topic pair.

    Args:
        dest_dir: directory where incoming log staging files are written.
            Defaults to private staging beside the workspace job log roots; uses
            a private temporary directory only when no workspace is available.
        idle_timeout: seconds without any message (data or heartbeat) before
            the receiver declares the sender dead and closes the stream
            (default 30.0).  Set to 0 to disable.
    """

    def __init__(self, dest_dir: str = None, idle_timeout: float = 30.0):
        super().__init__()
        self._dest_dir = dest_dir
        self._idle_timeout = idle_timeout
        # StreamContext arrives from the sender. File handles, paths, and the
        # validated owner must remain in receiver-owned state, never in it.
        self._stream_files = {}
        self._stream_lock = threading.RLock()
        self._temporary_dest = None
        # Tracks (client, job_id) pairs we've already logged a "site does not allow
        # streaming" error for, so the warning is emitted at most once per job.
        # Bounded with FIFO eviction (see _UNAUTHORIZED_LOG_CAP) and guarded by a
        # lock so the check-then-add is atomic across concurrent stream chunks.
        self._unauthorized_logged: OrderedDict = OrderedDict()
        self._unauthorized_lock = threading.Lock()
        # Trigger on every event that may bring up a fresh ObjectStreamer:
        #   - SYSTEM_START fires once in the long-lived server parent process.
        #   - ABOUT_TO_START_RUN fires only on the client side, but listing it
        #     keeps the receiver usable if it is ever placed there.
        #   - START_RUN is the only one that fires in the server-job subprocess
        #     (see ServerRunner.run) and is also the point at which an in-process
        #     deployment swaps engine.run_manager to the job's RunManager.
        # Re-registering on each trigger is safe: the underlying registry.set
        # is idempotent for the same channel/topic, and registering against a
        # new ObjectStreamer is exactly what we want when the run manager has
        # been replaced.
        self.register_event_handler(
            [EventType.SYSTEM_START, EventType.ABOUT_TO_START_RUN, EventType.START_RUN],
            self._register,
        )

    def _staging_root(self, fl_ctx):
        if self._dest_dir:
            return self._dest_dir
        workspace = getattr(fl_ctx.get_engine(), "get_workspace", lambda: None)()
        if workspace is not None:
            return client_log_staging_root(workspace)
        if self._temporary_dest is None:
            self._temporary_dest = tempfile.mkdtemp(prefix="nvflare-client-logs-")
        return self._temporary_dest

    @staticmethod
    def _open_log(root, job_id, client, file_name, leased=False):
        if leased and client in (ClientLogStaging.DELETED_MARKER, ClientLogStaging.DELETE_INTENT):
            raise ValueError("Reserved client log path component")
        path = client_log_path(root, job_id, client, file_name)
        root = os.path.realpath(root)
        for directory in (root, os.path.join(root, job_id), os.path.dirname(path)):
            private_directory(directory)
        # Recheck after creating directories; never follow a final symlink.
        client_log_path(root, job_id, client, file_name)
        if os.path.islink(path):
            raise ValueError("Client log file must not be a symlink")
        fd = os.open(
            path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
            0o600,
        )
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("Client log must be a regular file")
            if leased and not lock_file(fd):
                raise ValueError("Client log already has an active stream")
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o600)
            os.ftruncate(fd, 0)
            return path, os.fdopen(fd, "wb")
        except BaseException:
            os.close(fd)
            raise

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
        client = fl_ctx.get_prop(FLContextKey.CLIENT_NAME)
        if peer_ctx is None or not isinstance(client, str) or peer_ctx.get_identity_name() != client:
            raise ValueError("Log stream identity does not match the authenticated sender")
        return client, peer_ctx.get_job_id()

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

    def _record_unauthorized(self, key: tuple) -> bool:
        """Atomically record an unauthorized (client, job_id) pair.

        Returns True if this is the first time we've seen this key (caller
        should emit the once-per-job log line). Bounds the bookkeeping with
        FIFO eviction so the receiver can run for the lifetime of the
        server without unbounded memory growth.
        """
        with self._unauthorized_lock:
            if key in self._unauthorized_logged:
                return False
            if len(self._unauthorized_logged) >= _UNAUTHORIZED_LOG_CAP:
                self._unauthorized_logged.popitem(last=False)
            self._unauthorized_logged[key] = None
            return True

    def _on_chunk_received(self, data: bytes, stream_ctx: StreamContext, fl_ctx: FLContext):
        client, job_id = self._get_trusted_stream_identity(fl_ctx)
        with self._stream_lock:
            state = self._stream_files.get(id(stream_ctx))
            if state is None:
                if not self._is_site_allowed_to_stream(client, fl_ctx):
                    if self._record_unauthorized((client, job_id)):
                        self.log_error(
                            fl_ctx,
                            f"Dropping live log chunk from {client} for job {job_id}: site has "
                            f"'{ALLOW_LOG_STREAMING_VAR}' disabled in its resources.json",
                        )
                    return
                file_name = LogStreamer.get_file_name(stream_ctx) or WorkspaceConstants.LOG_FILE_NAME
                workspace = getattr(fl_ctx.get_engine(), "get_workspace", lambda: None)()
                staging = ClientLogStaging(workspace) if self._dest_dir is None and workspace is not None else None
                with staging.locked() if staging else nullcontext():
                    job_manager = fl_ctx.get_engine().get_component(SystemComponents.JOB_MANAGER)
                    if staging and staging.is_deleting(job_id) and job_manager is not None:
                        staging.recover_delete(job_id, job_manager, fl_ctx)
                    if staging and staging.is_deleted(job_id):
                        staging.remove_closed_job_logs(job_id)
                        raise ValueError("Cannot stream logs for a deleted or unknown job")
                    if staging and (
                        staging.is_deleting(job_id)
                        or (job_manager is not None and job_manager.get_job(job_id, fl_ctx) is None)
                    ):
                        raise ValueError("Cannot stream logs for a deleted or unknown job")
                    path, stream = self._open_log(
                        self._staging_root(fl_ctx), job_id, client, file_name, leased=staging is not None
                    )
                state = {
                    "file": stream,
                    "path": path,
                    "client": client,
                    "job_id": job_id,
                    "file_name": file_name,
                    "staging": staging,
                }
                self._stream_files[id(stream_ctx)] = state
            elif (state["client"], state["job_id"]) != (client, job_id):
                raise ValueError("Log stream owner changed")
            state["file"].write(data)
            state["file"].flush()

    def _on_stream_done(self, stream_ctx: StreamContext, fl_ctx: FLContext):
        with self._stream_lock:
            state = self._stream_files.pop(id(stream_ctx), None)
        if state is None:
            if LogStreamer.get_file_name(stream_ctx) == WorkspaceConstants.ERROR_LOG_FILE_NAME:
                return
            client, job_id = self._get_trusted_stream_identity(fl_ctx)
            self.log_warning(fl_ctx, f"No log data received from {client} for job {job_id}")
            return
        staging = state["staging"]
        with staging.locked() if staging else nullcontext():
            state["file"].close()
            job_manager = fl_ctx.get_engine().get_component(SystemComponents.JOB_MANAGER)
            if staging and staging.is_deleting(state["job_id"]):
                try:
                    if job_manager is None:
                        raise RuntimeError("Job manager unavailable for deletion recovery")
                    staging.recover_delete(state["job_id"], job_manager, fl_ctx)
                except Exception:
                    self.log_exception(fl_ctx, "Cannot confirm interrupted job deletion; partial log retained")
                    return
            if staging and staging.is_deleted(state["job_id"]):
                staging.remove_closed_job_logs(state["job_id"])
                return
            if staging and job_manager is not None and job_manager.get_job(state["job_id"], fl_ctx) is None:
                # Missing metadata can also mean a temporary storage failure.
                # Only an explicit successful deletion authorizes removing logs.
                self.log_warning(fl_ctx, f"Job {state['job_id']} unavailable; partial log retained at {state['path']}")
                return
            self._finish_stream(state, stream_ctx, fl_ctx)

    def _finish_stream(self, state, stream_ctx, fl_ctx):
        client, job_id, log_type = state["client"], state["job_id"], state["file_name"]
        file_path = state["path"]
        rc = LogStreamer.get_rc(stream_ctx)
        if rc != ReturnCode.OK:
            self.log_warning(
                fl_ctx,
                f"Live log stream from {client} job {job_id} ended with rc={rc}; "
                f"partial log retained at {file_path}",
            )
            return

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
                        if os.path.abspath(file_path) != os.path.abspath(dest_path):
                            shutil.move(file_path, dest_path)
                        self.log_info(fl_ctx, f"Saved live log '{log_type}' from {client} to {dest_path}")
                        return
                    except Exception:
                        self.log_exception(fl_ctx, f"Failed to move live log to workspace; retained at {file_path}")
                        return
            self.log_info(fl_ctx, f"Live log '{log_type}' from {client} retained at {file_path}")
            return
        data_type = self._storage_data_type(log_type)
        job_manager.set_client_data(job_id, file_path, client, data_type, fl_ctx)
        self.log_info(fl_ctx, f"Saved live log '{log_type}' from {client} for job {job_id}")

    def _remove_deleted_job_logs(self, fl_ctx):
        if self._dest_dir is not None:
            return
        engine = fl_ctx.get_engine()
        if engine is None:
            return
        workspace = getattr(engine, "get_workspace", lambda: None)()
        job_manager = engine.get_component(SystemComponents.JOB_MANAGER)
        if workspace is None or job_manager is None:
            return
        staging = ClientLogStaging(workspace)
        if not os.path.isdir(staging.root):
            return
        with staging.locked():
            with os.scandir(staging.root) as entries:
                for entry in entries:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    try:
                        staging.recover_delete(entry.name, job_manager, fl_ctx)
                        if staging.is_deleted(entry.name):
                            staging.remove_closed_job_logs(entry.name)
                    except Exception:
                        self.log_exception(fl_ctx, f"Cannot recover job log deletion for {entry.name}; logs retained")

    def _register(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.SYSTEM_START:
            # Recover abandoned leases after a server crash/restart. Logs for
            # existing jobs remain available, including retained partial logs.
            try:
                self._remove_deleted_job_logs(fl_ctx)
            except Exception:
                self.log_exception(fl_ctx, "Failed to clean deleted-job log staging")
        # Re-register on every triggering event. The active ObjectStreamer is
        # owned by the current run_manager and is replaced whenever the engine
        # transitions to a new run, so caching a "registered once" flag would
        # leave the new run_manager's registry empty and produce
        # "no stream processing info registered for log_streaming:live_log"
        # on the first incoming chunk.
        LogStreamer.register_stream_processing(
            fl_ctx,
            channel=Channels.LOG_STREAMING_CHANNEL,
            topic=LIVE_LOG_TOPIC,
            chunk_received_cb=self._on_chunk_received,
            stream_done_cb=self._on_stream_done,
            idle_timeout=self._idle_timeout,
        )
