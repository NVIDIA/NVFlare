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
import threading

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, JobConstants, ProcessType, StreamCtxKey, WorkspaceConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.workspace import Workspace
from nvflare.app_common.logging.constants import LIVE_LOG_TOPIC, Channels, is_log_streaming_allowed
from nvflare.app_common.streamers.log_streamer import LogStreamer
from nvflare.security.logging import secure_format_exception
from nvflare.widgets.widget import Widget

_LOG_STREAMER_PATH = "nvflare.app_common.logging.job_log_streamer.JobLogStreamer"


class SystemLogStreamer(Widget):
    """System-level widget that injects a :class:`JobLogStreamer` into every job
    that does not already declare one.

    Place this in the client's ``resources.json`` so that live log streaming is
    provided automatically for every job — without requiring each job to include
    a ``JobLogStreamer`` in its own configuration.

    On ``BEFORE_JOB_LAUNCH`` (after the job config is deployed to disk but
    before the job subprocess starts) ``SystemLogStreamer`` reads the deployed
    ``config_fed_client.json``.  If no ``JobLogStreamer`` component is found, it
    appends one with the configured parameters and writes the file back.  The
    job subprocess then picks up the modified config and ``JobLogStreamer`` runs
    inside the job as if the user had declared it explicitly.

    When configured for ``error_log.txt``, ``SystemLogStreamer`` also uploads a
    post-run snapshot from ``CLIENT_PARENT`` on ``JOB_COMPLETED``. This preserves
    error-log delivery for launch/config/bootstrap failures where the job
    subprocess never reaches ``START_RUN`` and therefore never loads the
    injected ``JobLogStreamer``.

    The server side must have a
    :class:`~nvflare.app_common.logging.job_log_receiver.JobLogReceiver`
    in its ``resources.json`` (or job config) to receive and store the stream.

    Args:
        log_file_name: base name of the log file to stream.  Defaults to
            ``WorkspaceConstants.LOG_FILE_NAME`` (``"log.txt"``).
        liveness_interval: seconds between heartbeat messages when no new log
            bytes have been written (default 10.0).  Must be strictly less than
            the receiver's ``idle_timeout``.
        poll_interval: seconds between polls when no new data has been written
            to the log (default 0.5).
    """

    def __init__(
        self,
        log_file_name: str = WorkspaceConstants.LOG_FILE_NAME,
        liveness_interval: float = 10.0,
        poll_interval: float = 0.5,
    ):
        super().__init__()
        self._log_file_name = log_file_name
        self._liveness_interval = liveness_interval
        self._poll_interval = poll_interval
        self.register_event_handler(EventType.BEFORE_JOB_LAUNCH, self._on_before_job_launch)
        self.register_event_handler(EventType.JOB_COMPLETED, self._on_job_completed)

    def _on_before_job_launch(self, event_type: str, fl_ctx: FLContext):
        job_meta = fl_ctx.get_prop(FLContextKey.JOB_META)
        if not job_meta:
            return
        job_id = job_meta.get(JobMetaKey.JOB_ID)
        if not job_id:
            return

        workspace_root = fl_ctx.get_prop(FLContextKey.WORKSPACE_ROOT)
        client_name = fl_ctx.get_identity_name()
        if not workspace_root or not client_name:
            return

        workspace = Workspace(root_dir=workspace_root, site_name=client_name)
        config_path = os.path.join(workspace.get_app_config_dir(job_id), JobConstants.CLIENT_JOB_CONFIG)
        if not os.path.exists(config_path):
            return

        try:
            with open(config_path) as f:
                cfg = json.load(f)
        except Exception as ex:
            self.log_exception(
                fl_ctx,
                f"Failed to read {config_path}; skipping log streamer injection: {secure_format_exception(ex)}",
            )
            return

        components = cfg.get("components")
        if components is None:
            components = []
            cfg["components"] = components

        # Site-level kill switch. When the site's resources.json doesn't enable
        # log streaming, strip any pre-declared JobLogStreamer from the deployed
        # job config and skip injection.
        if not is_log_streaming_allowed(fl_ctx):
            filtered = [c for c in components if "JobLogStreamer" not in c.get("path", "")]
            if len(filtered) != len(components):
                cfg["components"] = filtered
                try:
                    with open(config_path, "w") as f:
                        json.dump(cfg, f, indent=2)
                    self.log_warning(
                        fl_ctx,
                        f"Removed JobLogStreamer from job {job_id}: site does not allow log streaming",
                    )
                except Exception as ex:
                    self.log_exception(
                        fl_ctx,
                        f"Failed to write {config_path} after stripping JobLogStreamer: "
                        f"{secure_format_exception(ex)}",
                    )
            else:
                self.log_debug(
                    fl_ctx,
                    f"Job {job_id}: site does not allow log streaming; not injecting JobLogStreamer",
                )
            return

        for c in components:
            if "JobLogStreamer" in c.get("path", ""):
                self.log_debug(fl_ctx, f"Job {job_id} already has JobLogStreamer; skipping injection")
                return

        # Build the component entry with non-default args only.
        args = {}
        if self._log_file_name != WorkspaceConstants.LOG_FILE_NAME:
            args["log_file_name"] = self._log_file_name
        if self._liveness_interval != 10.0:
            args["liveness_interval"] = self._liveness_interval
        if self._poll_interval != 0.5:
            args["poll_interval"] = self._poll_interval

        entry = {
            "id": "auto_log_streamer",
            "path": _LOG_STREAMER_PATH,
        }
        if args:
            entry["args"] = args
        components.append(entry)

        try:
            with open(config_path, "w") as f:
                json.dump(cfg, f, indent=2)
        except Exception as ex:
            self.log_exception(
                fl_ctx,
                f"Failed to write {config_path}; log streamer not injected: {secure_format_exception(ex)}",
            )
            return

        self.log_info(fl_ctx, f"Injected JobLogStreamer into job {job_id}")

    def _stream_completed_log(self, fl_ctx: FLContext, log_path: str, client_name: str, job_id: str):
        stop_event = threading.Event()
        stop_event.set()

        try:
            LogStreamer.stream_log(
                channel=Channels.LOG_STREAMING_CHANNEL,
                topic=LIVE_LOG_TOPIC,
                stream_ctx={
                    StreamCtxKey.CLIENT_NAME: client_name,
                    StreamCtxKey.JOB_ID: job_id,
                },
                targets=["server"],
                file_name=log_path,
                fl_ctx=fl_ctx,
                stop_event=stop_event,
                poll_interval=self._poll_interval,
                liveness_interval=self._liveness_interval,
            )
        except Exception as ex:
            self.log_exception(
                fl_ctx,
                f"Failed to upload completed log '{self._log_file_name}' for job {job_id}: "
                f"{secure_format_exception(ex)}",
            )

    def _on_job_completed(self, event_type: str, fl_ctx: FLContext):
        if self._log_file_name != WorkspaceConstants.ERROR_LOG_FILE_NAME:
            return
        if fl_ctx.get_process_type() != ProcessType.CLIENT_PARENT:
            return
        if not is_log_streaming_allowed(fl_ctx):
            return

        workspace_root = fl_ctx.get_prop(FLContextKey.WORKSPACE_ROOT)
        client_name = fl_ctx.get_identity_name(default="") or fl_ctx.get_prop(FLContextKey.CLIENT_NAME)
        job_id = fl_ctx.get_prop(FLContextKey.CURRENT_JOB_ID) or fl_ctx.get_job_id()
        if not workspace_root or not client_name or not job_id:
            return

        workspace = Workspace(root_dir=workspace_root, site_name=client_name)
        log_path = workspace.get_app_error_log_file_path(job_id)
        if not os.path.exists(log_path):
            self.log_info(fl_ctx, f"No error log file found for {client_name} job {job_id}")
            return

        # The event FLContext can be reused after the handler returns, so hand
        # the background uploader its own fresh context.
        engine = fl_ctx.get_engine()
        stream_fl_ctx = engine.new_context() if engine else fl_ctx
        threading.Thread(
            target=self._stream_completed_log,
            args=(stream_fl_ctx, log_path, client_name, job_id),
            daemon=True,
        ).start()
        self.log_info(fl_ctx, f"Started completed error log upload for {client_name} job {job_id}: {log_path}")
