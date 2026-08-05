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
import secrets
import signal as os_signal
import subprocess
import tempfile
import threading
import time
import uuid
from typing import Any, Dict, List

from nvflare.apis.signal import Signal
from nvflare.collab.api.call_opt import DEFAULT_CALL_TIMEOUT
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.cellnet.utils import make_reply, new_cell_message
from nvflare.fuel.f3.message import Message
from nvflare.fuel.utils import fobs
from nvflare.security.logging import secure_format_exception
from nvflare.utils.job_launcher_utils import add_custom_dir_to_path
from nvflare.utils.process_utils import log_subprocess_output, prepare_subprocess_command

from .defs import (
    DIST_CHANNEL,
    MSG_CHANNEL,
    MSG_TOPIC,
    DistributedKey,
    DistributedTopic,
    ObjectCallKey,
    decode_message,
    encode_message,
)
from .dispatch import _error_reply

_PROTOCOL_VERSION = 1
_READY_POLL_INTERVAL = 0.1
_PROCESS_POLL_INTERVAL = 0.1
_PROCESS_STOP_GRACE = 5.0
_LOG_THREAD_JOIN_TIMEOUT = 5.0
_WORKER_LEAF_PREFIX = "collab_worker"


class DistributedClientSession:
    """Own a Collab rank group and its private rank-zero Cell connection."""

    def __init__(self, command: str, startup_timeout: float, shutdown_timeout: float, logger):
        self.command = command
        self.startup_timeout = startup_timeout
        self.shutdown_timeout = shutdown_timeout
        self.logger = logger
        self.fl_ctx = None
        self.abort_signal = None
        self.cell = None
        self.parent_fqcn = None
        self.worker_fqcn = None
        self.protocol_id = uuid.uuid4().hex
        self.auth_token = secrets.token_urlsafe(32)
        self.session_id = None
        self.allowed_targets = set()
        self.world_size = None
        self.bootstrap_path = None
        self.process = None
        self.process_group_id = None
        self.log_thread = None
        self.ready = threading.Event()
        self.ready_error = None
        self.failed_reason = None
        self.call_lock = threading.Lock()
        self.stop_lock = threading.Lock()
        self.started = False
        self.stopped = False
        self.closed = False

    def start(
        self,
        *,
        fl_ctx,
        client_obj_id: str,
        collab_obj_ids: List[str],
        props: Dict[str, Any],
        server_spec: dict,
        client_specs: List[dict],
        abort_signal: Signal,
    ) -> None:
        self.fl_ctx = fl_ctx
        self.abort_signal = abort_signal
        self.cell = fl_ctx.get_engine().get_cell()
        if self.cell is None:
            raise RuntimeError("distributed Collab execution requires the site Cell")
        self.parent_fqcn = self.cell.get_fqcn()
        self.worker_fqcn = FQCN.join([self.parent_fqcn, f"{_WORKER_LEAF_PREFIX}_{self.protocol_id}"])
        self.allowed_targets = {server_spec[DistributedKey.TARGET]}
        self.allowed_targets.update(spec[DistributedKey.TARGET] for spec in client_specs)

        workspace = fl_ctx.get_workspace()
        job_id = fl_ctx.get_job_id()
        client_name = fl_ctx.get_identity_name()
        self.cell.make_internal_listener()
        parent_url = self.cell.get_internal_listener_url()
        if not parent_url:
            raise RuntimeError("site Cell has no internal listener for the distributed Collab worker")

        self.bootstrap_path = os.path.join(
            workspace.get_run_dir(job_id), f".collab_distributed_{self.protocol_id}.fobs"
        )
        self._write_bootstrap(
            self.bootstrap_path,
            {
                "workspace": workspace.get_root_dir(),
                "client_name": client_name,
                "job_id": job_id,
                "client_config": workspace.get_client_app_config_file_path(job_id),
                "client_obj_id": client_obj_id,
                "collab_obj_ids": list(collab_obj_ids or []),
                "props": props or {},
                "server": server_spec,
                "clients": client_specs,
                DistributedKey.PROTOCOL_VERSION: _PROTOCOL_VERSION,
                DistributedKey.PROTOCOL_ID: self.protocol_id,
                DistributedKey.AUTH_TOKEN: self.auth_token,
                DistributedKey.PARENT_FQCN: self.parent_fqcn,
                DistributedKey.PARENT_URL: parent_url,
                DistributedKey.WORKER_FQCN: self.worker_fqcn,
                DistributedKey.SECURE_SUPPORTED: self.cell.core_cell.supports_secure_messages(),
                DistributedKey.STARTUP_TIMEOUT: self.startup_timeout,
            },
        )

        self._register_callbacks()
        try:
            self._launch_worker(workspace, client_name, job_id)
            self._wait_until_ready()
            if self.ready_error:
                raise RuntimeError(self.ready_error)
            self.started = True
            self.logger.info(f"distributed Collab client is ready with world_size={self.world_size}")
        except BaseException:
            self.stop(finalize=False)
            raise

    def _topic(self, topic: str) -> str:
        return f"{self.protocol_id}:{topic}"

    @staticmethod
    def _write_bootstrap(path: str, data: dict) -> None:
        """Atomically write an owner-only bootstrap containing the session proof."""
        directory = os.path.dirname(os.path.abspath(path))
        fd, temp_path = tempfile.mkstemp(dir=directory, prefix=".collab-distributed-", suffix=".tmp")
        fd_owned = True
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as stream:
                fd_owned = False
                fobs.dump(data, stream)
            os.replace(temp_path, path)
        except BaseException:
            if fd_owned:
                try:
                    os.close(fd)
                except OSError:
                    pass
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass
            raise

    def _register_callbacks(self) -> None:
        self.cell.register_request_cb(
            channel=DIST_CHANNEL,
            topic=self._topic(DistributedTopic.HELLO),
            cb=self._handle_hello,
        )
        self.cell.register_request_cb(
            channel=DIST_CHANNEL,
            topic=self._topic(DistributedTopic.READY),
            cb=self._handle_ready,
        )
        self.cell.register_request_cb(
            channel=DIST_CHANNEL,
            topic=self._topic(DistributedTopic.OUTBOUND),
            cb=self._handle_outbound,
        )

    def _worker_command(self, workspace, client_name: str, job_id: str) -> list:
        command = prepare_subprocess_command(self.command)
        command.extend(
            [
                "-m",
                "nvflare.collab.runtime.distributed_worker",
                "--collab-bootstrap",
                self.bootstrap_path,
                "--workspace",
                workspace.get_root_dir(),
                "--client-name",
                client_name,
                "--job-id",
                job_id,
                "--startup-timeout",
                str(self.startup_timeout),
            ]
        )
        return command

    def _launch_worker(self, workspace, client_name: str, job_id: str) -> None:
        env = os.environ.copy()
        add_custom_dir_to_path(workspace.get_app_custom_dir(job_id), env)
        self.logger.info("launching distributed Collab worker")
        process = subprocess.Popen(
            self._worker_command(workspace, client_name, job_id),
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=workspace.get_app_dir(job_id),
            env=env,
            start_new_session=(os.name == "posix"),
        )
        self.process = process
        if os.name == "posix":
            self.process_group_id = process.pid
        self.log_thread = threading.Thread(
            target=log_subprocess_output,
            args=(process, self.logger),
            name=f"collab_worker_log_{self.protocol_id[:8]}",
            daemon=True,
        )
        self.log_thread.start()

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        while not self.ready.wait(_READY_POLL_INTERVAL):
            if self.abort_signal.triggered:
                raise RuntimeError("distributed Collab startup was aborted")
            if not self._process_group_alive():
                rc = self.process.poll() if self.process else None
                raise RuntimeError(f"distributed Collab worker exited before READY (rc={rc})")
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"distributed Collab worker did not become ready within {self.startup_timeout} seconds"
                )

    def _handle_hello(self, request: Message) -> Message:
        if self.closed:
            return make_reply(ReturnCode.INVALID_REQUEST, error="distributed Collab session is closed")
        payload = request.payload
        if not isinstance(payload, dict):
            return make_reply(ReturnCode.INVALID_REQUEST, error="HELLO payload must be a dict")
        origin = request.get_header(MessageHeaderKey.ORIGIN) or ""
        if origin != self.worker_fqcn:
            return make_reply(ReturnCode.INVALID_REQUEST, error=f"unexpected worker origin {origin!r}")
        proof = payload.get(DistributedKey.AUTH_TOKEN)
        if not isinstance(proof, str) or not secrets.compare_digest(proof, self.auth_token):
            return make_reply(ReturnCode.INVALID_REQUEST, error="distributed worker authentication failed")
        if payload.get(DistributedKey.PROTOCOL_VERSION) != _PROTOCOL_VERSION:
            return make_reply(ReturnCode.INVALID_REQUEST, error="distributed worker protocol version mismatch")
        if self.session_id is None:
            self.session_id = uuid.uuid4().hex
        return make_reply(ReturnCode.OK, body={DistributedKey.SESSION_ID: self.session_id})

    def _validate_worker_request(self, request: Message):
        if self.closed:
            return None, "distributed Collab session is closed"
        payload = request.payload
        if not isinstance(payload, dict):
            return None, "request payload must be a dict"
        origin = request.get_header(MessageHeaderKey.ORIGIN) or ""
        if origin != self.worker_fqcn:
            return None, f"unexpected worker origin {origin!r}"
        if not self.session_id or payload.get(DistributedKey.SESSION_ID) != self.session_id:
            return None, "stale or unknown distributed Collab session"
        return payload, None

    def _handle_ready(self, request: Message) -> Message:
        payload, error = self._validate_worker_request(request)
        if error:
            return make_reply(ReturnCode.INVALID_REQUEST, error=error)
        world_size = payload.get(DistributedKey.WORLD_SIZE)
        if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size < 1:
            self.ready_error = f"invalid distributed world size {world_size!r}"
        elif not payload.get(DistributedKey.OK, False):
            self.ready_error = payload.get(DistributedKey.ERROR) or "distributed Collab initialization failed"
        else:
            self.world_size = world_size
        self.ready.set()
        return make_reply(ReturnCode.OK)

    def invoke(self, request: Message) -> Message:
        payload = request.payload if isinstance(request.payload, dict) else {}
        timeout = payload.get(ObjectCallKey.TIMEOUT, DEFAULT_CALL_TIMEOUT)
        with self.call_lock:
            if not self.started or self.stopped or self.failed_reason:
                reason = self.failed_reason or "distributed Collab client is not running"
                return _error_reply(reason, self.logger)
            reply = self._send_worker_request(DistributedTopic.INVOKE, request, timeout, self.abort_signal)
            rc = reply.get_header(MessageHeaderKey.RETURN_CODE, ReturnCode.OK)
            if self.abort_signal.triggered or rc in (
                ReturnCode.TIMEOUT,
                ReturnCode.INVALID_TARGET,
                ReturnCode.TARGET_UNREACHABLE,
                ReturnCode.COMM_ERROR,
                ReturnCode.AUTHENTICATION_ERROR,
                ReturnCode.SERVICE_UNAVAILABLE,
                ReturnCode.INVALID_SESSION,
                ReturnCode.ABORT_RUN,
                ReturnCode.UNAUTHENTICATED,
            ):
                self.failed_reason = (
                    "distributed Collab client session failed after an interrupted invocation " f"(return_code={rc})"
                )
            return reply

    def _send_worker_request(self, topic: str, payload, timeout: float, abort_signal: Signal = None) -> Message:
        body = {
            DistributedKey.SESSION_ID: self.session_id,
            DistributedKey.PAYLOAD: encode_message(payload) if isinstance(payload, Message) else payload,
        }
        try:
            reply = self.cell.send_request(
                channel=DIST_CHANNEL,
                topic=self._topic(topic),
                target=self.worker_fqcn,
                request=new_cell_message({}, body),
                timeout=timeout,
                abort_signal=abort_signal,
            )
        except Exception as ex:
            if topic == DistributedTopic.INVOKE:
                self.failed_reason = (
                    "distributed Collab client session failed after an interrupted invocation: "
                    f"{secure_format_exception(ex)}"
                )
            return _error_reply(
                f"distributed Collab {topic} failed: {secure_format_exception(ex)}",
                self.logger,
                error_type=type(ex).__name__,
            )
        if not isinstance(reply, Message):
            if topic == DistributedTopic.INVOKE:
                self.failed_reason = "distributed Collab client session failed because its worker returned no Message"
            return _error_reply(f"distributed Collab {topic} returned no Message", self.logger)
        return reply

    def _handle_outbound(self, request: Message) -> Message:
        payload, error = self._validate_worker_request(request)
        if error:
            return make_reply(ReturnCode.INVALID_REQUEST, error=error)
        target = payload.get(DistributedKey.TARGET)
        if target not in self.allowed_targets:
            return _error_reply(f"outbound Collab target is not allowed: {target}", self.logger)
        if target == self.parent_fqcn:
            return _error_reply("a distributed Collab client cannot invoke its own site proxy", self.logger)
        try:
            target_request = decode_message(payload.get(DistributedKey.REQUEST))
        except (TypeError, ValueError):
            return _error_reply("outbound Collab request must be a Message", self.logger)

        timeout = payload.get(DistributedKey.TIMEOUT, DEFAULT_CALL_TIMEOUT)
        secure = bool(payload.get(DistributedKey.SECURE, False))
        optional = bool(payload.get(DistributedKey.OPTIONAL, False))
        try:
            if payload.get(DistributedKey.EXPECT_RESULT, True):
                return self.cell.send_request(
                    channel=MSG_CHANNEL,
                    target=target,
                    topic=MSG_TOPIC,
                    request=target_request,
                    timeout=timeout,
                    secure=secure,
                    optional=optional,
                    abort_signal=self.abort_signal,
                )
            self.cell.fire_and_forget(
                channel=MSG_CHANNEL,
                topic=MSG_TOPIC,
                targets=target,
                message=target_request,
                secure=secure,
                optional=optional,
            )
            return make_reply(ReturnCode.OK)
        except Exception as ex:
            return _error_reply(
                f"outbound Collab relay failed: {secure_format_exception(ex)}",
                self.logger,
                error_type=type(ex).__name__,
            )

    def stop(self, finalize: bool = True) -> None:
        with self.stop_lock:
            if self.stopped:
                return
            self.stopped = True
            try:
                if finalize and self.started:
                    with self.call_lock:
                        reply = self._send_worker_request(
                            DistributedTopic.FINALIZE,
                            None,
                            self.shutdown_timeout,
                            abort_signal=Signal(),
                        )
                    if reply.get_header(MessageHeaderKey.RETURN_CODE, ReturnCode.OK) != ReturnCode.OK:
                        self.logger.error(f"distributed Collab finalization failed: {reply.payload}")
            except Exception as ex:
                self.logger.error(f"error while finalizing distributed Collab client: {secure_format_exception(ex)}")
            finally:
                self._close_worker()
                self._wait_or_terminate_worker()
                self.closed = True
                self.auth_token = ""
                self.session_id = None
                if self.bootstrap_path:
                    try:
                        os.remove(self.bootstrap_path)
                    except OSError:
                        pass

    def _close_worker(self) -> None:
        if self.cell is None or not self.worker_fqcn or not self.session_id:
            return
        try:
            self.cell.send_request(
                channel=DIST_CHANNEL,
                topic=self._topic(DistributedTopic.CLOSE),
                target=self.worker_fqcn,
                request=new_cell_message({}, {DistributedKey.SESSION_ID: self.session_id}),
                timeout=min(_PROCESS_STOP_GRACE, self.shutdown_timeout),
                optional=True,
            )
        except Exception as ex:
            self.logger.debug(f"failed to close distributed Collab worker cleanly: {ex}")

    def _process_group_alive(self) -> bool:
        if self.process is None:
            return False
        # Reap an exited launcher before probing its process group. The group may
        # still contain torchrun workers even after the launcher has exited.
        self.process.poll()
        if os.name == "posix" and self.process_group_id is not None:
            try:
                os.killpg(self.process_group_id, 0)
                return True
            except ProcessLookupError:
                return False
            except PermissionError:
                return True
        return self.process.poll() is None

    def _wait_for_group_exit(self, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        while self._process_group_alive():
            if time.monotonic() >= deadline:
                return False
            time.sleep(_PROCESS_POLL_INTERVAL)
        if self.process is not None:
            try:
                self.process.wait(timeout=0)
            except (subprocess.TimeoutExpired, ChildProcessError):
                pass
        return True

    def _signal_worker(self, hard: bool) -> None:
        if not self._process_group_alive():
            return
        if os.name == "posix" and self.process_group_id is not None:
            try:
                os.killpg(self.process_group_id, os_signal.SIGKILL if hard else os_signal.SIGTERM)
                return
            except ProcessLookupError:
                return
            except PermissionError:
                pass
        if self.process is not None:
            self.process.kill() if hard else self.process.terminate()

    def _wait_or_terminate_worker(self) -> None:
        if self.process is None:
            return
        if not self._wait_for_group_exit(self.shutdown_timeout):
            self.logger.info("terminating distributed Collab worker process group")
            self._signal_worker(hard=False)
            if not self._wait_for_group_exit(min(_PROCESS_STOP_GRACE, self.shutdown_timeout)):
                self.logger.warning("distributed Collab worker survived SIGTERM; killing")
                self._signal_worker(hard=True)
                self._wait_for_group_exit(_LOG_THREAD_JOIN_TIMEOUT)
        if self.log_thread is not None and self.log_thread.is_alive():
            self.log_thread.join(timeout=_LOG_THREAD_JOIN_TIMEOUT)
