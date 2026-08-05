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

import argparse
import copy
import json
import os
import threading
import time

from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.apis.workspace import Workspace
from nvflare.app_common.executors.multi_process_executor import WorkerComponentBuilder
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply, new_cell_message
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming.byte_streamer import reliable_retry_scheduler
from nvflare.fuel.f3.streaming.download_service import DownloadService
from nvflare.fuel.f3.streaming.stream_utils import stream_shutdown
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs import FOBSContextKey
from nvflare.fuel.utils.import_utils import optional_import
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.private.fed.utils.fed_utils import fobs_initialize
from nvflare.security.logging import secure_format_exception, secure_format_traceback

from .defs import DIST_CHANNEL, CallReplyKey, DistributedKey, DistributedTopic, decode_message, encode_message
from .dispatch import _call_app_method, _error_reply
from .executor import CollabExecutor

_HELLO_RETRY_INTERVAL = 0.5
_CLOSE = "close"


class _WorkerEngine:
    def __init__(self, workspace, components):
        self.workspace = workspace
        self.components = components

    def get_workspace(self):
        return self.workspace

    def get_component(self, component_id):
        return self.components.get(component_id)

    @staticmethod
    def system_panic(reason, fl_ctx):
        raise RuntimeError(reason)

    @staticmethod
    def new_context():
        return FLContext()


class _RelayCoreCell:
    def __init__(self, secure_supported: bool):
        self.secure_supported = secure_supported

    def supports_secure_messages(self):
        return self.secure_supported


class _RelayCell:
    """Cell-shaped proxy that sends rank-zero calls through the parent site Cell."""

    def __init__(
        self,
        *,
        cell,
        parent_fqcn: str,
        outbound_topic: str,
        session_id: str,
        rank: int,
        secure_supported: bool,
        abort_signal: Signal,
    ):
        self.cell = cell
        self.parent_fqcn = parent_fqcn
        self.outbound_topic = outbound_topic
        self.session_id = session_id
        self.rank = rank
        self.abort_signal = abort_signal
        self.core_cell = _RelayCoreCell(secure_supported)

    def get_fqcn(self):
        return self.cell.get_fqcn() if self.cell is not None else f"distributed-rank-{self.rank}"

    def _relay(self, target, request, timeout, secure, optional, expect_result):
        if self.rank != 0:
            raise RuntimeError("outbound Collab proxy calls from a distributed client require global rank 0")
        if self.cell is None:
            raise RuntimeError("distributed Collab rank zero has no parent Cell connection")
        return self.cell.send_request(
            channel=DIST_CHANNEL,
            topic=self.outbound_topic,
            target=self.parent_fqcn,
            request=new_cell_message(
                {},
                {
                    DistributedKey.SESSION_ID: self.session_id,
                    DistributedKey.TARGET: target,
                    DistributedKey.REQUEST: encode_message(request),
                    DistributedKey.TIMEOUT: timeout,
                    DistributedKey.SECURE: secure,
                    DistributedKey.OPTIONAL: optional,
                    DistributedKey.EXPECT_RESULT: expect_result,
                },
            ),
            timeout=timeout,
            abort_signal=self.abort_signal,
        )

    def send_request(
        self,
        *,
        channel,
        target,
        topic,
        request,
        timeout=None,
        secure=False,
        optional=False,
        abort_signal=None,
    ):
        return self._relay(target, request, timeout, secure, optional, True)

    def fire_and_forget(self, *, channel, topic, targets, message, secure=False, optional=False):
        self._relay(targets, message, 60.0, secure, optional, False)
        return {targets: ""}


class DistributedWorker:
    def __init__(self, args):
        self.args = args
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", str(self.rank)))
        self.env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.logger = get_obj_logger(self)
        self.workspace = Workspace(args.workspace, args.client_name)
        self.app = None
        self.collab_executor = None
        self.context = None
        self.abort_signal = Signal()
        self.components = {}
        self.dist = None
        self.world_size = 1
        self.finalized = False
        self.initialized = False
        self.cell = None
        self.parent_fqcn = None
        self.worker_fqcn = None
        self.protocol_id = None
        self.session_id = None
        self.secure_supported = False
        self.exit_event = threading.Event()
        self.call_lock = threading.Lock()

    def _load_bootstrap(self):
        return fobs.loadf(self.args.collab_bootstrap)

    def _topic(self, topic: str) -> str:
        return f"{self.protocol_id}:{topic}"

    def _start_cell(self, bootstrap) -> None:
        self.parent_fqcn = bootstrap[DistributedKey.PARENT_FQCN]
        self.worker_fqcn = bootstrap[DistributedKey.WORKER_FQCN]
        self.protocol_id = bootstrap[DistributedKey.PROTOCOL_ID]
        self.secure_supported = bool(bootstrap.get(DistributedKey.SECURE_SUPPORTED, False))
        self.cell = Cell(
            fqcn=self.worker_fqcn,
            root_url=None,
            secure=False,
            credentials={},
            parent_url=bootstrap[DistributedKey.PARENT_URL],
            create_internal_listener=False,
        )
        self.cell.update_fobs_context({FOBSContextKey.ABORT_SIGNAL: self.abort_signal})
        self.cell.register_request_cb(
            channel=DIST_CHANNEL,
            topic=self._topic(DistributedTopic.INVOKE),
            cb=self._handle_invoke,
        )
        self.cell.register_request_cb(
            channel=DIST_CHANNEL,
            topic=self._topic(DistributedTopic.FINALIZE),
            cb=self._handle_finalize,
        )
        self.cell.register_request_cb(
            channel=DIST_CHANNEL,
            topic=self._topic(DistributedTopic.CLOSE),
            cb=self._handle_close,
        )
        self.cell.start()

        timeout = float(bootstrap[DistributedKey.STARTUP_TIMEOUT])
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"parent Collab Cell did not accept HELLO within {timeout} seconds")
            reply = self.cell.send_request(
                channel=DIST_CHANNEL,
                topic=self._topic(DistributedTopic.HELLO),
                target=self.parent_fqcn,
                request=new_cell_message(
                    {},
                    {
                        DistributedKey.PROTOCOL_VERSION: bootstrap[DistributedKey.PROTOCOL_VERSION],
                        DistributedKey.AUTH_TOKEN: bootstrap[DistributedKey.AUTH_TOKEN],
                    },
                ),
                timeout=min(_HELLO_RETRY_INTERVAL, remaining),
            )
            if reply is not None and reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK:
                body = reply.payload
                self.session_id = body.get(DistributedKey.SESSION_ID) if isinstance(body, dict) else None
                if not self.session_id:
                    raise RuntimeError("distributed Collab HELLO reply has no session id")
                return
            time.sleep(min(_HELLO_RETRY_INTERVAL, max(0.0, deadline - time.monotonic())))

    def _send_ready(self, status) -> None:
        body = dict(status)
        body[DistributedKey.SESSION_ID] = self.session_id
        body[DistributedKey.WORLD_SIZE] = self.world_size
        reply = self.cell.send_request(
            channel=DIST_CHANNEL,
            topic=self._topic(DistributedTopic.READY),
            target=self.parent_fqcn,
            request=new_cell_message({}, body),
            timeout=float(self.args.startup_timeout),
        )
        if reply is None or reply.get_header(MessageHeaderKey.RETURN_CODE) != ReturnCode.OK:
            rc = None if reply is None else reply.get_header(MessageHeaderKey.RETURN_CODE)
            raise RuntimeError(f"parent Collab Cell rejected READY (rc={rc})")

    def _validate_parent_request(self, request: Message):
        payload = request.payload
        if not isinstance(payload, dict):
            return None, "request payload must be a dict"
        origin = request.get_header(MessageHeaderKey.ORIGIN) or ""
        if origin != self.parent_fqcn:
            return None, f"unexpected parent origin {origin!r}"
        if not self.session_id or payload.get(DistributedKey.SESSION_ID) != self.session_id:
            return None, "stale or unknown distributed Collab session"
        return payload, None

    def _handle_invoke(self, request: Message) -> Message:
        payload, error = self._validate_parent_request(request)
        if error:
            return make_reply(ReturnCode.INVALID_REQUEST, error=error)
        if not self.initialized or self.finalized:
            return _error_reply("distributed Collab worker is not available for invocation", self.logger)
        try:
            call = decode_message(payload.get(DistributedKey.PAYLOAD))
        except (TypeError, ValueError):
            return _error_reply("distributed Collab invocation must contain a Message", self.logger)
        with self.call_lock:
            self._broadcast(DistributedTopic.INVOKE, call)
            return self._invoke_all(call)

    def _handle_finalize(self, request: Message) -> Message:
        _, error = self._validate_parent_request(request)
        if error:
            return make_reply(ReturnCode.INVALID_REQUEST, error=error)
        with self.call_lock:
            if self.finalized:
                return make_reply(ReturnCode.OK)
            self._broadcast(DistributedTopic.FINALIZE)
            return self._finalize_local()

    def _handle_close(self, request: Message) -> Message:
        _, error = self._validate_parent_request(request)
        if error:
            return make_reply(ReturnCode.INVALID_REQUEST, error=error)
        if self.initialized and not self.finalized:
            self._broadcast(_CLOSE)
        # Let Cell send this acknowledgement before the main thread stops the
        # worker Cell and process-global streaming services.
        timer = threading.Timer(0.1, self.exit_event.set)
        timer.daemon = True
        timer.start()
        return make_reply(ReturnCode.OK)

    def _build_components(self, bootstrap):
        with open(bootstrap["client_config"], "r", encoding="utf-8") as f:
            app_config = json.load(f)
        entries = {entry.get("id"): (i, entry) for i, entry in enumerate(app_config.get("components", []), 1)}
        required_ids = [bootstrap["client_obj_id"]] + list(bootstrap.get("collab_obj_ids") or [])
        builder = WorkerComponentBuilder(workspace=self.workspace, enforce_authorization=False)
        for component_id in required_ids:
            found = entries.get(component_id)
            if not found:
                raise RuntimeError(f"cannot find component config for {component_id}")
            index, entry = found
            component_config = copy.deepcopy(entry)
            node = builder.make_component_node(component_config, index)
            self.components[component_id] = builder.build_component(component_config, node)

    def _build_app(self, bootstrap):
        engine = _WorkerEngine(self.workspace, self.components)
        fl_ctx = FLContext()
        fl_ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=False)
        fl_ctx.set_prop(ReservedKey.RUN_NUM, self.args.job_id, private=True, sticky=False)
        fl_ctx.set_prop(ReservedKey.IDENTITY_NAME, self.args.client_name, private=True, sticky=False)
        fl_ctx.set_prop(ReservedKey.RUN_ABORT_SIGNAL, self.abort_signal, private=True, sticky=False)
        fl_ctx.set_prop(FLContextKey.RANK_NUMBER, self.rank, private=True, sticky=False)
        fl_ctx.set_prop(FLContextKey.NUM_OF_PROCESSES, self.env_world_size, private=True, sticky=False)

        collab_executor = CollabExecutor(
            client_obj_id=bootstrap["client_obj_id"],
            collab_obj_ids=bootstrap.get("collab_obj_ids"),
            props=bootstrap.get("props"),
            max_call_threads=10,
        )
        collab_executor._handle_start_run("", fl_ctx)
        app = collab_executor.client_app
        if app is None:
            raise RuntimeError("could not construct distributed Collab client app")
        relay_cell = _RelayCell(
            cell=self.cell,
            parent_fqcn=self.parent_fqcn,
            outbound_topic=self._topic(DistributedTopic.OUTBOUND),
            session_id=self.session_id,
            rank=self.rank,
            secure_supported=self.secure_supported,
            abort_signal=self.abort_signal,
        )
        server = bootstrap["server"]
        server_proxy = collab_executor._prepare_proxy(
            server["name"],
            server["fqn"],
            server[DistributedKey.TARGET],
            relay_cell,
            server["interface"],
            self.abort_signal,
            fl_ctx,
        )
        client_proxies = []
        for client in bootstrap["clients"]:
            client_proxies.append(
                collab_executor._prepare_proxy(
                    client["name"],
                    client["fqn"],
                    client[DistributedKey.TARGET],
                    relay_cell,
                    client["interface"],
                    self.abort_signal,
                    fl_ctx,
                )
            )
        app.setup(self.workspace, server_proxy, client_proxies, self.abort_signal)
        self.app = app
        self.collab_executor = collab_executor
        self.context = app.new_context(app.name, app.name, set_call_ctx=False)

    def _prepare_torch(self):
        torch, torch_ok = optional_import("torch")
        if torch_ok and torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)

    def _status(self, error=None, error_type=None, traceback_text=None):
        return {
            "rank": self.rank,
            DistributedKey.OK: error is None,
            DistributedKey.ERROR: error,
            CallReplyKey.ERROR_TYPE: error_type,
            CallReplyKey.ERROR_TRACEBACK: traceback_text,
        }

    def _initialize_app(self):
        error = None
        error_type = None
        traceback_text = None
        try:
            self.app.initialize(self.context)
        except Exception as ex:
            error = secure_format_exception(ex)
            error_type = type(ex).__name__
            traceback_text = secure_format_traceback()
            self.logger.error(traceback_text)

        if self.env_world_size > 1:
            torch_dist, ok = optional_import("torch.distributed")
            if not ok or not torch_dist.is_available() or not torch_dist.is_initialized():
                if not error:
                    error = "the launcher created multiple ranks but @collab.init did not initialize torch.distributed"
                    error_type = RuntimeError.__name__
                return self._status(error, error_type, traceback_text)
            self.dist = torch_dist
            self.rank = torch_dist.get_rank()
            self.world_size = torch_dist.get_world_size()
            if self.world_size != self.env_world_size and not error:
                error = (
                    f"torch.distributed world size {self.world_size} does not match "
                    f"launcher WORLD_SIZE {self.env_world_size}"
                )
                error_type = RuntimeError.__name__
            return self._first_error(self._all_gather_status(self._status(error, error_type, traceback_text)))

        return self._status(error, error_type, traceback_text)

    def _all_gather_status(self, status):
        statuses = [None] * self.world_size
        self.dist.all_gather_object(statuses, status)
        return statuses

    @staticmethod
    def _first_error(statuses):
        for status in statuses:
            if not status.get(DistributedKey.OK):
                return status
        return {DistributedKey.OK: True}

    def _broadcast(self, topic, payload=None):
        if self.world_size > 1:
            command = [{DistributedKey.TOPIC: topic, DistributedKey.PAYLOAD: payload}]
            self.dist.broadcast_object_list(command, src=0)

    def _invoke_all(self, request):
        reply = _call_app_method(request, self.app, self.logger)
        if self.world_size <= 1:
            return reply

        rc = reply.get_header(MessageHeaderKey.RETURN_CODE, ReturnCode.OK)
        error = None
        error_type = None
        traceback_text = None
        if rc != ReturnCode.OK:
            if isinstance(reply.payload, dict):
                error = reply.payload.get(CallReplyKey.ERROR)
                error_type = reply.payload.get(CallReplyKey.ERROR_TYPE)
                traceback_text = reply.payload.get(CallReplyKey.ERROR_TRACEBACK)
            error = error or f"Collab call returned {rc}"
        failures = [
            status
            for status in self._all_gather_status(self._status(error, error_type, traceback_text))
            if not status.get(DistributedKey.OK)
        ]
        if not failures:
            return reply
        details = "; ".join(f"rank {status['rank']}: {status[DistributedKey.ERROR]}" for status in failures)
        first = failures[0]
        return _error_reply(
            f"distributed Collab call failed on {details}",
            self.logger,
            error_type=first.get(CallReplyKey.ERROR_TYPE),
            traceback_text=first.get(CallReplyKey.ERROR_TRACEBACK),
        )

    def _finalize_local(self):
        try:
            self.app.finalize(self.context)
            return make_reply(ReturnCode.OK)
        except Exception as ex:
            traceback_text = secure_format_traceback()
            self.logger.error(traceback_text)
            return _error_reply(
                secure_format_exception(ex),
                self.logger,
                error_type=type(ex).__name__,
                traceback_text=traceback_text,
            )
        finally:
            self.finalized = True

    def _run_nonzero_rank(self):
        while True:
            command = [None]
            self.dist.broadcast_object_list(command, src=0)
            command = command[0]
            topic = command.get(DistributedKey.TOPIC)
            if topic == DistributedTopic.INVOKE:
                self._invoke_all(command.get(DistributedKey.PAYLOAD))
            elif topic == DistributedTopic.FINALIZE:
                reply = self._finalize_local()
                if reply.get_header(MessageHeaderKey.RETURN_CODE, ReturnCode.OK) != ReturnCode.OK:
                    raise RuntimeError(f"Collab finalization failed on rank {self.rank}: {reply.payload}")
                return
            elif topic == _CLOSE:
                return
            else:
                raise RuntimeError(f"unknown distributed Collab command: {topic}")

    def run(self):
        fobs_initialize(workspace=self.workspace, job_id=self.args.job_id)
        bootstrap = self._load_bootstrap()
        self._prepare_torch()
        if self.rank == 0:
            self._start_cell(bootstrap)
        else:
            self.parent_fqcn = bootstrap[DistributedKey.PARENT_FQCN]
            self.protocol_id = bootstrap[DistributedKey.PROTOCOL_ID]
            self.secure_supported = bool(bootstrap.get(DistributedKey.SECURE_SUPPORTED, False))
        self._build_components(bootstrap)
        self._build_app(bootstrap)
        init_status = self._initialize_app()
        self.initialized = True
        if self.rank == 0:
            self._send_ready(init_status)
            self.exit_event.wait()
        else:
            self._run_nonzero_rank()

    def close(self):
        try:
            if self.collab_executor:
                self.collab_executor._shutdown_call_executors()
        finally:
            if self.cell is not None:
                self.cell.stop()
                for shutdown in (DownloadService.shutdown, reliable_retry_scheduler.shutdown, stream_shutdown):
                    try:
                        shutdown()
                    except Exception as ex:
                        self.logger.debug(f"failed to stop distributed worker streaming service: {ex}")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collab-bootstrap", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--client-name", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--startup-timeout", required=True)
    return parser.parse_args()


def main():
    worker = DistributedWorker(_parse_args())
    try:
        worker.run()
    finally:
        worker.close()


if __name__ == "__main__":
    main()
