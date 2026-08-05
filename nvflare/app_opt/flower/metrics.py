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

"""Direct Cell receiver for metrics emitted by a Flower ClientApp.

Flower owns the ClientApp process through its TIE applet, so the regular
``ClientAPIExecutor`` must not launch or execute it. This widget hosts only the
new Cell Client API session used by ``nvflare.client.tracking``. It deliberately
does not depend on the removed auxiliary metrics transport stack.
"""

import os
import secrets
import threading
import uuid
from typing import Optional

from nvflare.apis.analytix import ANALYTIC_EVENT_TYPE
from nvflare.apis.dxo import DXO
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.utils.analytix_utils import send_analytic_dxo
from nvflare.app_common.executors.client_api.backend_spec import ClientAPIBackendContext
from nvflare.app_common.executors.client_api.cell_backend import CellBackendBase, CellSession
from nvflare.app_common.executors.client_api_executor import FED_ANALYTIC_EVENT_TYPE
from nvflare.client.cell.bootstrap import (
    BOOTSTRAP_SCHEMA_VERSION,
    EXTERNAL_PROCESS_EXECUTION_MODE,
    BootstrapKey,
    write_bootstrap_config,
)
from nvflare.client.cell.defs import CHANNEL, PROTOCOL_VERSION, MsgKey, Topic
from nvflare.client.config import ExchangeFormat
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.cellnet.utils import make_reply as make_cell_reply
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.security.logging import secure_format_traceback
from nvflare.widgets.widget import Widget

_FLOWER_TRAINER_LEAF = "flower_client_api"


class _FlowerSession(CellSession):
    def __init__(self, token: str, trainer_fqcn: str):
        super().__init__(trainer_fqcn)
        self.token = token


class _FlowerMetricsBackend(CellBackendBase):
    """Hosts a metrics-only external-process Client API session."""

    def __init__(self, config_file_name: str):
        super().__init__()
        self._config_file_name = config_file_name
        self._bootstrap_path: Optional[str] = None
        self._session: Optional[_FlowerSession] = None
        self._session_lock = threading.Lock()

    def initialize(self, context: ClientAPIBackendContext, fl_ctx: FLContext) -> None:
        try:
            self._initialize_cell(context, fl_ctx, "flower metrics")
            workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
            if workspace is None:
                raise RuntimeError("workspace is not available in fl_ctx")

            self._cell.make_internal_listener()
            connect_url = self._cell.get_internal_listener_url()
            if not connect_url:
                raise RuntimeError("CJ Cell has no internal listener URL for the Flower Client API")

            trainer_fqcn = FQCN.join([self._cj_fqcn, _FLOWER_TRAINER_LEAF])
            session = _FlowerSession(secrets.token_urlsafe(32), trainer_fqcn)
            self._session = session
            self._cell.register_request_cb(channel=CHANNEL, topic=Topic.HELLO, cb=self._handle_hello)

            config_dir = workspace.get_app_config_dir(self._job_id)
            self._bootstrap_path = os.path.join(config_dir, self._config_file_name)
            write_bootstrap_config(
                self._bootstrap_path,
                {
                    BootstrapKey.SCHEMA_VERSION: BOOTSTRAP_SCHEMA_VERSION,
                    BootstrapKey.EXECUTION_MODE: EXTERNAL_PROCESS_EXECUTION_MODE,
                    BootstrapKey.CONNECT_URL: connect_url,
                    BootstrapKey.CJ_FQCN: self._cj_fqcn,
                    BootstrapKey.TRAINER_FQCN: trainer_fqcn,
                    BootstrapKey.LAUNCH_TOKEN: session.token,
                    BootstrapKey.JOB_ID: self._job_id,
                    BootstrapKey.SITE_NAME: self._site_name,
                    BootstrapKey.SECURE_MODE: self._secure_mode,
                    BootstrapKey.TASK_EXCHANGE: self._task_exchange_config(),
                    BootstrapKey.MEMORY_GC_ROUNDS: context.memory_gc_rounds,
                    BootstrapKey.CUDA_EMPTY_CACHE: context.cuda_empty_cache,
                },
            )
        except BaseException:
            self._cleanup()
            raise

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        return make_reply(ReturnCode.TASK_UNSUPPORTED)

    def finalize(self, fl_ctx: FLContext) -> None:
        if self._finalized:
            return
        self._finalized = True
        self._closed = True
        session = self._session
        if session is not None and session.ready.is_set() and session.session_id:
            try:
                self._cell.fire_and_forget(
                    channel=CHANNEL,
                    topic=Topic.SHUTDOWN,
                    targets=[session.trainer_fqcn],
                    message=new_cell_message(
                        {},
                        {MsgKey.SESSION_ID: session.session_id, MsgKey.REASON: "Flower job ended"},
                    ),
                    optional=True,
                )
            except Exception:
                self.logger.debug(secure_format_traceback())
        self._cleanup()

    def _get_protocol_session(self) -> Optional[CellSession]:
        return self._session

    def _handle_hello(self, request):
        if self._closed:
            return self._protocol_reply(Topic.HELLO_REJECTED, **{MsgKey.REASON: "backend is closed"})
        payload = request.payload
        if not isinstance(payload, dict):
            return make_cell_reply(CellReturnCode.INVALID_REQUEST, error="HELLO payload must be a dict")

        session = self._session
        if session is None:
            return self._hello_reject("no Flower Client API session")

        origin = request.get_header(MessageHeaderKey.ORIGIN) or ""
        claimed_fqcn = payload.get(MsgKey.TRAINER_FQCN)
        if origin != session.trainer_fqcn or claimed_fqcn != session.trainer_fqcn:
            return self._hello_reject(f"unexpected trainer identity (origin={origin!r}, claimed={claimed_fqcn!r})")

        proof = payload.get(MsgKey.PROOF)
        if (
            not isinstance(proof, str)
            or not session.token
            or not secrets.compare_digest(proof.encode("utf-8"), session.token.encode("utf-8"))
        ):
            return self._hello_reject("launch token mismatch")
        if payload.get(MsgKey.PROTOCOL_VERSION) != PROTOCOL_VERSION:
            return self._hello_reject(
                f"unsupported protocol version {payload.get(MsgKey.PROTOCOL_VERSION)!r} " f"(expect {PROTOCOL_VERSION})"
            )
        if payload.get(MsgKey.JOB_ID) != self._job_id:
            return self._hello_reject(f"job id mismatch: {payload.get(MsgKey.JOB_ID)!r}")
        if payload.get(MsgKey.SITE_NAME) != self._site_name:
            return self._hello_reject(f"site name mismatch: {payload.get(MsgKey.SITE_NAME)!r}")
        rank = payload.get(MsgKey.RANK)
        if str(rank) != "0":
            return self._hello_reject(f"only rank 0 may connect (got rank {rank!r})")

        with self._session_lock:
            if not session.ready.is_set():
                session.session_id = uuid.uuid4().hex
                session.ready.set()
                self.logger.info(
                    f"Flower Client API session established: fqcn={origin} session_id={session.session_id}"
                )
            session.touch()
        return self._protocol_reply(
            Topic.HELLO_ACCEPTED,
            **{
                MsgKey.SESSION_ID: session.session_id,
                MsgKey.JOB_ID: self._job_id,
                MsgKey.SITE_NAME: self._site_name,
                MsgKey.HEARTBEAT_INTERVAL: self._context.heartbeat_interval,
                MsgKey.HEARTBEAT_TIMEOUT: self._context.heartbeat_timeout,
            },
        )

    def _hello_reject(self, reason: str):
        self.logger.warning(f"rejecting Flower Client API HELLO: {reason}")
        return self._protocol_reply(Topic.HELLO_REJECTED, **{MsgKey.REASON: reason})

    def _cleanup(self) -> None:
        try:
            if self._bootstrap_path and os.path.exists(self._bootstrap_path):
                os.remove(self._bootstrap_path)
        except Exception as e:
            self.logger.debug(f"failed to remove {self._bootstrap_path}: {e}")
        self._bootstrap_path = None
        session = self._session
        if session is not None:
            session.token = ""
            session.session_id = None
        self._session = None
        self._disable_pass_through()


class FlowerMetricsReceiver(Widget):
    """Receives Flower ClientApp metrics through the direct Cell Client API."""

    def __init__(
        self,
        config_file_name: str,
        heartbeat_interval: float = 5.0,
        heartbeat_timeout: float = 30.0,
    ):
        super().__init__()
        if (
            not isinstance(config_file_name, str)
            or not config_file_name
            or os.path.basename(config_file_name) != config_file_name
        ):
            raise ValueError("config_file_name must be a non-empty file name without directory components")
        self._config_file_name = config_file_name
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_timeout = heartbeat_timeout
        self._backend: Optional[_FlowerMetricsBackend] = None
        self._fire_fed_event = False

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            try:
                backend = _FlowerMetricsBackend(self._config_file_name)
                backend.initialize(
                    ClientAPIBackendContext(
                        executor=self,
                        heartbeat_interval=self._heartbeat_interval,
                        heartbeat_timeout=self._heartbeat_timeout,
                        params_exchange_format=ExchangeFormat.RAW,
                        server_expected_format=ExchangeFormat.RAW,
                    ),
                    fl_ctx,
                )
                self._backend = backend
            except Exception as e:
                self.log_exception(fl_ctx, "cannot initialize the Flower metrics receiver")
                self.system_panic(f"cannot initialize the Flower metrics receiver: {e}", fl_ctx)
        elif event_type == EventType.END_RUN:
            backend = self._backend
            self._backend = None
            if backend is not None:
                backend.finalize(fl_ctx)

    def set_analytics_fire_fed_event(self, enabled: bool) -> None:
        self._fire_fed_event = bool(enabled)

    def fire_log_analytics(self, fl_ctx: FLContext, dxo: DXO) -> None:
        send_analytic_dxo(
            self,
            dxo=dxo,
            fl_ctx=fl_ctx,
            event_type=FED_ANALYTIC_EVENT_TYPE if self._fire_fed_event else ANALYTIC_EVENT_TYPE,
            fire_fed_event=self._fire_fed_event,
        )
