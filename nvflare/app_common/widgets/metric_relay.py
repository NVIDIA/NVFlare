# Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
import threading
from typing import Tuple

from nvflare.apis.analytix import ANALYTIC_EVENT_TYPE
from nvflare.apis.dxo import from_dict
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.utils.analytix_utils import send_analytic_dxo
from nvflare.app_common.metrics_exchange.metrics_sender import (
    ANALYTICS_BOOTSTRAP_FILE,
    CHANNEL,
    TOPIC_LOG,
    write_bootstrap,
)
from nvflare.client.config import ConfigKey
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.cellnet.utils import make_reply
from nvflare.fuel.utils.attributes_exportable import AttributesExportable
from nvflare.security.logging import secure_format_exception
from nvflare.widgets.widget import Widget


class MetricRelay(Widget, AttributesExportable):
    """Receive metrics on the existing CJ Cell and relay them as analytic events."""

    def __init__(
        self,
        event_type: str = ANALYTIC_EVENT_TYPE,
        fed_event: bool = True,
        bootstrap_file_name: str = ANALYTICS_BOOTSTRAP_FILE,
    ):
        super().__init__()
        self._event_type = event_type
        self._fed_event = fed_event
        self._bootstrap_file_name = bootstrap_file_name
        self._engine = None
        self._config = None
        self._bootstrap_path = None
        self._lock = threading.Lock()

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.ABOUT_TO_START_RUN:
            self._start(fl_ctx)
        elif event_type == EventType.ABOUT_TO_END_RUN:
            self._stop(fl_ctx)

    def _start(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        cell = engine.get_cell()
        cell.make_internal_listener()
        connect_url = cell.get_internal_listener_url()
        if not connect_url:
            raise RuntimeError("CJ Cell has no internal listener for metrics")
        receiver_fqcn = cell.get_fqcn()
        client_fqcn = FQCN.join([*FQCN.split(receiver_fqcn), f"metrics~{secrets.token_hex(12)}"])
        config = {
            "connect_url": connect_url,
            "receiver_fqcn": receiver_fqcn,
            "client_fqcn": client_fqcn,
        }
        workspace = engine.get_workspace()
        path = os.path.join(workspace.get_app_config_dir(fl_ctx.get_job_id()), self._bootstrap_file_name)
        with self._lock:
            self._engine = engine
            self._config = config
            self._bootstrap_path = path
        cell.register_request_cb(CHANNEL, TOPIC_LOG, self._receive)
        try:
            write_bootstrap(path, config)
        except BaseException:
            self._stop(fl_ctx)
            raise

    def _receive(self, request):
        with self._lock:
            config = self._config
            engine = self._engine
        if not config:
            return make_reply(CellReturnCode.SERVICE_UNAVAILABLE)
        if request.get_header(MessageHeaderKey.ORIGIN) != config["client_fqcn"]:
            return make_reply(CellReturnCode.INVALID_REQUEST)
        try:
            dxo = from_dict(request.payload)
            with engine.new_context() as fl_ctx:
                send_analytic_dxo(self, dxo, fl_ctx, self._event_type, fire_fed_event=self._fed_event)
        except Exception as e:
            self.logger.warning(f"invalid metric: {secure_format_exception(e)}")
            return make_reply(CellReturnCode.INVALID_REQUEST)
        return make_reply(CellReturnCode.OK)

    def _stop(self, fl_ctx: FLContext):
        with self._lock:
            path = self._bootstrap_path
            self._engine = None
            self._config = None
            self._bootstrap_path = None
        if path:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError as e:
                self.log_warning(fl_ctx, f"failed to remove analytics bootstrap: {secure_format_exception(e)}")

    def export(self, export_mode: str) -> Tuple[str, dict]:
        with self._lock:
            if not self._config:
                raise RuntimeError("MetricRelay has not started")
            config = dict(self._config)
        return ConfigKey.METRICS_EXCHANGE, config
