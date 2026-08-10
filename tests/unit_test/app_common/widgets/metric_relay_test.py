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
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.utils.analytix_utils import create_analytic_dxo
from nvflare.app_common.metrics_exchange.metrics_sender import (
    ANALYTICS_BOOTSTRAP_FILE,
    CHANNEL,
    TOPIC_LOG,
    read_bootstrap,
)
from nvflare.app_common.widgets.metric_relay import MetricRelay
from nvflare.client.config import ConfigKey
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.utils import new_cell_message


class _Cell:
    def __init__(self):
        self.callbacks = {}
        self.site_credentials = {"auth_token": "server-bearer", "auth_token_signature": "server-signature"}

    def make_internal_listener(self):
        pass

    def get_internal_listener_url(self):
        return "shared-file://0/tmp/metrics"

    def get_fqcn(self):
        return "site-1.job"

    def register_request_cb(self, channel, topic, callback):
        self.callbacks[(channel, topic)] = callback


def _start_relay(tmp_path):
    cell = _Cell()
    engine = MagicMock()
    engine.get_cell.return_value = cell
    engine.get_workspace.return_value.get_app_config_dir.return_value = str(tmp_path)
    engine.new_context.return_value = nullcontext(MagicMock(spec=FLContext))
    fl_ctx = MagicMock(spec=FLContext)
    fl_ctx.get_engine.return_value = engine
    fl_ctx.get_job_id.return_value = "job"
    relay = MetricRelay()
    relay._start(fl_ctx)
    config = read_bootstrap(str(tmp_path / ANALYTICS_BOOTSTRAP_FILE))[1]
    return relay, cell, fl_ctx, config


def _request(config, origin=None, payload=None):
    headers = {MessageHeaderKey.ORIGIN: origin or config["client_fqcn"]}
    if payload is None:
        payload = create_analytic_dxo("loss", 0.25, AnalyticsDataType.SCALAR, global_step=3).to_dict()
    return new_cell_message(headers, payload)


def test_relay_exports_the_direct_cell_config(tmp_path):
    relay, _, fl_ctx, config = _start_relay(tmp_path)
    try:
        assert relay.export("peer") == (ConfigKey.METRICS_EXCHANGE, config)
    finally:
        relay._stop(fl_ctx)


def test_relay_accepts_only_the_launched_cell_origin(tmp_path):
    relay, cell, fl_ctx, config = _start_relay(tmp_path)
    try:
        reply = cell.callbacks[(CHANNEL, TOPIC_LOG)](_request(config, origin="attacker"))
        assert reply.get_header(MessageHeaderKey.RETURN_CODE) == CellReturnCode.INVALID_REQUEST
    finally:
        relay._stop(fl_ctx)


def test_relay_rejects_invalid_log_payload(tmp_path):
    relay, cell, fl_ctx, config = _start_relay(tmp_path)
    try:
        reply = cell.callbacks[(CHANNEL, TOPIC_LOG)](_request(config, payload="invalid"))
        assert reply.get_header(MessageHeaderKey.RETURN_CODE) == CellReturnCode.INVALID_REQUEST
    finally:
        relay._stop(fl_ctx)


def test_relay_forwards_concurrent_logs_and_cleans_up_without_bearer_leak(tmp_path):
    relay, cell, fl_ctx, config = _start_relay(tmp_path)
    serialized = json.dumps(config)
    assert "server-bearer" in json.dumps(cell.site_credentials)
    assert "server-bearer" not in serialized and "server-signature" not in serialized
    try:
        with patch("nvflare.app_common.widgets.metric_relay.send_analytic_dxo") as send:
            with ThreadPoolExecutor(max_workers=8) as pool:
                replies = list(pool.map(lambda _: cell.callbacks[(CHANNEL, TOPIC_LOG)](_request(config)), range(24)))
        assert all(r.get_header(MessageHeaderKey.RETURN_CODE) == CellReturnCode.OK for r in replies)
        assert send.call_count == 24
        assert send.call_args.kwargs["fire_fed_event"] is True
    finally:
        relay._stop(fl_ctx)
    assert not (tmp_path / ANALYTICS_BOOTSTRAP_FILE).exists()
    assert relay._config is None and relay._bootstrap_path is None
