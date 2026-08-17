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
from unittest.mock import MagicMock, patch

from nvflare.apis.analytix import AnalyticsDataType, LogWriterName
from nvflare.app_common.metrics_exchange.metrics_sender import MetricsSender
from nvflare.client import tracking
from nvflare.client.api_context import APIContext


def _reset_tracking_globals():
    tracking._tracking_context = None


def test_summary_writer_uses_explicit_analytics_context():
    context = MagicMock(spec=MetricsSender)
    writer = tracking.SummaryWriter(ctx=context)

    writer.add_scalar("loss", 0.5, global_step=2)

    context.add.assert_called_once_with(
        tag="loss",
        value=0.5,
        data_type=AnalyticsDataType.SCALAR,
        global_step=2,
        writer=LogWriterName.TORCH_TB,
    )


def test_summary_writer_preserves_client_api_context_compatibility():
    context = MagicMock(spec=APIContext)
    context.api = MagicMock()
    writer = tracking.SummaryWriter(ctx=context)

    writer.add_scalar("loss", 0.5, global_step=2)

    context.api.log.assert_called_once_with(
        "loss",
        0.5,
        AnalyticsDataType.SCALAR,
        global_step=2,
        writer=LogWriterName.TORCH_TB,
    )


def test_concurrent_tracking_init_creates_one_context(tmp_path):
    _reset_tracking_globals()
    context = MagicMock(spec=MetricsSender)
    contexts = []
    barrier = threading.Barrier(16)

    def initialize():
        barrier.wait()
        contexts.append(tracking.init(config_file=str(tmp_path / "metrics.json")))

    with (patch("nvflare.client.tracking.MetricsSender", return_value=context) as sender_cls,):
        threads = [threading.Thread(target=initialize) for _ in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2.0)

    try:
        assert contexts == [context] * 16
        sender_cls.assert_called_once()
        context.init.assert_called_once_with(rank=None)
    finally:
        _reset_tracking_globals()
