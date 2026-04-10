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
import sys
import types
from unittest.mock import Mock

from nvflare.apis.fl_constant import ReservedTopic
from nvflare.app_common.np.recipes.fedavg import NumpyFedAvgRecipe
from nvflare.metrics.job_metrics_collector import JobMetricsCollector
from nvflare.metrics.metrics_keys import MetricKeys, MetricTypes


def initialize_stub(**kwargs):
    return None


def increment_stub(*args, **kwargs):
    return None


def gauge_stub(*args, **kwargs):
    return None


def _get_component(components: list[dict], component_id: str) -> dict:
    return next(c for c in components if c["id"] == component_id)


def test_statsd_reporter_endpoint_is_preserved_in_exported_job_config(tmp_path, monkeypatch):
    fake_datadog = types.ModuleType("datadog")
    fake_datadog.initialize = initialize_stub
    fake_datadog.statsd = types.SimpleNamespace(increment=increment_stub, gauge=gauge_stub)
    monkeypatch.setitem(sys.modules, "datadog", fake_datadog)

    from nvflare.fuel_opt.statsd.statsd_reporter import StatsDReporter

    test_port = 19125
    client_script = tmp_path / "client.py"
    client_script.write_text("print('hello')\n", encoding="utf-8")

    recipe = NumpyFedAvgRecipe(
        name="test-monitoring-fedavg",
        min_clients=1,
        num_rounds=1,
        model=[[1, 2], [3, 4]],
        train_script=str(client_script),
    )
    recipe.job.to_server(
        JobMetricsCollector(tags={"site": "server", "env": "test"}),
        id="server_job_metrics_collector",
    )
    recipe.job.to_server(
        StatsDReporter(site="server", host="statsd-exporter.monitoring.svc.cluster.local", port=test_port),
        id="server_statsd_reporter",
    )
    recipe.job.to_clients(
        JobMetricsCollector(tags={"env": "test"}),
        id="client_job_metrics_collector",
    )
    recipe.job.to_clients(
        StatsDReporter(site="site-1", host="statsd-exporter.monitoring.svc.cluster.local", port=test_port),
        id="client_statsd_reporter",
    )

    recipe.job.export_job(str(tmp_path))
    job_dir = tmp_path / "test-monitoring-fedavg"

    with open(job_dir / "app" / "config" / "config_fed_server.json", encoding="utf-8") as f:
        server_config = json.load(f)
    with open(job_dir / "app" / "config" / "config_fed_client.json", encoding="utf-8") as f:
        client_config = json.load(f)

    server_reporter = _get_component(server_config["components"], "server_statsd_reporter")
    client_reporter = _get_component(client_config["components"], "client_statsd_reporter")

    assert server_reporter["args"]["site"] == "server"
    assert server_reporter["args"]["host"] == "statsd-exporter.monitoring.svc.cluster.local"
    assert server_reporter["args"]["port"] == test_port

    assert client_reporter["args"]["site"] == "site-1"
    assert client_reporter["args"]["host"] == "statsd-exporter.monitoring.svc.cluster.local"
    assert client_reporter["args"]["port"] == test_port


def test_statsd_reporter_disables_itself_after_init_failure(monkeypatch):
    init_calls = 0

    def failing_initialize(**kwargs):
        nonlocal init_calls
        init_calls += 1
        raise RuntimeError("datadog init failed")

    fake_datadog = types.ModuleType("datadog")
    fake_datadog.initialize = failing_initialize
    fake_datadog.statsd = types.SimpleNamespace(increment=increment_stub, gauge=gauge_stub)
    monkeypatch.setitem(sys.modules, "datadog", fake_datadog)

    from nvflare.fuel_opt.statsd.statsd_reporter import StatsDReporter

    reporter = StatsDReporter(site="server", host="statsd-exporter.monitoring.svc.cluster.local", port=9125)
    reporter.logger = Mock()
    metrics = [
        {
            MetricKeys.metric_name: "_system_start_count",
            MetricKeys.value: 1,
            MetricKeys.tags: {"site": "server"},
            MetricKeys.type: MetricTypes.COUNTER,
        }
    ]

    reporter.process_metrics(topic=ReservedTopic.APP_METRICS, metrics=metrics, data_bus=None)
    reporter.process_metrics(topic=ReservedTopic.APP_METRICS, metrics=metrics, data_bus=None)

    assert init_calls == 1
    reporter.logger.warning.assert_called_once()
