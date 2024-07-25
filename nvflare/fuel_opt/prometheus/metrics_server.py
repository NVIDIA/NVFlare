# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

# metrics_server.py
import argparse
import json
import logging
import time
from http.server import HTTPServer

from prometheus_client import REGISTRY
from prometheus_client.exposition import MetricsHandler

# Load the metrics configuration
from prometheus_client.metrics_core import GaugeMetricFamily
from prometheus_client.registry import Collector

from nvflare.metrics.metrics_keys import MetricKeys

metrics_store = {}
logger = logging.getLogger("CustomMetricsHandler")


# Use a custom collector to yield the stored metrics
class CustomCollector(Collector):
    def collect(self):
        for metric in metrics_store.values():
            yield metric


REGISTRY.register(CustomCollector())


class CustomMetricsHandler(MetricsHandler):
    def __init__(self, *args, **kwargs):
        self.metrics_store = kwargs.pop("metrics_store", {})
        super().__init__(*args, **kwargs)

    def do_POST(self):
        if self.path == "/update_metrics":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            content = json.loads(post_data)

            if content:
                for metric_data in content:
                    metric_name = metric_data.get(MetricKeys.metric_name)
                    value = metric_data.get(MetricKeys.value)
                    labels = metric_data.get(MetricKeys.labels, {})
                    timestamp = metric_data.get(MetricKeys.timestamp, int(time.time()))

                    # Create a unique key based on metric name and labels
                    metric_key = (metric_name, tuple(sorted(labels.items())))

                    if metric_key not in metrics_store:
                        # Register/update GaugeMetricFamily with timestamp
                        gauge = GaugeMetricFamily(
                            metric_name, f"Description of {metric_name}", labels=list(labels.keys())
                        )
                        metrics_store[metric_key] = gauge
                    else:
                        # Update the existing gauge
                        gauge = metrics_store[metric_key]

                    gauge.add_metric(list(labels.values()), value, timestamp=timestamp)

            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()


def run_http_server(port):
    # Create a custom HTTP server
    server = HTTPServer(("0.0.0.0", port), CustomMetricsHandler)

    # Start the HTTP server in a separate thread
    from threading import Thread

    thread = Thread(target=server.serve_forever)
    thread.daemon = True

    thread.start()
    print(f"started prometheus metrics server on port {port}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Start/Stop Prometheus metrics collection server.")
    parser.add_argument("--config", type=str, help="Path to the JSON configuration file")
    parser.add_argument("--start", action="store_true", help="Start the Prometheus HTTP server")
    parser.add_argument("--port", type=int, default=8000, help="Port number for the Prometheus HTTP server")

    return parser


if __name__ == "__main__":
    p = parse_arguments()
    args = p.parse_args()
    if args.start:
        run_http_server(args.port)
        # Keep the main thread alive to prevent the server from shutting down
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down the server...")
    else:
        p.print_help()
