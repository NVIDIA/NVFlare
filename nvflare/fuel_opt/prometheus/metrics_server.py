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
import os.path
import time
from http.server import HTTPServer

from load_metrics import load_metrics_config
from prometheus_client import Counter, Gauge
from prometheus_client.exposition import MetricsHandler

# Load the metrics configuration
metrics_store = {}
logger = logging.getLogger("CustomMetricsHandler")


class CustomMetricsHandler(MetricsHandler):
    def __init__(self, *args, **kwargs):
        self.metrics_store = kwargs.pop("metrics_store", {})
        super().__init__(*args, **kwargs)

    def do_POST(self):
        if self.path == "/update_metrics":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            metrics_data = json.loads(post_data)

            # Update the metrics store
            for metric_name, value in metrics_data.items():
                if metric_name in metrics_store:
                    p1 = metrics_store[metric_name]
                    if isinstance(p1, Gauge):
                        p1.set(value)
                    elif isinstance(p, Counter):
                        p1.inc(value)
                else:
                    p1 = Gauge(metric_name, metric_name)
                    metrics_store[metric_name] = p1
                    p1.set(value)

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
    logger.info(f"started prometheus metrics server on port {port}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Start/Stop Prometheus metrics collection server.")
    parser.add_argument("--config", type=str, help="Path to the JSON configuration file")
    parser.add_argument("--start", action="store_true", help="Start the Prometheus HTTP server")
    parser.add_argument("--port", type=int, default=9090, help="Port number for the Prometheus HTTP server")

    return parser


if __name__ == "__main__":

    p = parse_arguments()
    args = p.parse_args()
    if args.start:
        current_dir = os.path.dirname(__file__)
        app_metrics_config_path = os.path.join(current_dir, "app_metrics_config.json")
        metrics_store = load_metrics_config(app_metrics_config_path)
        if args.config:
            metrics_store = load_metrics_config(args.config)

        run_http_server(args.port)
        # Keep the main thread alive to prevent the server from shutting down
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down the server...")
    else:
        p.print_help()
