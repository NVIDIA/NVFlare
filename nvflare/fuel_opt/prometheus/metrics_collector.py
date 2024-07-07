import traceback

import requests

from nvflare.apis.fl_constant import ReservedTopic
from nvflare.fuel.data_event.data_bus import DataBus


class MetricsCollector:

    def __init__(self, metrics_server_url='http://localhost:8000/update_metrics'):
        self.metrics = {}
        self.data_bus = DataBus()
        self.data_bus.subscribe([ReservedTopic.APP_METRICS], self.process_metrics)
        self.metrics_server_url = metrics_server_url

    def process_metrics(self, topic, metrics, data_bus):
        try:
            if topic == ReservedTopic.APP_METRICS:
                # Send metrics data via HTTP POST
                try:
                    print(f"post metrics = {metrics} to {self.metrics_server_url}")
                    response = requests.post(self.metrics_server_url, json=metrics)
                    response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    print(f"Failed to send metrics: {e}")

        except Exception as e:
            print(traceback.format_exc())
