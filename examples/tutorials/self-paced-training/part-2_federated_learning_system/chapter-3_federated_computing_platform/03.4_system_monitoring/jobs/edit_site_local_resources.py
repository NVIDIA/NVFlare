# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import os
import sys


def get_sys_metrics_collector(site: str, streaming_to_server: bool = False) -> dict:
    return json.loads(
        f"""
        {{
            "id": "sys_metrics_collector",
            "path": "nvflare.metrics.system_metrics_collector.SysMetricsCollector",
            "args": {{
                "tags": {{
                    "site": "{site}",
                    "env": "dev"
                }}, 
                "streaming_to_server": {str(streaming_to_server).lower()}
            }}
        }}
        """
    )


def get_sys_metrics_receiver() -> dict:
    return json.loads(
        """
        {

            "id": "remote_metrics_receiver",
            "path": "nvflare.metrics.remote_metrics_receiver.RemoteMetricsReceiver",
            "args": {
                "events": [
                    "fed.metrics_event"
                ]
            }
        }
      """
    )


def get_fed_event_converter() -> dict:
    return json.loads(
        """
        {
            "id": "sys_metrics_event_to_fed",
            "path": "nvflare.app_common.widgets.convert_to_fed_event.ConvertToFedEvent",
            "args": {
                "events_to_convert": ["metrics_event"]
            }
        }
        """
    )


def get_statsd_reporter() -> dict:
    return json.loads(
        """
        {
            "id": "statsd_reporter",
            "path": "nvflare.fuel_opt.statsd.statsd_reporter.StatsDReporter",
            "args": {
                "host": "localhost",
                "port": 9125
            }
        }
        """
    )


def add_components_to_json(
    input_file_path, output_file_path, site: str, receiving: bool = False, streaming_to_server: bool = False
):

    try:
        with open(input_file_path, "r") as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Error: Unable to read or parse JSON file.")
        return

    if "components" not in data or not isinstance(data["components"], list):
        print("Error: 'components' key not found or is not a list.")
        return

    new_components = [get_sys_metrics_collector(site, streaming_to_server)]

    if receiving is True:
        new_components.append(get_sys_metrics_receiver())

    if streaming_to_server is True:
        new_components.append(get_fed_event_converter())
    else:
        new_components.append(get_statsd_reporter())

    # Append new components to the list
    data["components"].extend(new_components)

    # Write the updated JSON back to the file
    with open(output_file_path, "w") as file:
        json.dump(data, file, indent=4)

    print(f"Successfully generate file: '{output_file_path}'.")


if __name__ == "__main__":

    n_sites = int(sys.argv[1])
    streaming_to_server = sys.argv[2].lower() == "true"

    project_root_dir = sys.argv[3]

    receiving = streaming_to_server

    sites = [("server", receiving, False)]

    for n in range(n_sites):
        sites.append((f"site-{n + 1}", False, streaming_to_server))

    for site_name, receiving, streaming in sites:
        input_file_path = os.path.join(project_root_dir, site_name, "local", "resources.json.default")
        output_file_path = os.path.join(project_root_dir, site_name, "local", "resources.json")
        add_components_to_json(
            input_file_path, output_file_path, site_name, receiving=receiving, streaming_to_server=streaming
        )
