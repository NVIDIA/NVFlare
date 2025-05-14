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
import argparse
import logging

from nvflare.edge.simulation.config import ConfigParser
from nvflare.edge.simulation.devices.tp import TPDeviceFactory
from nvflare.edge.simulation.feg_api import FegApi
from nvflare.edge.simulation.simulated_device import SimulatedDevice
from nvflare.edge.simulation.simulator import Simulator
from nvflare.edge.web.models.job_request import JobRequest
from nvflare.edge.web.models.result_report import ResultReport
from nvflare.edge.web.models.task_request import TaskRequest

log = logging.getLogger(__name__)


def run_device_simulator(config_file: str):
    parser = ConfigParser(config_file)
    num = parser.get_num_devices()
    endpoint_url = parser.get_endpoint()
    log.info(f"Running {num} devices. Endpoint URL: {endpoint_url}")

    simulator = Simulator(
        device_factory=TPDeviceFactory(parser),
        num_active_devices=parser.get_num_active_devices(),
        num_devices=parser.get_num_devices(),
        num_workers=parser.get_num_workers(),
        cycle_duration=parser.get_cycle_duration(),
        device_reuse_rate=parser.device_reuse_rate,
    )

    simulator.set_send_func(_send_request, parser=parser)
    simulator.start()

    log.info("DeviceSimulator run ended")


def _send_request(request, device: SimulatedDevice, parser: ConfigParser):
    api = FegApi(
        endpoint=parser.get_endpoint(),
        device_info=device.get_device_info(),
        user_info=device.get_user_info(),
    )
    if isinstance(request, TaskRequest):
        return api.get_task(request)

    if isinstance(request, JobRequest):
        return api.get_job(request)

    if isinstance(request, ResultReport):
        return api.report_result(request)

    raise ValueError(f"unknown type of request {type(request)}")


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run NVFlare edge DeviceSimulator")

    parser.add_argument(
        "config_file",
        type=str,
        help="Location of JSON configuration file",
    )

    # Parse arguments
    args = parser.parse_args()

    # Run DeviceSimulator
    run_device_simulator(args.config_file)


if __name__ == "__main__":
    main()
