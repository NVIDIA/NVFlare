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
import traceback
from concurrent.futures import ThreadPoolExecutor, wait

from nvflare.edge.emulator.config import ConfigParser
from nvflare.edge.emulator.device_emulator import DeviceEmulator
from nvflare.edge.emulator.device_task_processor import DeviceTaskProcessor
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.user_info import UserInfo

log = logging.getLogger(__name__)


def device_run(
    endpoint_url: str, device_info: DeviceInfo, user_info: UserInfo, capabilities: dict, processor: DeviceTaskProcessor
):
    device_id = device_info.device_id
    try:
        emulator = DeviceEmulator(endpoint_url, device_info, user_info, capabilities, processor)
        emulator.run()

        log.info(f"Emulator run for device {device_id} ended")
    except Exception as ex:
        traceback.print_exc()
        log.error(f"Device {device_id} failed to run: {ex}")


def run_emulator(config_file: str):
    parser = ConfigParser(config_file)
    num = parser.get_num_devices()
    endpoint_url = parser.get_endpoint()
    log.info(f"Running {num} devices. Endpoint URL: {endpoint_url}")

    with ThreadPoolExecutor(max_workers=num) as thread_pool:
        futures = []
        for i in range(num):
            prefix = parser.get_device_id_prefix()
            if not prefix:
                prefix = "device-"
            device_id = f"{prefix}{i}"
            device_info = DeviceInfo(f"{device_id}", "flare_mobile", "1.0")
            user_info = UserInfo("demo_id", "demo_user")
            variables = {"device_id": device_id, "user_id": user_info.user_id}
            processor = parser.get_processor(variables)
            f = thread_pool.submit(
                device_run, endpoint_url, device_info, user_info, parser.get_capabilities(), processor
            )
            futures.append(f)

        wait(futures)

    log.info("Emulator run ended")


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run NVFlare edge device emulator")

    parser.add_argument(
        "config_file",
        type=str,
        help="Location of JSON configuration file",
    )

    # Parse arguments
    args = parser.parse_args()

    # Run emulator
    run_emulator(args.config_file)


if __name__ == "__main__":
    main()
