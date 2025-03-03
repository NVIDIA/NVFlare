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
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, wait

from nvflare.edge.emulator.device_emulator import DeviceEmulator
from nvflare.edge.emulator.device_task_processor import DeviceTaskProcessor
from nvflare.edge.emulator.sample_task_processor import SampleTaskProcessor
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.user_info import UserInfo

log = logging.getLogger(__name__)


def device_run(endpoint_url: str, device_info: DeviceInfo, user_info: UserInfo, processor: DeviceTaskProcessor):
    device_id = device_info.device_id
    try:
        capabilities = {"methods": ["xgboost", "cnn"], "cpu": 16, "gpu": 1024}
        emulator = DeviceEmulator(endpoint_url, device_info, user_info, capabilities, processor)
        emulator.run()

        log.info(f"Emulator run for device {device_id} ended")
    except Exception as ex:
        log.error(f"Device {device_id} failed to run: {ex}")


def run_emulator(endpoint_url: str, num: int):
    with ThreadPoolExecutor(max_workers=num) as thread_pool:
        futures = []
        for i in range(num):
            device_info = DeviceInfo(f"device-{i}", "flare_mobile", "1.0")
            user_info = UserInfo("demo_id", "demo_user")
            processor = SampleTaskProcessor(device_info, user_info)
            f = thread_pool.submit(device_run, endpoint_url, device_info, user_info, processor)
            futures.append(f)

        wait(futures)

    log.info("Emulator run ended")


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    n = len(sys.argv)
    if n >= 2:
        endpoint = sys.argv[1]
    else:
        endpoint = "http://localhost:9007"

    if n >= 3:
        num_devices = int(sys.argv[2])
    else:
        num_devices = 4

    run_emulator(endpoint, num_devices)
