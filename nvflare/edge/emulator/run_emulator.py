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

from nvflare.edge.emulator.device_emulator import DeviceEmulator
from nvflare.edge.emulator.sample_task_processor import SampleTaskProcessor
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.user_info import UserInfo

log = logging.getLogger(__name__)


def run_emulator():

    # read from JSON, a list of devices
    device_info = DeviceInfo("1234", "flare_mobile", "1.0")
    user_info = UserInfo("demo_id", "demo_user")
    # Configure processor
    processor = SampleTaskProcessor(device_info, user_info)
    capabilities = {
        "methods": ["xgboost", "cnn"],
        "cpu": 16,
        "gpu": 1024
    }
    endpoint = "http://localhost:4321"
    emulator = DeviceEmulator(endpoint, device_info, user_info, capabilities, processor)
    emulator.run()

    log.info("Emulator run ended")


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

    run_emulator()
