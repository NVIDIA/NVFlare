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
import importlib
import json
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, wait

from nvflare.edge.emulator.device_emulator import DeviceEmulator
from nvflare.edge.emulator.device_task_processor import DeviceTaskProcessor
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.user_info import UserInfo

log = logging.getLogger(__name__)

from typing import Any, Dict, Tuple, Type


def load_class_from_config(config_path: str) -> Tuple[Type[Any], Dict[str, Any]]:
    """Load class and config from JSON file.

    Args:
        config_path: Path to JSON config file

    Returns:
        tuple: (class, config_dict)

    Raises:
        FileNotFoundError: If config file doesn't exist
        KeyError: If required fields are missing
        ImportError: If class cannot be imported
    """
    with open(config_path) as f:
        config = json.load(f)

    if "class_path" not in config:
        raise KeyError("Config must contain 'class_path'")

    try:
        module_path, class_name = config["class_path"].rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import {config['class_path']}: {e}")

    return cls, config.get("class_args", {})


def create_task_processor(filename: str, device_info, user_info):
    try:
        cls, config = load_class_from_config(filename)
        processor = cls(device_info, user_info, **config)
        return processor
    except Exception as e:
        logging.error(f"Failed to create task processor {filename}: {e}")
        raise


def device_run(endpoint_url: str, device_info: DeviceInfo, user_info: UserInfo, processor: DeviceTaskProcessor):
    device_id = device_info.device_id
    try:
        capabilities = {"methods": ["xor", "cifar10"], "cpu": 16, "gpu": 1024}
        emulator = DeviceEmulator(endpoint_url, device_info, user_info, capabilities, processor)
        emulator.run()

        log.info(f"Emulator run for device {device_id} ended")
    except Exception as ex:
        traceback.print_exc()
        log.error(f"Device {device_id} failed to run: {ex}")


def run_emulator(endpoint_url: str, num: int, emulator_config: str):
    with ThreadPoolExecutor(max_workers=num) as thread_pool:
        futures = []
        for i in range(num):
            device_info = DeviceInfo(f"device-{i}", "flare_mobile", "1.0")
            user_info = UserInfo("demo_id", "demo_user")
            processor = create_task_processor(emulator_config, device_info, user_info)
            f = thread_pool.submit(device_run, endpoint_url, device_info, user_info, processor)
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
    parser = argparse.ArgumentParser(description="Run NVFlare emulator")

    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:9007",
        help="Server endpoint URL (default: http://localhost:9007)",
    )

    parser.add_argument("--num-devices", type=int, default=4, help="Number of devices to emulate (default: 4)")
    parser.add_argument("--emulator_config", type=str, default="", help="Emulator config file")

    # Parse arguments
    args = parser.parse_args()

    # Run emulator
    run_emulator(args.endpoint, args.num_devices, args.emulator_config)


if __name__ == "__main__":
    main()
