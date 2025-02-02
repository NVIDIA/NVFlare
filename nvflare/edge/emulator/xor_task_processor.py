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

import base64
import json
import logging
import os
import shutil
import subprocess

import torch

from nvflare.edge.emulator.device_task_processor import DeviceTaskProcessor
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.task_response import TaskResponse
from nvflare.edge.web.models.user_info import UserInfo

log = logging.getLogger(__name__)

SOURCE_BINARY = "train_xor"


def save_to_pte(model_string: str, filename: str):
    binary_data = base64.b64decode(model_string)
    with open(filename, "wb") as f:
        f.write(binary_data)


def run_training_with_timeout(train_program: str, model_path: str, result_path: str, timeout_seconds: int = 300) -> int:
    try:
        process = subprocess.Popen(
            [train_program, "--model_path", model_path, "--output_path", result_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for process to complete with timeout
        stdout, stderr = process.communicate(timeout=timeout_seconds)

        if process.returncode != 0:
            print(f"Error output: {stderr}")
            raise subprocess.CalledProcessError(process.returncode, ["./train_xor"], stdout, stderr)

        print(f"Output: {stdout}")
        return process.returncode

    except subprocess.TimeoutExpired:
        process.kill()
        print("Training timed out")
        raise
    except Exception as e:
        print(f"Error during training: {e}")
        raise


def read_training_result(result_path: str = "training_result.json"):
    try:
        with open(result_path, "r") as f:
            results = json.load(f)

        return results

    except FileNotFoundError:
        print(f"Could not find file: {result_path}")
        raise
    except json.JSONDecodeError:
        print(f"Error parsing JSON file: {result_path}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise


class XorTaskProcessor(DeviceTaskProcessor):
    def __init__(self, device_info: DeviceInfo, user_info: UserInfo):
        super().__init__(device_info, user_info)
        self.job_id = None
        self.job_name = None
        self.device_info = device_info

        device_io_dir = f"{device_info.device_id}_output"
        os.makedirs(device_io_dir, exist_ok=True)
        self.model_path = os.path.abspath(os.path.join(device_io_dir, "xor.pte"))
        self.result_path = os.path.abspath(os.path.join(device_io_dir, "training_result.json"))
        self.train_binary = os.path.abspath(os.path.join(device_io_dir, "train.xor"))
        self._setup_train_program()

    def _setup_train_program(self):
        if not os.path.exists(self.train_binary):
            shutil.copy2(SOURCE_BINARY, self.train_binary)
            # Make it executable
            os.chmod(self.train_binary, 0o755)

    def setup(self, job: JobResponse) -> None:
        self.job_id = job.job_id
        self.job_name = job.job_name

    def shutdown(self) -> None:
        pass

    def process_task(self, task: TaskResponse) -> dict:
        log.info(f"Processing task {task}")

        # Local training or validation
        result = None
	if task.task_name == "train":
            # save received pte
            save_to_pte(task.task_data["task_data"], self.model_path)
            try:
                result = run_training_with_timeout(
                    self.train_binary, self.model_path, self.result_path, timeout_seconds=600
                )
                print("Training completed successfully")
            except subprocess.TimeoutExpired:
                print("Training took too long and was terminated")
            except subprocess.CalledProcessError as e:
                print(f"Training failed with return code {e.returncode}")
            except Exception as e:
                print(f"Training Unexpected error: {e}")

            try:
                diff_dict = read_training_result(self.result_path)

            except Exception as e:
                print(f"Failed to read results: {e}")
                raise

            result = {
                "result": diff_dict,
            }
        else:
            log.error(f"Received unknown task: {task.task_name}")

        return result
