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

from nvflare.edge.constants import MsgKey
from nvflare.edge.emulator.device_task_processor import DeviceTaskProcessor
from nvflare.edge.model_protocol import (
    ModelBufferType,
    ModelEncoding,
    ModelExchangeFormat,
    ModelNativeFormat,
    verify_payload,
)
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.task_response import TaskResponse
from nvflare.edge.web.models.user_info import UserInfo

log = logging.getLogger(__name__)


def save_to_pte(model_string: str, filename: str):
    binary_data = base64.b64decode(model_string)
    with open(filename, "wb") as f:
        f.write(binary_data)


def run_training_with_timeout(
    train_program: str, model_path: str, result_path: str, data_path: str = "", timeout_seconds: int = 300
) -> int:
    try:
        command = [train_program, "--model_path", model_path, "--output_path", result_path, "--data_path", data_path]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for process to complete with timeout
        stdout, stderr = process.communicate(timeout=timeout_seconds)

        if process.returncode != 0:
            print(f"Error output: {stderr}")
            raise subprocess.CalledProcessError(process.returncode, command, stdout, stderr)

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


class ETTaskProcessor(DeviceTaskProcessor):
    def __init__(
        self, device_info: DeviceInfo, user_info: UserInfo, et_binary_path: str, et_model_path: str, data_path: str
    ):
        super().__init__(device_info, user_info)
        self.job_id = None
        self.job_name = None
        self.device_info = device_info
        self.et_binary_path = et_binary_path
        self.et_model_path = et_model_path
        self.data_path = data_path

        device_io_dir = f"{device_info.device_id}_output"
        os.makedirs(device_io_dir, exist_ok=True)
        self.model_path = os.path.abspath(os.path.join(device_io_dir, self.et_model_path))
        self.result_path = os.path.abspath(os.path.join(device_io_dir, "training_result.json"))
        self.train_binary = os.path.abspath(os.path.join(device_io_dir, self.et_binary_path))
        self._setup_train_program()

    def _setup_train_program(self):
        if not os.path.exists(self.train_binary):
            shutil.copy2(self.et_binary_path, self.train_binary)
            # Make it executable
            os.chmod(self.train_binary, 0o755)

    def setup(self, job: JobResponse) -> None:
        self.job_id = job.job_id
        self.job_name = job.job_name

    def shutdown(self) -> None:
        pass

    def process_task(self, task: TaskResponse) -> dict:
        """Process received task and return results.

        Args:
            task: The task response containing model and instructions

        Returns:
            dict: Results from training

        Raises:
            ValueError: If task data is invalid or protocol validation fails
            RuntimeError: If training operations fail
        """
        log.info(f"Processing task {task.task_name=}")

        if task.task_name != "train":
            log.error(f"Received unknown task: {task.task_name}")
            raise ValueError(f"Unsupported task type: {task.task_name}")

        # Validate inputs first - fail fast if invalid
        payload = verify_payload(
            task.task_data[MsgKey.PAYLOAD],
            expected_type=ModelBufferType.EXECUTORCH,
            expected_format=ModelNativeFormat.BINARY,
            expected_encoding=ModelEncoding.BASE64,
        )

        # Save model to disk for training
        try:
            save_to_pte(payload[ModelExchangeFormat.MODEL_BUFFER], self.model_path)
        except Exception as e:
            log.error(f"Failed to save model: {e}")
            raise RuntimeError("Failed to save model to disk") from e

        # Run training with timeout
        try:
            result = run_training_with_timeout(
                self.train_binary, self.model_path, self.result_path, self.data_path, timeout_seconds=600
            )
            log.info("Training completed successfully")
        except subprocess.TimeoutExpired as e:
            log.error("Training exceeded timeout limit")
            raise RuntimeError("Training took too long and was terminated") from e
        except subprocess.CalledProcessError as e:
            log.error(f"Training process failed with return code {e.returncode}")
            raise RuntimeError("Training process failed") from e
        except Exception as e:
            log.error(f"Training failed with unexpected error: {e}")
            raise RuntimeError("Training failed unexpectedly") from e

        # Read and return results
        try:
            diff_dict = read_training_result(self.result_path)
            return {"result": diff_dict}
        except Exception as e:
            log.error(f"Failed to read training results: {e}")
            raise RuntimeError("Failed to read training results") from e
