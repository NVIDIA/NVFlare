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
import random
import time

from nvflare.apis.dxo import DXO, from_dict
from nvflare.edge.simulation.device_task_processor import DeviceTaskProcessor
from nvflare.edge.simulation.simulated_device import DeviceFactory, SimulatedDevice
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.task_response import TaskResponse


class NumDevice(SimulatedDevice):

    def __init__(self, device_id, min_train_time=1.0, max_train_time=5.0):
        SimulatedDevice.__init__(self, device_id)
        self.min_train_time = min_train_time
        self.max_train_time = max_train_time

    def do_task(self, task: TaskResponse) -> dict:
        task_data = task.task_data
        assert isinstance(task_data, dict)
        model = from_dict(task_data)
        if not isinstance(model, DXO):
            self.logger.error(f"expect model to be DXO but got {type(model)}")
            raise ValueError("bad model data")

        if model.data_kind != "number":
            self.logger.error(f"expect model data kind to be 'number' but got {model.data_kind}")
            raise ValueError("bad model data kind")

        value = model.data.get("value", 0)
        result = value + 1
        result_dxo = DXO(data_kind="number", data={"value": result})
        delay = random.uniform(self.min_train_time, self.max_train_time)
        time.sleep(delay)
        return result_dxo.to_dict()


class NumDeviceFactory(DeviceFactory):

    def __init__(self, min_train_time=1.0, max_train_time=5.0):
        DeviceFactory.__init__(self)
        self.min_train_time = min_train_time
        self.max_train_time = max_train_time

    def make_device(self, device_id: str) -> SimulatedDevice:
        return NumDevice(device_id, self.min_train_time, self.max_train_time)


class NumProcessor(DeviceTaskProcessor):

    def __init__(self, min_train_time=1.0, max_train_time=5.0):
        DeviceTaskProcessor.__init__(self)
        self.min_train_time = min_train_time
        self.max_train_time = max_train_time

    def setup(self, job: JobResponse) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def process_task(self, task: TaskResponse) -> dict:
        task_data = task.task_data
        assert isinstance(task_data, dict)
        model = from_dict(task_data)
        if not isinstance(model, DXO):
            self.logger.error(f"expect model to be DXO but got {type(model)}")
            raise ValueError("bad model data")

        if model.data_kind != "number":
            self.logger.error(f"expect model data kind to be 'number' but got {model.data_kind}")
            raise ValueError("bad model data kind")

        value = model.data.get("value", 0)
        result = value + 1
        result_dxo = DXO(data_kind="number", data={"value": result})
        delay = random.uniform(self.min_train_time, self.max_train_time)
        time.sleep(delay)
        return result_dxo.to_dict()
