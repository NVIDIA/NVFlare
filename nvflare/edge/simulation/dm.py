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
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from nvflare.edge.constants import EdgeApiStatus
from nvflare.edge.simulation.simulated_device import DeviceFactory, DeviceState, SimulatedDevice
from nvflare.edge.web.models.job_request import JobRequest
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.result_report import ResultReport
from nvflare.edge.web.models.result_response import ResultResponse
from nvflare.edge.web.models.task_request import TaskRequest
from nvflare.edge.web.models.task_response import TaskResponse
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.security.logging import secure_format_exception


class DeviceManager:

    def __init__(
        self,
        device_factory: DeviceFactory,
        num_active_devices: int = 1000,
        max_num_devices: int = 10000,
        num_workers: int = 10,
        cycle_length: float = 30,
        device_reuse_rate: float = 0,
    ):
        self.device_factory = device_factory
        self.num_active_devices = num_active_devices
        self.max_num_devices = max_num_devices
        self.num_workers = num_workers
        self.cycle_length = cycle_length
        self.device_reuse_rate = device_reuse_rate
        self.send_f = None
        self.send_kwargs = None

        self.done = False
        self.devices = []
        self.used_devices = []
        self.worker_pool = ThreadPoolExecutor(num_workers)
        self.logger = get_obj_logger(self)
        self.update_lock = threading.Lock()

    def set_send_func(self, send_f, **kwargs):
        if not callable(send_f):
            raise ValueError("send_f is not callable")
        self.send_f = send_f
        self.send_kwargs = kwargs

    def start(self):
        if self.send_f is None:
            raise ValueError("send_f has not been set - please call set_send_func before start")

        for _ in range(self.num_active_devices):
            device = self.device_factory.make_device()
            self.devices.append(device)

        interval = self.cycle_length / len(self.devices)
        if interval > 1.0:
            interval = 1.0

        if interval < 0.01:
            interval = 0.01

        self.logger.info(f"starting device runner - ping interval {interval}")
        while not self.done:
            for i, d in enumerate(self.devices):
                if self.done:
                    return

                assert isinstance(d, SimulatedDevice)
                if d.state != DeviceState.IDLE:
                    continue

                if d.get_job_id():
                    # the device already got a job: ask for task
                    status, resp = self._ask_for_task(d)
                    self.logger.debug(f"tried to get task for device {d.device_id}: {status=}")

                    if status == EdgeApiStatus.OK and resp:
                        assert isinstance(resp, TaskResponse)
                        self.logger.debug(f"TaskResponse status for device {d.device_id}: {resp.status}")

                        if resp.status == EdgeApiStatus.OK:
                            # got a task to do
                            self.logger.debug(f"device {d.device_id} got a task")
                            d.state = DeviceState.LEARNING
                            d.cookie = resp.cookie

                            self.devices[i] = self._get_device_replacement()

                            # submit the task
                            self.worker_pool.submit(self._do_learn, task_data=resp, device=d)
                        elif resp.status in [EdgeApiStatus.RETRY, EdgeApiStatus.NO_TASK]:
                            pass
                        elif resp.status in [EdgeApiStatus.DONE]:
                            # this device is done - discard it
                            self.devices[i] = self._get_device_replacement()
                        else:
                            # ERROR or NO_JOB
                            self.logger.info(f"stop running due to bad TaskResponse status: {resp.status}")
                            return
                    else:
                        self.logger.debug(f"failed to get task ({status=}): will retry")
                else:
                    # the device has no job yet - asking for job
                    status, resp = self._ask_for_job(d)
                    if status == EdgeApiStatus.OK and resp:
                        assert isinstance(resp, JobResponse)
                        if resp.status == EdgeApiStatus.OK:
                            self.logger.debug(f"Device {d.device_id} got job {resp.job_id}")
                            d.set_job(
                                job_id=resp.job_id,
                                job_name=resp.job_name,
                                job_data=resp.job_data,
                                method=resp.method,
                            )
                        elif resp.status in [EdgeApiStatus.RETRY, EdgeApiStatus.NO_JOB]:
                            pass
                        elif resp.status in [EdgeApiStatus.DONE]:
                            # this device is done - discard it
                            self.devices[i] = self._get_device_replacement()
                        else:
                            self.logger.info(f"stop running due to bad JobResponse status: {resp.status}")
                            return
                    else:
                        self.logger.debug(f"failed to get job ({status=}): will retry")

                # pause before next query
                time.sleep(interval)

            # pause before next cycle
            time.sleep(interval)

    def stop(self):
        self.done = True
        self.worker_pool.shutdown()
        self.device_factory.shutdown()

    def _ask_for_task(self, device: SimulatedDevice) -> (str, TaskResponse):
        req = TaskRequest(
            device_info=device.get_device_info(),
            user_info=device.get_user_info(),
            job_id=device.get_job_id(),
            cookie=device.cookie,
        )
        status, resp = self.send_f(req, **self.send_kwargs)
        if resp and not isinstance(resp, TaskResponse):
            self.logger.error(f"received response must be TaskResponse but got {type(resp)}")
            if status == EdgeApiStatus.OK:
                status = EdgeApiStatus.ERROR
        return status, resp

    def _ask_for_job(self, device: SimulatedDevice) -> (str, JobResponse):
        req = JobRequest(
            device_info=device.get_device_info(),
            user_info=device.get_user_info(),
            capabilities=device.get_capabilities(),
        )
        status, resp = self.send_f(req, **self.send_kwargs)
        if resp and not isinstance(resp, JobResponse):
            self.logger.error(f"received response must be JobResponse but got {type(resp)}")
            if status == EdgeApiStatus.OK:
                status = EdgeApiStatus.ERROR
        return status, resp

    def _pick_a_used_device(self):
        # randomly pick a used device
        i = random.randint(0, len(self.used_devices) - 1)
        with self.update_lock:
            return self.used_devices.pop(i)

    def _get_device_replacement(self):
        if not self.used_devices:
            # get a new device
            return self.device_factory.make_device()

        if len(self.devices) + len(self.used_devices) >= self.max_num_devices:
            # enough devices - use a used one
            return self._pick_a_used_device()

        # should we use a used device or make a new one?
        odd = random.uniform(0.0, 1.0)
        if odd < self.device_reuse_rate:
            # use a used device
            return self._pick_a_used_device()
        else:
            # use a new device
            return self.device_factory.make_device()

    def _do_learn(self, task_data: TaskResponse, device: SimulatedDevice):
        try:
            result = device.do_task(task_data)
            status = EdgeApiStatus.OK
        except Exception as ex:
            self.logger.error(f"exception when processing task: {secure_format_exception(ex)}")
            result = {}
            status = EdgeApiStatus.ERROR

        if not isinstance(result, dict):
            self.logger.error(f"bad result from device: expect dict but got {type(result)}")
            result = {}
            status = EdgeApiStatus.ERROR

        report = ResultReport(
            device_info=device.get_device_info(),
            user_info=device.get_user_info(),
            job_id=task_data.job_id,
            task_id=task_data.task_id,
            task_name=task_data.task_name,
            result=result,
            status=status,
            cookie=device.cookie,
        )

        _, resp = self.send_f(report, **self.send_kwargs)
        if resp and not isinstance(resp, ResultResponse):
            self.logger.error(f"received response must be ResultResponse but got {type(resp)}")

        device.state = DeviceState.IDLE

        # put the device in used_devices
        with self.update_lock:
            self.used_devices.append(device)
