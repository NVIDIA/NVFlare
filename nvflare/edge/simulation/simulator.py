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
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from nvflare.edge.constants import CookieKey, EdgeApiStatus, SpecialDeviceId
from nvflare.edge.simulation.simulated_device import DeviceFactory, DeviceState, SimulatedDevice
from nvflare.edge.web.models.job_request import JobRequest
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.result_report import ResultReport
from nvflare.edge.web.models.result_response import ResultResponse
from nvflare.edge.web.models.selection_request import SelectionRequest
from nvflare.edge.web.models.selection_response import SelectionResponse
from nvflare.edge.web.models.task_request import TaskRequest
from nvflare.edge.web.models.task_response import TaskResponse
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.security.logging import secure_format_exception


class Simulator:

    def __init__(
        self,
        job_name: str,
        device_factory: DeviceFactory,
        num_devices: int = 10000,
        num_workers: int = 10,
        get_job_timeout: float = 60.0,
    ):
        """Constructor of Simulator.

        Args:
            device_factory: object for creating new devices
            num_devices: max number of devices to be created
            num_workers: number of threads for doing tasks
        """
        self.job_name = job_name
        self.device_factory = device_factory
        self.num_devices = num_devices
        self.num_workers = num_workers
        self.get_job_timeout = get_job_timeout
        self.send_f = None
        self.send_kwargs = None

        self.done = False

        # devices that are currently busy doing task
        self.busy_devices = {}  # device_id => selection id

        # devices that have finished tasks
        self.used_devices = {}  # device_id => selection id

        # all created devices
        self.all_devices = {}  # device_id => SimulatedDevice

        # thread pool for doing tasks
        self.worker_pool = ThreadPoolExecutor(num_workers)

        self.logger = get_obj_logger(self)
        self.update_lock = threading.Lock()
        self.device_id_prefix = str(uuid.uuid4())

    def set_send_func(self, send_f, **kwargs):
        """Set the function for sending request to Flare

        Args:
            send_f: the function to be set
            **kwargs: args to be passed to the function when invoked

        Returns: None

        """
        if not callable(send_f):
            raise ValueError("send_f is not callable")
        self.send_f = send_f
        self.send_kwargs = kwargs

    def _determine_my_devices(self, selected_devices: dict):
        """Determines active devices for the next cycle.

        Args:
            selected_devices: the devices that have been selected for task

        Returns: a dict of devices for next query cycle

        """
        result = {}

        # Selected devices have the highest priority.
        # Note that selected_devices are created by the server. It may contain devices of all sources.
        # We only handle our own devices.
        for did, sid in selected_devices.items():
            assert isinstance(did, str)
            if not did.startswith(self.device_id_prefix):
                # ignore since it's not our device
                continue

            if did in self.busy_devices:
                # This device is working a task already
                continue

            if did in self.used_devices and self.used_devices[did] == sid:
                # we already reported result, but it has not been processed by server
                continue

            # include this device
            result[did] = sid
            self.logger.debug(f"added selected device to active: {did} {sid=}")

        for did in result.keys():
            # Once a device is selected for query, we remove it from used_devices
            self.used_devices.pop(did, None)

        return result

    def _control_flow(self):
        interval = 0.5
        self.logger.info(f"starting device simulator: {interval=}")

        # get job using a dummy device
        device = self.device_factory.make_device(SpecialDeviceId.DUMMY)
        start_time = time.time()
        while True:
            resp = self._ask_for_job(device)
            if resp:
                assert isinstance(resp, JobResponse)
                if resp.status == EdgeApiStatus.OK:
                    self.logger.debug(f"Got job {resp.job_id}")
                    job_response = resp
                    job_id = resp.job_id
                    break
                elif resp.status not in [EdgeApiStatus.RETRY, EdgeApiStatus.NO_JOB]:
                    self.logger.info(f"stop running due to bad JobResponse status: {resp.status}")
                    return

            if time.time() - start_time > self.get_job_timeout:
                self.logger.info(f"cannot get a job for {self.get_job_timeout} seconds - exiting")
                return

            self.logger.debug("failed to get job: will retry")
            time.sleep(1.0)

        device_id_for_selection = f"{self.device_id_prefix}{SpecialDeviceId.MAX_INDICATOR}{self.num_devices}"
        cycle_num = 0
        while not self.done:
            cycle_num += 1
            self.logger.info(f"Starting query cycle: {cycle_num}")

            while True:
                if self.done:
                    return

                resp = self._ask_for_selection(job_id, device_id_for_selection)
                if resp:
                    assert isinstance(resp, SelectionResponse)
                    if resp.status == EdgeApiStatus.NO_JOB:
                        # all done
                        self.logger.info(f"no selection: job {job_id} is gone - exiting")
                        return

                    if resp.status == EdgeApiStatus.OK:
                        # no longer need to send device range
                        device_id_for_selection = SpecialDeviceId.DUMMY

                    if resp.selection:
                        assert isinstance(resp.selection, dict)
                        selected_devices = resp.selection

                        # find my devices
                        my_devices = self._determine_my_devices(selected_devices)
                        if my_devices:
                            break

                self.logger.debug("no selection for me - will retry")
                time.sleep(1.0)

            self.logger.debug(f"got my selected devices: {my_devices}")
            for did, sid in my_devices.items():
                if self.done:
                    return

                device = self.all_devices.get(did)
                if not device:
                    device = self._make_new_device(did)
                assert isinstance(device, SimulatedDevice)

                device.set_job(
                    job_id=job_response.job_id,
                    job_name=job_response.job_name,
                    job_data=job_response.job_data,
                    method=job_response.method,
                )

                # ask for task
                resp = self._ask_for_task(device)
                self.logger.debug(f"tried to get task for device {did}: {resp}")

                if resp.status == EdgeApiStatus.OK:
                    assert isinstance(resp, TaskResponse)
                    self.logger.debug(f"TaskResponse status for device {did}: {resp.status}")

                    if resp.status == EdgeApiStatus.OK:
                        # got a task to do
                        self.logger.debug(f"device {did} got a task")
                        device.state = DeviceState.LEARNING
                        device.cookie = resp.cookie

                        # submit the task
                        selection_id = sid
                        if isinstance(resp.cookie, dict):
                            selection_id = resp.cookie.get(CookieKey.DEVICE_SELECTION_ID, sid)
                        self.worker_pool.submit(
                            self._do_learn, task_data=resp, device=device, selection_id=selection_id
                        )

                        # mark the device to be busy and remove from device candidates for next cycle.
                        self.busy_devices[did] = selection_id
                elif resp.status in [EdgeApiStatus.NO_JOB, EdgeApiStatus.RETRY]:
                    # the job is gone
                    self.logger.info("job is gone when getting task - exiting")
                    return
                elif resp.status == EdgeApiStatus.DONE:
                    # this device is done - job is done
                    device.job_id = None
                else:
                    # ERROR
                    self.logger.info(f"stop running due to bad TaskResponse status: {resp.status}")
                    return

                # pause before next query
                time.sleep(interval)

    def start(self):
        if self.send_f is None:
            raise ValueError("send_f has not been set - please call set_send_func before start")

        self._control_flow()

        # Stop all tasks if any
        self.worker_pool.shutdown()

        # shut down all devices
        for d in self.all_devices.values():
            assert isinstance(d, SimulatedDevice)
            d.shutdown()

        self.device_factory.shutdown()

    def stop(self):
        """Stop the simulator.

        Returns: None

        """
        # set the flag to stop the query loop
        self.done = True

    def _send_request(self, req, device, default_resp, **kwargs):
        try:
            return self.send_f(req, device, **self.send_kwargs)
        except Exception as ex:
            self.logger.warning(f"exception sending request: {secure_format_exception(ex)}")
            return default_resp

    def _ask_for_task(self, device: SimulatedDevice) -> TaskResponse:
        """Send a request to Flare to ask for a task for a device

        Args:
            device: the device that the request is for

        Returns: TaskResponse

        """
        req = TaskRequest(
            device_info=device.get_device_info(),
            user_info=device.get_user_info(),
            job_id=device.get_job_id(),
            cookie=device.cookie,
        )
        resp = self._send_request(req, device, TaskResponse(EdgeApiStatus.RETRY), **self.send_kwargs)
        self.logger.debug(f"got task response: {resp}")
        return resp

    def _ask_for_selection(self, job_id: str, device_id) -> SelectionResponse:
        """Send a request to Flare for ask for the current device selection.
        Note: this is used for simulation purpose. Real devices don't make this request.

        Returns: SelectionResponse

        """
        device = self.device_factory.make_device(device_id)
        req = SelectionRequest(device.get_device_info(), job_id)
        resp = self._send_request(req, device, SelectionResponse(EdgeApiStatus.RETRY), **self.send_kwargs)
        self.logger.debug(f"got selection response: {resp}")
        return resp

    def _ask_for_job(self, device: SimulatedDevice) -> JobResponse:
        """Send a request to Flare to ask for a Job for the specified device

        Args:
            device: the device that the request is for.

        Returns:

        """
        req = JobRequest(
            job_name=self.job_name,
            device_info=device.get_device_info(),
            user_info=device.get_user_info(),
            capabilities=device.get_capabilities(),
        )
        resp = self._send_request(req, device, JobResponse(EdgeApiStatus.RETRY), **self.send_kwargs)
        self.logger.debug(f"got job response: {resp}")
        return resp

    def _make_new_device(self, device_id: str):
        """Create a new device for inclusion to active device list.

        Returns: a device

        """
        device = self.device_factory.make_device(device_id)
        self.all_devices[device.device_id] = device
        return device

    def _do_learn(self, task_data: TaskResponse, device: SimulatedDevice, selection_id):
        """Do the task.

        Args:
            task_data: task data
            device: device to do the task
            selection_id: selection id of the device

        Returns:

        """
        self.logger.info(f"Device {device.device_id} is selected ({selection_id}): started training ")
        try:
            result = device.do_task(task_data)
            status = EdgeApiStatus.OK
        except Exception as ex:
            self.logger.error(f"exception processing task: {secure_format_exception(ex)}")
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

        # report the result to Flare
        self.logger.info(f"Device {device.device_id} finished training")
        self.logger.debug(f"Device {device.device_id} result: {report}")
        resp = self._send_request(report, device, ResultResponse(EdgeApiStatus.RETRY), **self.send_kwargs)
        self.logger.debug(f"got result response: {resp}")

        if resp and not isinstance(resp, ResultResponse):
            self.logger.error(f"received response must be ResultResponse but got {type(resp)}")

        device.state = DeviceState.IDLE

        # put the device in used_devices
        with self.update_lock:
            self.busy_devices.pop(device.device_id, None)
            self.used_devices[device.device_id] = selection_id
