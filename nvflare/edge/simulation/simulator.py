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

from nvflare.edge.constants import CookieKey, EdgeApiStatus
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
        device_factory: DeviceFactory,
        num_active_devices: int = 1000,
        num_devices: int = 10000,
        num_workers: int = 10,
        cycle_duration: float = 30,
        device_reuse_rate: float = 0,
    ):
        """Constructor of Simulator.

        Args:
            device_factory: object for creating new devices
            num_active_devices: number of active devices
            num_devices: max number of devices to be created
            num_workers: number of threads for doing tasks
            cycle_duration: time duration for one query cycle
            device_reuse_rate: odds for reusing used devices
        """
        self.device_factory = device_factory
        self.num_active_devices = num_active_devices
        self.num_devices = num_devices
        self.num_workers = num_workers
        self.cycle_duration = cycle_duration
        self.device_reuse_rate = device_reuse_rate
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

    def _determine_active_devices(self, selected_devices: dict, active_device_candidates: dict):
        """Determines active devices for the next cycle.

        Args:
            selected_devices: the devices that have been selected for task
            active_device_candidates: active device candidates. They are the active devices used for previous cycle.

        Returns: a dict of devices for next query cycle

        """
        result = {}

        # Selected devices have the highest priority.
        # Note that selected_devices are created by the server. It may contain devices of all sources.
        # We only handle our own devices.
        for did, sid in selected_devices.items():
            if did not in self.all_devices:
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

            if len(result) >= self.num_active_devices:
                # enough devices
                return result

        # active_device_candidates next
        # These are active devices from the current cycle. Once a device is active, we try to keep it active
        # until it is assigned a task.
        for did, sid in active_device_candidates.items():
            if did not in result:
                result[did] = sid
                if len(result) >= self.num_active_devices:
                    # enough devices
                    return result

        for did in result.keys():
            # Once a device is selected for query, we remove it from used_devices
            self.used_devices.pop(did, None)

        while len(result) < self.num_active_devices:
            # Pick a device (either by making a new device or pick one from used devices) until we've got
            # required number of active devices.
            device = self._get_a_device()
            result[device.device_id] = 0

        return result

    def start(self):
        if self.send_f is None:
            raise ValueError("send_f has not been set - please call set_send_func before start")

        interval = self.cycle_duration / self.num_active_devices
        if interval > 1.0:
            interval = 1.0

        if interval < 0.01:
            interval = 0.01

        self.logger.info(f"starting device simulator: {interval=} {self.cycle_duration=}")
        active_device_candidates = {}
        cycle_num = 0
        job_id = None

        while not self.done:
            cycle_num += 1
            self.logger.info(f"Starting query cycle: {cycle_num}")

            # Get selections from Flare for two reasons:
            # 1. Proactively place selected devices in active_devices can make task assignment happen quickly;
            # 2. More importantly, the Server could assign task to a device used long time ago. If we only reuse
            # current active devices, then the device used before will never be active, and hence will never get
            # the assigned task!
            selected_devices = {}
            if self.all_devices and job_id:
                _, resp = self._ask_for_selection(job_id)
                if resp:
                    assert isinstance(resp, SelectionResponse)
                    if resp.selection:
                        assert isinstance(resp.selection, dict)
                        selected_devices = resp.selection

            if selected_devices:
                self.logger.debug(f"got selected devices: {selected_devices}")

            with self.update_lock:
                # determine active devices for query
                active_devices = self._determine_active_devices(selected_devices, active_device_candidates)

            self.logger.debug(f"got active devices: {len(active_devices)}")
            for did, sid in active_devices.items():
                if self.done:
                    return

                device = self.all_devices[did]
                assert isinstance(device, SimulatedDevice)

                # by default, every device in the current cycle is also candidate for the next cycle.
                active_device_candidates[did] = sid

                if device.get_job_id():
                    # the device already got a job: ask for task
                    status, resp = self._ask_for_task(device)
                    self.logger.debug(f"tried to get task for device {did}: {status=}")

                    if status == EdgeApiStatus.OK and resp:
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
                            active_device_candidates.pop(did)
                        elif resp.status in [EdgeApiStatus.RETRY, EdgeApiStatus.NO_TASK]:
                            pass
                        elif resp.status in [EdgeApiStatus.DONE]:
                            # this device is done - discard it
                            pass
                        else:
                            # ERROR or NO_JOB
                            self.logger.info(f"stop running due to bad TaskResponse status: {resp.status}")
                            return
                    else:
                        self.logger.debug(f"failed to get task ({status=}): will retry")
                else:
                    # the device has no job yet - asking for job
                    self.logger.debug(f"asking for job for device {did}")
                    status, resp = self._ask_for_job(device)
                    if status == EdgeApiStatus.OK and resp:
                        assert isinstance(resp, JobResponse)
                        if resp.status == EdgeApiStatus.OK:
                            self.logger.debug(f"Device {did} got job {resp.job_id}")
                            device.set_job(
                                job_id=resp.job_id,
                                job_name=resp.job_name,
                                job_data=resp.job_data,
                                method=resp.method,
                            )
                            if not job_id:
                                job_id = resp.job_id
                            elif job_id != resp.job_id:
                                self.logger.warning(f"multiple jobs detected: {job_id}, {resp.job_id}")
                        elif resp.status in [EdgeApiStatus.RETRY, EdgeApiStatus.NO_JOB]:
                            pass
                        elif resp.status in [EdgeApiStatus.DONE]:
                            # this device is done
                            pass
                        else:
                            self.logger.info(f"stop running due to bad JobResponse status: {resp.status}")
                            return
                    else:
                        self.logger.debug(f"failed to get job ({status=}): will retry")

                # pause before next query
                time.sleep(interval)

            # pause before next cycle
            self.logger.info(f"Finished cycle {cycle_num}")
            time.sleep(interval)

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

    def _ask_for_task(self, device: SimulatedDevice) -> (str, TaskResponse):
        """Send a request to Flare to ask for a task for a device

        Args:
            device: the device that the request is for

        Returns: tuple of (status, TaskResponse)

        """
        req = TaskRequest(
            device_info=device.get_device_info(),
            user_info=device.get_user_info(),
            job_id=device.get_job_id(),
            cookie=device.cookie,
        )
        status, resp = self.send_f(req, device, **self.send_kwargs)
        if resp and not isinstance(resp, TaskResponse):
            self.logger.error(f"received response must be TaskResponse but got {type(resp)}")
            if status == EdgeApiStatus.OK:
                status = EdgeApiStatus.ERROR
        return status, resp

    def _ask_for_selection(self, job_id: str) -> (str, SelectionResponse):
        """Send a request to Flare for ask for the current device selection.
        Note: this is used for simulation purpose. Real devices don't make this request.

        Returns: tuple of (status, SelectionResponse)

        """
        first_key = next(iter(self.all_devices))
        device = self.all_devices[first_key]
        req = SelectionRequest(device.get_device_info(), job_id)
        status, resp = self.send_f(req, device, **self.send_kwargs)
        if resp and not isinstance(resp, SelectionResponse):
            self.logger.error(f"received response must be SelectionResponse but got {type(resp)}")
            if status == EdgeApiStatus.OK:
                status = EdgeApiStatus.ERROR
            resp = None
        return status, resp

    def _ask_for_job(self, device: SimulatedDevice) -> (str, JobResponse):
        """Send a request to Flare to ask for a Job for the specified device

        Args:
            device: the device that the request is for.

        Returns:

        """
        req = JobRequest(
            device_info=device.get_device_info(),
            user_info=device.get_user_info(),
            capabilities=device.get_capabilities(),
        )
        status, resp = self.send_f(req, device, **self.send_kwargs)
        if resp and not isinstance(resp, JobResponse):
            self.logger.error(f"received response must be JobResponse but got {type(resp)}")
            if status == EdgeApiStatus.OK:
                status = EdgeApiStatus.ERROR
        return status, resp

    def _pick_a_used_device(self):
        """Pick a used device for inclusion to active device list.
        Remove the picked device from used_devices.

        Returns: a device

        """
        # pick the first
        first_key = next(iter(self.used_devices))
        device_id = self.used_devices[first_key]
        self.used_devices.pop(device_id)
        return self.all_devices[device_id]

    def _make_new_device(self):
        """Create a new device for inclusion to active device list.

        Returns: a device

        """
        device = self.device_factory.make_device()
        self.all_devices[device.device_id] = device
        return device

    def _get_a_device(self):
        """Get a device for inclusion to active device list.
        It gets the device either from the used_devices or be making a new device.

        Returns: a device.

        """
        if not self.used_devices:
            # no used devices - make a new device
            return self._make_new_device()

        if len(self.all_devices) >= self.num_devices:
            # We've got max allowed devices - have to pick a used one
            return self._pick_a_used_device()

        # Should we use a used device or make a new one?
        # This is decided by device reuse rate.
        odd = random.uniform(0.0, 1.0)
        if odd < self.device_reuse_rate:
            # use a used device
            return self._pick_a_used_device()
        else:
            # use a new device
            return self._make_new_device()

    def _do_learn(self, task_data: TaskResponse, device: SimulatedDevice, selection_id):
        """Do the task.

        Args:
            task_data: task data
            device: device to do the task
            selection_id: selection id of the device

        Returns:

        """
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
        _, resp = self.send_f(report, device, **self.send_kwargs)
        if resp and not isinstance(resp, ResultResponse):
            self.logger.error(f"received response must be ResultResponse but got {type(resp)}")

        device.state = DeviceState.IDLE

        # put the device in used_devices
        with self.update_lock:
            self.busy_devices.pop(device.device_id, None)
            self.used_devices[device.device_id] = selection_id
