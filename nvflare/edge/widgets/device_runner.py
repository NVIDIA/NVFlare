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
from concurrent.futures import ThreadPoolExecutor

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.edge.constants import EdgeApiStatus, EdgeEventType, Status
from nvflare.edge.simulated_device import DeviceFactory, DeviceState, SimulatedDevice
from nvflare.edge.web.models.result_report import ResultReport
from nvflare.edge.web.models.result_response import ResultResponse
from nvflare.edge.web.models.task_request import TaskRequest
from nvflare.edge.web.models.task_response import TaskResponse
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.fuel.utils.validation_utils import check_number_range, check_positive_int, check_str
from nvflare.security.logging import secure_format_exception
from nvflare.widgets.widget import Widget


class DeviceRunner(Widget):

    def __init__(self, device_factory_id: str, num_devices=100, num_workers=10):
        Widget.__init__(self)

        check_str("device_factory_id", device_factory_id)
        check_positive_int("num_devices", num_devices)
        check_positive_int("num_workers", num_workers)
        check_number_range("num_workers", num_workers, 1, num_devices)
        check_number_range("num_workers", num_workers, 1, 100)

        self.device_factory_id = device_factory_id
        self.num_devices = num_devices
        self.device_factory = None
        self.done = False
        self.job_id = None
        self.engine = None
        self.devices = []
        self.worker_pool = ThreadPoolExecutor(num_workers)

        self.register_event_handler(EventType.ABOUT_TO_START_RUN, self._dr_about_to_start)
        self.register_event_handler(EventType.START_RUN, self._dr_start_run)
        self.register_event_handler(EventType.END_RUN, self._dr_end_run)

    def _dr_about_to_start(self, event_type: str, fl_ctx: FLContext):
        self.log_info(fl_ctx, "device runner about to start ...")
        engine = fl_ctx.get_engine()
        factory = engine.get_component(self.device_factory_id)
        if not isinstance(factory, DeviceFactory):
            self.system_panic(
                f"component {self.device_factory_id} must be DeviceFactory but got {type(factory)}",
                fl_ctx,
            )
            return

        self.engine = engine
        self.device_factory = factory
        self.job_id = fl_ctx.get_job_id()
        self.log_info(fl_ctx, f"got device factory {type(factory)} for job {self.job_id}")

    def _dr_start_run(self, event_type: str, fl_ctx: FLContext):
        for _ in range(self.num_devices):
            try:
                device = self.device_factory.make_device()
            except Exception as ex:
                self.system_panic(
                    f"{type(self.device_factory)} failed to make device: {secure_format_exception(ex)}", fl_ctx
                )
                return

            if not isinstance(device, SimulatedDevice):
                self.system_panic(
                    f"bad device from {type(self.device_factory)}: expect SimulatedDevice but got {type(device)}",
                    fl_ctx,
                )
                return

            self.devices.append(device)

        self.log_info(fl_ctx, f"created {len(self.devices)} simulated devices")
        self.runner = threading.Thread(target=self._run, daemon=True, args=(fl_ctx,))
        self.runner.start()

    def _dr_end_run(self, event_type: str, fl_ctx: FLContext):
        self.done = True
        self.worker_pool.shutdown(wait=False, cancel_futures=True)
        self.device_factory.shutdown()

    def _post_request(self, request):
        cell_msg = CellMessage(payload=request)
        with self.engine.new_context() as fl_ctx:
            assert isinstance(fl_ctx, FLContext)
            fl_ctx.set_prop(FLContextKey.CELL_MESSAGE, cell_msg, private=True, sticky=False)
            self.fire_event(EdgeEventType.EDGE_REQUEST_RECEIVED, fl_ctx)
            reply_dict = fl_ctx.get_prop(FLContextKey.TASK_RESULT)

            if reply_dict is None:
                # client not ready yet
                return ReturnCode.OK, None

            if not isinstance(reply_dict, dict):
                raise RuntimeError(f"prop {FLContextKey.TASK_RESULT} should be dict but got {type(reply_dict)}")

            status = reply_dict.get("status", ReturnCode.OK)
            response = reply_dict.get("response")
            return status, response

    def _ask_for_task(self, device: SimulatedDevice) -> (str, TaskResponse):
        req = TaskRequest(
            device_info=device.get_device_info(),
            user_info=device.get_user_info(),
            job_id=self.job_id,
            cookie=device.cookie,
        )
        status, resp = self._post_request(req)
        if resp and not isinstance(resp, TaskResponse):
            self.logger.error(f"received response must be TaskResponse but got {type(resp)}")
            if status == ReturnCode.OK:
                status = ReturnCode.ERROR
        return status, resp

    def _run(self, fl_ctx: FLContext):
        interval = 30 / len(self.devices)
        if interval > 1.0:
            interval = 1.0

        self.log_info(fl_ctx, f"starting device runner - ping interval {interval}")
        while not self.done:
            for d in self.devices:
                assert isinstance(d, SimulatedDevice)
                if d.state == DeviceState.IDLE:
                    # ask for task
                    status, resp = self._ask_for_task(d)

                    if status == ReturnCode.OK and resp:
                        assert isinstance(resp, TaskResponse)
                        if resp.status == EdgeApiStatus.OK:
                            # got a task to do
                            self.logger.info(f"device {d.device_id} got a task")
                            d.state = DeviceState.LEARNING
                            d.cookie = resp.cookie

                            # submit the task
                            self.worker_pool.submit(self._do_learn, task_data=resp, device=d)
                        elif resp.status in [EdgeApiStatus.RETRY, EdgeApiStatus.NO_TASK]:
                            pass
                        elif resp.status in [EdgeApiStatus.DONE]:
                            # this device is done
                            d.state = DeviceState.DONE
                        else:
                            # ERROR or NO_JOB
                            self.logger.info(f"stop running due to response status: {resp.status}")
                            return

                    time.sleep(interval)
                    if self.done:
                        return

            time.sleep(interval)

    def _do_learn(self, task_data: TaskResponse, device: SimulatedDevice):
        try:
            result = device.do_task(task_data)
            status = Status.OK
        except Exception as ex:
            self.logger.error(f"exception when processing task: {secure_format_exception(ex)}")
            result = None
            status = Status.PROCESS_EXCEPTION

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

        _, resp = self._post_request(report)
        if resp and not isinstance(resp, ResultResponse):
            self.logger.error(f"received response must be ResultResponse but got {type(resp)}")

        device.state = DeviceState.IDLE
