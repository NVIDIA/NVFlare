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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.edge.constants import EdgeApiStatus, EdgeEventType
from nvflare.edge.simulation.dm import DeviceManager
from nvflare.edge.simulation.simulated_device import DeviceFactory
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.fuel.utils.validation_utils import check_number_range, check_positive_int, check_positive_number, check_str
from nvflare.widgets.widget import Widget


class DeviceRunner(Widget):

    def __init__(
        self,
        device_factory_id: str,
        num_devices=100,
        num_workers=10,
        max_num_devices=10000,
        cycle_length: float = 30,
        device_reuse_rate: float = 0,
    ):
        Widget.__init__(self)

        check_str("device_factory_id", device_factory_id)
        check_positive_int("num_devices", num_devices)

        check_positive_int("max_num_devices", max_num_devices)
        check_number_range("max_num_devices", max_num_devices, num_devices, 1000000)

        check_positive_int("num_workers", num_workers)
        check_number_range("num_workers", num_workers, 1, num_devices)
        check_number_range("num_workers", num_workers, 1, 100)

        check_positive_number("cycle_length", cycle_length)

        check_number_range("device_reuse_rate", device_reuse_rate, 0.0, 1.0)

        self.device_factory_id = device_factory_id
        self.num_devices = num_devices
        self.max_num_devices = max_num_devices
        self.cycle_length = cycle_length
        self.device_reuse_rate = device_reuse_rate
        self.num_workers = num_workers
        self.manager = None

        self.register_event_handler(EventType.START_RUN, self._dr_start_run)
        self.register_event_handler(EventType.END_RUN, self._dr_end_run)

    def _dr_start_run(self, event_type: str, fl_ctx: FLContext):
        is_leaf = fl_ctx.get_prop(ReservedKey.IS_LEAF)
        if not is_leaf:
            # devices are only for leaf nodes
            return

        self.log_info(fl_ctx, "device runner about to start ...")
        engine = fl_ctx.get_engine()
        factory = engine.get_component(self.device_factory_id)
        if not isinstance(factory, DeviceFactory):
            self.system_panic(
                f"component {self.device_factory_id} must be DeviceFactory but got {type(factory)}",
                fl_ctx,
            )
            return

        manager = DeviceManager(
            device_factory=factory,
            num_active_devices=self.num_devices,
            max_num_devices=self.max_num_devices,
            num_workers=self.num_workers,
            cycle_length=self.cycle_length,
            device_reuse_rate=self.device_reuse_rate,
        )
        manager.set_send_func(self._post_request, engine=engine)
        self.manager = manager

        runner = threading.Thread(target=self._run, daemon=True)
        runner.start()

    def _dr_end_run(self, event_type: str, fl_ctx: FLContext):
        if self.manager:
            self.manager.stop()

    def _post_request(self, request, engine):
        cell_msg = CellMessage(payload=request)
        with engine.new_context() as fl_ctx:
            assert isinstance(fl_ctx, FLContext)
            fl_ctx.set_prop(FLContextKey.CELL_MESSAGE, cell_msg, private=True, sticky=False)
            self.fire_event(EdgeEventType.EDGE_REQUEST_RECEIVED, fl_ctx)
            reply_dict = fl_ctx.get_prop(FLContextKey.TASK_RESULT)

            if reply_dict is None:
                # client not ready yet
                return EdgeApiStatus.OK, None

            if not isinstance(reply_dict, dict):
                raise RuntimeError(f"prop {FLContextKey.TASK_RESULT} should be dict but got {type(reply_dict)}")

            status = reply_dict.get("status", EdgeApiStatus.OK)
            response = reply_dict.get("response")
            return status, response

    def _run(self):
        self.manager.start()
