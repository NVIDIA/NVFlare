# Copyright (c) 2022, NVIDIA CORPORATION.
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

import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nvflare.apis.client import Client
from nvflare.apis.dxo import from_shareable
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import ClientTask, Controller, Task
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal


class SupportedTasks(object):
    HISTOGRAM = "histogram"


class AnalysisController(Controller):
    def __init__(self, min_clients: int = 1, task_name_histogram: str = SupportedTasks.HISTOGRAM):
        """Controller for federated analysis.

        Args:
            min_clients: how many statistics to gather before computing the global statisitcs.
            task_name_histogram: task name for histogram computation.
        """
        super().__init__()
        self.histograms = dict()
        self._min_clients = min_clients
        self.run_dir = None
        self.task_name_histogram = task_name_histogram

    def start_controller(self, fl_ctx: FLContext):
        self.run_dir = os.path.join(fl_ctx.get_prop(FLContextKey.APP_ROOT), "..")

    def stop_controller(self, fl_ctx: FLContext):
        pass

    def process_result_of_unknown_task(
        self,
        client: Client,
        task_name: str,
        client_task_id: str,
        result: Shareable,
        fl_ctx: FLContext,
    ):
        self.log_warning(fl_ctx, f"Unknown task: {task_name} from client {client.name}.")

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        self.log_info(fl_ctx, "Analysis control flow started.")
        if abort_signal.triggered:
            return
        task = Task(name=self.task_name_histogram, data=Shareable(), result_received_cb=self._process_result_histogram)
        self.broadcast_and_wait(
            task=task,
            min_responses=self._min_clients,
            fl_ctx=fl_ctx,
            wait_time_after_min_received=1,
            abort_signal=abort_signal,
        )
        if abort_signal.triggered:
            return

        # Visualize the histograms
        self.visualize(fl_ctx)

        self.log_info(fl_ctx, "Analysis control flow finished.")

    def _process_result_histogram(self, client_task: ClientTask, fl_ctx: FLContext):
        task_name = client_task.task.name
        client_name = client_task.client.name
        self.log_info(fl_ctx, f"Processing {task_name} result from client {client_name}")
        result = client_task.result
        rc = result.get_return_code()

        if rc == ReturnCode.OK:
            dxo = from_shareable(result)
            data_stat_dict = dxo.data
            self.log_info(fl_ctx, f"Received result entries {data_stat_dict.keys()}")
            self.histograms.update({client_name: data_stat_dict})
            if "histogram" in data_stat_dict.keys():
                self.log_info(fl_ctx, f"Client {client_name} finished {task_name} returned histogram.")
            else:
                self.log_info(fl_ctx, f"Client {client_name} finished {task_name} but return no histogram.")
        else:
            self.log_error(fl_ctx, f"Ignore the client train result. {task_name} tasked returned error code: {rc}")

        # Cleanup task result
        client_task.result = None

    def visualize(self, fl_ctx: FLContext):
        self.log_warning("Result visualization not implemented.")
        pass
