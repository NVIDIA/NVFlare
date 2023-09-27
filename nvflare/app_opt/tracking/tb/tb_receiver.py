# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from typing import List, Optional

from torch.utils.tensorboard import SummaryWriter

from nvflare.apis.analytix import AnalyticsData, AnalyticsDataType
from nvflare.apis.dxo import from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.widgets.streaming import AnalyticsReceiver

FUNCTION_MAPPING = {
    AnalyticsDataType.SCALAR: "add_scalar",
    AnalyticsDataType.TEXT: "add_text",
    AnalyticsDataType.IMAGE: "add_image",
    AnalyticsDataType.SCALARS: "add_scalars",
    AnalyticsDataType.PARAMETER: "add_scalar",
    AnalyticsDataType.PARAMETERS: "add_scalars",
    AnalyticsDataType.METRIC: "add_scalar",
    AnalyticsDataType.METRICS: "add_scalars",
}


class TBAnalyticsReceiver(AnalyticsReceiver):
    def __init__(self, tb_folder="tb_events", events: Optional[List[str]] = None):
        """Receives analytics data to save to TensorBoard.

        Args:
            tb_folder (str): the folder to store tensorboard files.
            events (optional, List[str]): A list of events to be handled by this receiver.

        .. code-block:: text
            :caption: Folder structure

            Inside run_XX folder:
              - workspace
                - run_01 (already created):
                  - output_dir (default: tb_events):
                    - peer_name_1:
                    - peer_name_2:

                - run_02 (already created):
                  - output_dir (default: tb_events):
                    - peer_name_1:
                    - peer_name_2:

        """
        super().__init__(events=events)
        self.writers_table = {}
        self.tb_folder = tb_folder
        self.root_log_dir = None

    def initialize(self, fl_ctx: FLContext):
        workspace = fl_ctx.get_engine().get_workspace()
        run_dir = workspace.get_run_dir(fl_ctx.get_job_id())
        root_log_dir = os.path.join(run_dir, self.tb_folder)
        os.makedirs(root_log_dir, exist_ok=True)
        self.root_log_dir = root_log_dir

    def save(self, fl_ctx: FLContext, shareable: Shareable, record_origin):
        dxo = from_shareable(shareable)
        analytic_data = AnalyticsData.from_dxo(dxo)
        if not analytic_data:
            return

        writer = self.writers_table.get(record_origin)
        if writer is None:
            peer_log_dir = os.path.join(self.root_log_dir, record_origin)
            writer = SummaryWriter(log_dir=peer_log_dir)
            self.writers_table[record_origin] = writer

        # do different things depending on the type in dxo
        self.log_debug(
            fl_ctx,
            f"save data {analytic_data} from {record_origin}",
            fire_event=False,
        )
        func_name = FUNCTION_MAPPING.get(analytic_data.data_type, None)
        if func_name is None:
            self.log_error(fl_ctx, f"The data_type {analytic_data.data_type} is not supported.", fire_event=False)
            return

        func = getattr(writer, func_name)
        if analytic_data.step:
            func(analytic_data.tag, analytic_data.value, analytic_data.step)
        else:
            func(analytic_data.tag, analytic_data.value)

    def finalize(self, fl_ctx: FLContext):
        for writer in self.writers_table.values():
            writer.flush()
            writer.close()
