# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import tensorflow as tf

from nvflare.apis.analytix import AnalyticsData, AnalyticsDataType
from nvflare.apis.dxo import from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.widgets.streaming import AnalyticsReceiver


def _create_new_data(key, value, sender):
    if isinstance(value, (int, float)):
        data_type = AnalyticsDataType.SCALAR
    elif isinstance(value, str):
        data_type = AnalyticsDataType.TEXT
    else:
        return None

    return AnalyticsData(key=key, value=value, data_type=data_type, sender=sender)


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
        self.log_info(
            fl_ctx,
            f"Tensorboard records can be found in {self.root_log_dir} you can view it using `tensorboard --logdir={self.root_log_dir}`",
        )

    def _convert_to_records(self, analytic_data: AnalyticsData, fl_ctx: FLContext) -> List[AnalyticsData]:
        # break dict of stuff to smaller items to support
        # AnalyticsDataType.PARAMETER and AnalyticsDataType.PARAMETERS
        records = []

        if analytic_data.data_type in (AnalyticsDataType.PARAMETER, AnalyticsDataType.PARAMETERS):
            items = (
                analytic_data.value.items()
                if analytic_data.data_type == AnalyticsDataType.PARAMETERS
                else [(analytic_data.tag, analytic_data.value)]
            )
            for k, v in items:
                new_data = _create_new_data(k, v, analytic_data.sender)
                if new_data is None:
                    self.log_warning(fl_ctx, f"Entry {k} of type {type(v)} is not supported.", fire_event=False)
                else:
                    records.append(new_data)
        elif analytic_data.data_type in (AnalyticsDataType.SCALARS, AnalyticsDataType.METRICS):
            data_type = (
                AnalyticsDataType.SCALAR
                if analytic_data.data_type == AnalyticsDataType.SCALARS
                else AnalyticsDataType.METRIC
            )
            records.extend(
                AnalyticsData(key=k, value=v, data_type=data_type, sender=analytic_data.sender)
                for k, v in analytic_data.value.items()
            )
        else:
            records.append(analytic_data)

        return records

    def save(self, fl_ctx: FLContext, shareable: Shareable, record_origin):
        dxo = from_shareable(shareable)
        analytic_data = AnalyticsData.from_dxo(dxo)
        if not analytic_data:
            return

        writer = self.writers_table.get(record_origin)
        if writer is None:
            peer_log_dir = os.path.join(self.root_log_dir, record_origin)
            writer = tf.summary.create_file_writer(peer_log_dir)
            self.writers_table[record_origin] = writer

        # do different things depending on the type in dxo
        self.log_info(
            fl_ctx,
            f"try to save data {analytic_data} from {record_origin}",
            fire_event=False,
        )

        data_records = self._convert_to_records(analytic_data, fl_ctx)

        with writer.as_default():
            for data_record in data_records:
                if data_record.data_type in (AnalyticsDataType.METRIC, AnalyticsDataType.SCALAR):
                    tf.summary.scalar(data_record.tag, data_record.value, data_record.step)
                elif data_record.data_type == AnalyticsDataType.TEXT:
                    tf.summary.text(data_record.tag, data_record.value, data_record.step)
                elif data_record.data_type == AnalyticsDataType.IMAGE:
                    tf.summary.image(data_record.tag, data_record.value, data_record.step)
                else:
                    self.log_warning(
                        fl_ctx, f"The data_type {data_record.data_type} is not supported.", fire_event=False
                    )

    def finalize(self, fl_ctx: FLContext):
        for writer in self.writers_table.values():
            tf.summary.flush(writer)
