import os
from typing import List, Optional

import tensorflow as tf

from nvflare.apis.analytix import AnalyticsData, AnalyticsDataType
from nvflare.apis.dxo import from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.widgets.streaming import AnalyticsReceiver

FUNCTION_MAPPING = {
    AnalyticsDataType.SCALAR: "scalar",
    AnalyticsDataType.TEXT: "text",
    AnalyticsDataType.IMAGE: "image",
    AnalyticsDataType.SCALARS: "scalars",
    AnalyticsDataType.METRIC: "scalar",
    AnalyticsDataType.METRICS: "scalars",
}


def _create_new_data(key, value, sender):
    if isinstance(value, (int, float)):
        data_type = AnalyticsDataType.SCALAR
    elif isinstance(value, str):
        data_type = AnalyticsDataType.TEXT
    else:
        return None

    return AnalyticsData(key=key, value=value, data_type=data_type, sender=sender)


class TFTBAnalyticsReceiver(AnalyticsReceiver):
    def __init__(self, tb_folder="tb_events", events: Optional[List[str]] = None):
        """Receives analytics data to save to TensorBoard using TensorFlow's summary writer.

        Args:
            tb_folder (str): the folder to store tensorboard files.
            events (optional, List[str]): A list of events to be handled by this receiver.
        """
        super().__init__(events=events, client_side_supported=True)
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
        records = []

        if analytic_data.data_type in (AnalyticsDataType.PARAMETER, AnalyticsDataType.PARAMETERS):
            for k, v in (
                analytic_data.value.items()
                if analytic_data.data_type == AnalyticsDataType.PARAMETERS
                else [(analytic_data.tag, analytic_data.value)]
            ):
                new_data = _create_new_data(k, v, analytic_data.sender)
                if new_data is None:
                    self.log_warning(fl_ctx, f"Entry {k} of type {type(v)} is not supported.", fire_event=False)
                else:
                    records.append(new_data)
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

        self.log_debug(
            fl_ctx,
            f"try to save data {analytic_data} from {record_origin}",
            fire_event=False,
        )
        data_records = self._convert_to_records(analytic_data, fl_ctx)

        for data_record in data_records:
            func_name = FUNCTION_MAPPING.get(data_record.data_type, None)
            if func_name is None:
                self.log_warning(fl_ctx, f"The data_type {data_record.data_type} is not supported.", fire_event=False)
                return

            with writer.as_default():
                # TF's summary writer uses a different API style
                tf.summary.scalar(data_record.tag, data_record.value, step=data_record.step or 0)
                writer.flush()

    def finalize(self, fl_ctx: FLContext):
        for writer in self.writers_table.values():
            writer.flush()
