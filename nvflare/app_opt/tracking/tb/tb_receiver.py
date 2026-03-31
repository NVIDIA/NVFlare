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

import io
import os
import time
from typing import List, Optional

import numpy as np
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.plugins.text.summary_v2 import text_pb
from tensorboard.summary.writer.event_file_writer import EventFileWriter

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
    AnalyticsDataType.METRIC: "add_scalar",
    AnalyticsDataType.METRICS: "add_scalars",
}


def _create_scalar_summary(tag: str, value: float) -> Summary:
    return Summary(value=[Summary.Value(tag=tag, simple_value=float(value))])


def _convert_image_to_hwc(value) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    elif image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))

    if image.ndim != 3 or image.shape[2] not in (1, 3, 4):
        raise ValueError(f"Expect image to have shape HW, HWC, or CHW with 1/3/4 channels, but got {image.shape}")

    scale_factor = 1 if image.dtype == np.uint8 else 255
    image = image.astype(np.float32)
    image = (image * scale_factor).clip(0, 255).astype(np.uint8)
    return image


def _create_image_summary(tag: str, value) -> Summary:
    from PIL import Image

    image = _convert_image_to_hwc(value)
    height, width, channels = image.shape
    encoded_image = image.squeeze(axis=2) if channels == 1 else image

    output = io.BytesIO()
    Image.fromarray(encoded_image).save(output, format="PNG")

    summary_image = Summary.Image(
        height=height,
        width=width,
        colorspace=channels,
        encoded_image_string=output.getvalue(),
    )
    return Summary(value=[Summary.Value(tag=tag, image=summary_image)])


class _EventTBWriter:
    """Write TensorBoard event files without depending on a specific ML framework."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.writer = EventFileWriter(log_dir)
        self.scalar_writers = {}

    def add_scalar(self, tag: str, scalar_value: float, global_step: Optional[int] = None):
        self._add_summary(self.writer, _create_scalar_summary(tag, scalar_value), global_step)

    def add_text(self, tag: str, text_string: str, global_step: Optional[int] = None):
        self._add_summary(self.writer, text_pb(tag, text_string), global_step)

    def add_image(self, tag: str, img_tensor, global_step: Optional[int] = None):
        self._add_summary(self.writer, _create_image_summary(tag, img_tensor), global_step)

    def add_scalars(self, main_tag: str, tag_scalar_dict: dict, global_step: Optional[int] = None):
        for tag, scalar_value in tag_scalar_dict.items():
            writer_key = f"{main_tag.replace('/', '_')}_{tag}"
            writer = self.scalar_writers.get(writer_key)
            if writer is None:
                writer = EventFileWriter(os.path.join(self.log_dir, writer_key))
                self.scalar_writers[writer_key] = writer
            self._add_summary(writer, _create_scalar_summary(main_tag, scalar_value), global_step)

    def flush(self):
        self.writer.flush()
        for writer in self.scalar_writers.values():
            writer.flush()

    def close(self):
        self.writer.close()
        for writer in self.scalar_writers.values():
            writer.close()

    @staticmethod
    def _add_summary(writer: EventFileWriter, summary: Summary, global_step: Optional[int] = None):
        event = Event(wall_time=time.time(), summary=summary)
        if global_step is not None:
            event.step = global_step
        writer.add_event(event)


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

              - site_workspace
                - job_id1 (already created):
                  - output_dir (default: tb_events):
                    - peer_name_1:
                    - peer_name_2:

                - job_id2 (already created):
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
            writer = _EventTBWriter(log_dir=peer_log_dir)
            self.writers_table[record_origin] = writer

        # do different things depending on the type in dxo
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

            func = getattr(writer, func_name)
            if data_record.step is not None:
                func(data_record.tag, data_record.value, data_record.step)
            else:
                func(data_record.tag, data_record.value)

    def finalize(self, fl_ctx: FLContext):
        for writer in self.writers_table.values():
            writer.flush()
            writer.close()
