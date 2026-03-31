# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Optional

import numpy as np
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.plugins.text.summary_v2 import text_pb
from tensorboard.summary.writer.event_file_writer import EventFileWriter


def _create_scalar_summary(tag: str, value: float) -> Summary:
    return Summary(value=[Summary.Value(tag=tag, simple_value=float(value))])


def _convert_image_to_hwc(value) -> np.ndarray:
    """Normalize HW/HWC/CHW image inputs to HWC uint8 for TensorBoard encoding.

    Float inputs are treated like TensorBoard image helpers typically do: values are
    expected in [0, 1] and scaled up to [0, 255] before PNG encoding. Callers with
    float images already expressed in [0, 255] should convert to uint8 first.
    """

    image = np.asarray(value)
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    elif image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))

    if image.ndim != 3 or image.shape[2] not in (1, 3, 4):
        raise ValueError(f"Expect image to have shape HW, HWC, or CHW with 1/3/4 channels, but got {image.shape}")

    # Match TensorBoard's common convention: non-uint8 arrays are treated as normalized images.
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


class TensorBoardEventWriter:
    """Framework-neutral TensorBoard writer backed by tensorboard's EventFileWriter.

    The standalone tensorboard package exposes the low-level event-file writer, but
    not a high-level SummaryWriter that avoids importing PyTorch or TensorFlow.
    This adapter preserves the TensorBoard-like methods used by TBAnalyticsReceiver
    while keeping the dependency surface limited to tensorboard.
    """

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
