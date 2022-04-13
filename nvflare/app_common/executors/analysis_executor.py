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

import glob
import os

import numpy as np

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.workflows.analysis_controller import SupportedTasks


class AnalysisExecutor(Executor):
    def __init__(
        self,
        data_root: str = "./data",
        data_list_key: str = "data",
        min_images: int = 10,
        n_bins: int = 256,
        range_min: float = 0.0,
        range_max: float = 255.0,  # assumes BMP/PNG/JPEGSs as input
        supported_task: str = SupportedTasks.HISTOGRAM,
    ):
        """Executor for federated analysis.

        Args:
            data_root: directory with local image data.
            data_list_key: data list key to use.
            min_images: minimum number of images needed to compute histogram (if less images are included in the computation, no statistics will be sent to the server).
            n_bins: number of bins for the histogram.
            range_min: minimum intensity value to include in the histogram.
            range_max: maximum intensity value to include in the histogram.
            supported_task: task name for histogram computation.
        Returns:
            a Shareable with the computed statistics after `execute()`
        """
        super().__init__()
        self.data_list_key = data_list_key
        self.data_root = data_root
        self.data = None

        self.n_bins = n_bins
        self.range_min = range_min
        self.range_max = range_max

        self.supported_task = supported_task

        self._min_images = min_images

        # needs to be provided by derived class
        self.data_list = None
        self.loader = None

    def load_data_list(self, fl_ctx: FLContext):
        self.log_warning("Data list loading not implemented.")
        pass

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        client_name = fl_ctx.get_prop(ReservedKey.CLIENT_NAME)
        if self.loader is None:
            self.log_error(fl_ctx, f"No loader function specified for client {client_name}!")
            return make_reply(ReturnCode.ERROR)

        self.load_data_list(fl_ctx=fl_ctx)
        if self.data_list is None:
            self.log_error(fl_ctx, f"No data list for client {client_name} loaded!")
            return make_reply(ReturnCode.ERROR)

        self.log_info(fl_ctx, f"Executing {task_name}")
        try:
            if task_name == self.supported_task:
                result_dict = self._compute_histo(fl_ctx, abort_signal)
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                if result_dict:
                    dxo = DXO(data_kind=DataKind.METRICS, data=result_dict)
                    return dxo.to_shareable()
                else:
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            else:
                self.log_error(fl_ctx, f"{task_name} is not a supported task.")
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except BaseException as e:
            self.log_exception(fl_ctx, f"Task {task_name} failed. Exception: {e.__str__()}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def _compute_histo(self, fl_ctx: FLContext, abort_signal: Signal):
        n_images = len(self.data_list)
        n_included_images = 0
        bin_edges = []

        if n_images < self._min_images:  # don't send stats if too few cases available
            return {
                "n_images": n_images,
                "n_included_images": n_included_images,
            }

        histogram = np.zeros((self.n_bins,), dtype=np.int64)
        for i, entry in enumerate(self.data_list):  # TODO: use multi-processing
            if abort_signal.triggered:
                return None
            file = entry.get("image")

            try:
                img, meta = self.loader(file)
                curr_histogram, bin_edges = np.histogram(img, bins=self.n_bins, range=(self.range_min, self.range_max))
                histogram += curr_histogram
                n_included_images += 1

                if i % 100 == 0:
                    self.log_info(fl_ctx, f"adding {i + 1} of {len(self.data_list)}: {file}")
            except BaseException as e:
                self.log_exception(
                    fl_ctx, f"Failed to load file {file} with exception: {e.__str__()}. Skipping this image..."
                )

        if n_included_images < self._min_images:  # don't send stats if too few cases were included
            return {
                "n_images": n_images,
                "n_included_images": n_included_images,
            }

        self.log_info(fl_ctx, f"Computed histogram for {n_included_images} of {n_images} images.")

        return {
            "n_images": n_images,
            "n_included_images": n_included_images,
            "histogram": histogram,
            "bin_edges": bin_edges,
        }
