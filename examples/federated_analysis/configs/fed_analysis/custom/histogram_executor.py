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

from monai.data import ITKReader, load_decathlon_datalist
from monai.transforms import LoadImage

from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.executors.analysis_executor import AnalysisExecutor
from nvflare.app_common.workflows.analysis_controller import SupportedTasks


class HistogramExecutor(AnalysisExecutor):
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
        super().__init__(
            data_root=data_root,
            data_list_key=data_list_key,
            min_images=min_images,
            n_bins=n_bins,
            range_min=range_min,
            range_max=range_max,
            supported_task=supported_task,
        )

        # set data list and loader for this task
        self.loader = LoadImage()
        self.loader.register(ITKReader())

    def load_data_list(self, fl_ctx: FLContext):
        client_name = fl_ctx.get_prop(ReservedKey.CLIENT_NAME)
        dataset_json = glob.glob(os.path.join(self.data_root, client_name + "*.json"))
        if len(dataset_json) != 1:
            self.log_error(
                fl_ctx, f"No unique matching dataset list found in {self.data_root} for client {client_name}"
            )
            return False
        dataset_json = dataset_json[0]
        self.log_info(fl_ctx, f"Reading data from {dataset_json}")
        self.data_list = load_decathlon_datalist(
            data_list_file_path=dataset_json, data_list_key=self.data_list_key, base_dir=self.data_root
        )
        self.log_info(fl_ctx, f"Client {client_name} has {len(self.data_list)} images")
