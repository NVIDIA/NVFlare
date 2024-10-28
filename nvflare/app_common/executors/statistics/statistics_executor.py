# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.dxo import DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.executors.error_handling_executor import ErrorHandlingExecutor
from nvflare.app_common.executors.statistics.statistics_task_handler import StatisticsTaskHandler

"""
    StatisticsExecutor is client-side executor that perform local statistics generation and communication to
    FL Server global statistics controller.
    The actual local statistics calculation would delegate to Statistics spec implementor.
"""


class StatisticsExecutor(ErrorHandlingExecutor):
    def __init__(
        self,
        generator_id: str,
        precision=4,
    ):
        """

        Args:
            generator_id: id of the statistics component
            precision: number of precision digits
        """
        super().__init__()
        self.generator_id = generator_id
        self.precision = precision

    def get_data_kind(self) -> str:
        return DataKind.STATISTICS

    def get_task_handler(self, fl_ctx: FLContext):
        task_handler = StatisticsTaskHandler(self.generator_id, self.precision)
        task_handler.initialize(fl_ctx)
        return task_handler
