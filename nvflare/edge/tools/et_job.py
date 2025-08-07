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
from nvflare.edge.executors.et_edge_model_executor import ETEdgeModelExecutor
from nvflare.edge.models.model import DeviceModel

from .edge_job import EdgeJob


class ETJob(EdgeJob):

    def __init__(
        self,
        name: str,
        edge_method: str,
        device_model: DeviceModel,
        input_shape,
        output_shape,
        min_clients: int = 1,
    ):
        """Constructor of EdgeJob

        Args:
            name: name of the job.
            edge_method: method for matching job request. Goes to the job's meta.
            min_clients: min number of clients required for the job.
        """
        EdgeJob.__init__(self, name, edge_method, min_clients)
        self.device_model = device_model
        self.input_shape = input_shape
        self.output_shape = output_shape

    def _configure_executor(self, aggr_factory_id, max_model_versions, update_timeout):
        return ETEdgeModelExecutor(
            et_model=self.device_model,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            aggr_factory_id=aggr_factory_id,
            max_model_versions=max_model_versions,
            update_timeout=update_timeout,
        )
