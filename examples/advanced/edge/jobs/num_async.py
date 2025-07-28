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

from nvflare.edge.aggregators.num_dxo_factory import NumDXOAggrFactory
from nvflare.edge.assessors.async_num import AsyncNumAssessor
from nvflare.edge.edge_job import EdgeJob
from nvflare.edge.simulation.devices.num import NumProcessor


job = EdgeJob(
    name="num_async_job",
    edge_method="cnn",
)

factory = NumDXOAggrFactory()
job.configure_client(
    aggregator_factory=factory, max_model_versions=3,
)

job.configure_simulation(
    task_processor=NumProcessor()
)

job.configure_server(
    assessor=AsyncNumAssessor(
        num_updates_for_model=10,
        max_model_history=3,
        max_model_version=10,
        device_selection_size=30,
    )
)

job.export_job("/tmp/nvflare/jobs/")
