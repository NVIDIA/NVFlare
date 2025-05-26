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

from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.edge.aggregators.model_update_dxo_factory import ModelUpdateDXOAggrFactory
from nvflare.edge.assessors.model_update import ModelUpdateAssessor
from nvflare.edge.edge_job import EdgeJob
from nvflare.edge.models.model import Cifar10ConvNet
from nvflare.edge.widgets.evaluator import GlobalEvaluator

job = EdgeJob(
    name="cifar10_sync_job",
    edge_method="cnn",
)

factory = ModelUpdateDXOAggrFactory()
job.configure_client(
    aggregator_factory=factory,
    max_model_versions=3,
    update_timeout=300.0,
    simulation_config_file="configs/cifar10_sync_config.json",
)

evaluator = GlobalEvaluator(
    model_path="nvflare.edge.models.model.Cifar10ConvNet",
    torchvision_dataset={"name": "CIFAR10", "path": "/tmp/nvflare/datasets/cifar10"},
)
job.to_server(evaluator, id="evaluator")

persistor = PTFileModelPersistor(model=Cifar10ConvNet())
job.to_server(persistor, id="persistor")

assessor = ModelUpdateAssessor(
    persistor_id="persistor",
    max_model_version=10,
    max_model_history=1,
    num_updates_for_model=16,
    device_selection_size=16,
    min_hole_to_fill=16,
    device_reuse=True,
    const_selection=True,
)
job.configure_server(
    assessor=assessor,
)

job.export_job("/tmp/nvflare/jobs/")
