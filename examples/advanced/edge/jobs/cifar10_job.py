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

import argparse

from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.edge.aggregators.model_update_dxo_factory import ModelUpdateDXOAggrFactory
from nvflare.edge.assessors.buff_device_manager import BuffDeviceManager
from nvflare.edge.assessors.buff_model_manager import BuffModelManager
from nvflare.edge.assessors.model_update import ModelUpdateAssessor
from nvflare.edge.edge_job import EdgeJob
from nvflare.edge.models.model import Cifar10ConvNet
from nvflare.edge.widgets.evaluator import GlobalEvaluator


def export_job(args):
    job = EdgeJob(
        name=args.job_name,
        edge_method="cnn",
    )

    factory = ModelUpdateDXOAggrFactory()
    if args.simulation_config_file:
        job.configure_client(
            aggregator_factory=factory,
            max_model_versions=args.max_model_aggr,
            update_timeout=300.0,
            simulation_config_file=args.simulation_config_file,
        )
    else:
        job.configure_client(
            aggregator_factory=factory,
            max_model_versions=args.max_model_aggr,
            update_timeout=300.0,
        )

    evaluator = GlobalEvaluator(
        model_path="nvflare.edge.models.model.Cifar10ConvNet",
        torchvision_dataset={"name": "CIFAR10", "path": "/tmp/nvflare/datasets/cifar10"},
        eval_frequency=args.eval_frequency,
    )
    job.to_server(evaluator, id="evaluator")

    # add persistor, model_manager, and device_manager
    persistor = PTFileModelPersistor(model=Cifar10ConvNet())
    persistor_id = job.to_server(persistor, id="persistor")

    model_manager = BuffModelManager(
        num_updates_for_model=args.num_updates_for_model,
        max_model_version=args.max_model_version,
        max_model_history=args.max_model_history,
        global_lr=args.global_lr,
        staleness_weight=args.staleness_weight,
    )
    model_manager_id = job.to_server(model_manager, id="model_manager")

    device_manager = BuffDeviceManager(
        device_selection_size=args.device_selection_size,
        min_hole_to_fill=args.min_hole_to_fill,
        device_reuse=args.device_reuse,
        const_selection=args.const_selection,
    )
    device_manager_id = job.to_server(device_manager, id="device_manager")

    # add model_update_assessor
    assessor = ModelUpdateAssessor(
        persistor_id=persistor_id,
        model_manager_id=model_manager_id,
        device_manager_id=device_manager_id,
    )
    job.configure_server(
        assessor=assessor,
    )

    job.export_job("/tmp/nvflare/jobs/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export CIFAR10 EdgeJob")
    parser.add_argument("--job_name", help="Name of the job to export")
    parser.add_argument("--simulation_config_file", help="Path to the simulation config file, leaf job if provided")
    parser.add_argument("--max_model_aggr", type=int, default=3, help="Maximum number of model aggregations")
    parser.add_argument(
        "--max_model_version", type=int, default=10, help="Maximum number of model versions to generate"
    )
    parser.add_argument("--max_model_history", type=int, default=1, help="Maximum number of model history to keep")
    parser.add_argument("--num_updates_for_model", type=int, default=16, help="Number of updates for each model")
    parser.add_argument(
        "--device_selection_size", type=int, default=16, help="Number of devices to select for each update"
    )
    parser.add_argument("--global_lr", type=float, default=1.0, help="Global learning rate for model updates")
    parser.add_argument("--staleness_weight", action="store_true", help="Enable staleness weighting for model updates")
    parser.add_argument("--min_hole_to_fill", type=int, default=16, help="Minimum hole to fill in the model")
    parser.add_argument("--device_reuse", action="store_true", help="Enable device reuse")
    parser.add_argument("--const_selection", action="store_true", help="Enable constant selection for device updates")
    parser.add_argument(
        "--eval_frequency", type=int, default=1, help="Frequency of evaluation in terms of model updates"
    )
    args = parser.parse_args()
    export_job(args)
