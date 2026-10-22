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

"""Export the same backend probe for CPU/GPU and resident/task execution."""

import argparse
from pathlib import Path

from nvflare.apis.dxo import DataKind
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator
from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor
from nvflare.app_common.np.np_model_persistor import NPModelPersistor
from nvflare.app_common.shareablegenerators.full_model_shareable_generator import FullModelShareableGenerator
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.job_config.api import FedJob


def probe_meta(image, device, gpu_count, lifetime):
    client = {"image": image}
    # The serial resident control gets explicit UUIDs from each site's trusted
    # device_requests. A job count would override those UUIDs with Count=N.
    if device != "cuda" or lifetime != "job":
        client["num_of_gpus"] = gpu_count if device == "cuda" else 0
    return {
        "launcher_spec": {
            "default": {"docker": client},
            "server": {"docker": {"image": image, "num_of_gpus": 0}},
        },
    }


def export_probe(image, device, gpu_count, clients, lifetime, output):
    gpu_count = gpu_count if device == "cuda" else 0
    if clients < 1 or (device == "cuda" and gpu_count < 1):
        raise ValueError("clients and requested CUDA GPU count must be positive")
    job = FedJob(
        name=f"docker-probe-{device}-{lifetime}",
        min_clients=clients,
        execution_lifetime=lifetime,
        meta_props=probe_meta(image, device, gpu_count, lifetime),
    )
    job.to_server(NPModelPersistor(model=[[1.0, 2.0]]), id="persistor")
    job.to_server(InTimeAccumulateWeightedAggregator(expected_data_kind=DataKind.WEIGHTS), id="aggregator")
    job.to_server(FullModelShareableGenerator(), id="shareable_generator")
    job.to_server(ScatterAndGather(min_clients=clients, num_rounds=2, persistor_id="persistor"))
    job.to_clients(
        ClientAPIExecutor(
            execution_mode="in_process",
            task_script_path="compute_probe.py",
            task_script_args=["--device", device, "--gpu-count", str(gpu_count)],
        ),
        tasks=["train"],
    )
    job.add_file_to_clients(str(Path(__file__).with_name("compute_probe.py")))
    job.export_job(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--clients", type=int, default=1)
    parser.add_argument("--lifetime", choices=("job", "task"), default="task")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    gpu_count = args.gpu_count if args.device == "cuda" else 0
    if args.clients < 1 or (args.device == "cuda" and gpu_count < 1):
        parser.error("clients and requested CUDA GPU count must be positive")
    export_probe(args.image, args.device, gpu_count, args.clients, args.lifetime, args.output)


if __name__ == "__main__":
    main()
