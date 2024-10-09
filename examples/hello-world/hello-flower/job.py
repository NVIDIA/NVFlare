# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from argparse import ArgumentParser

from nvflare.app_opt.flower.flower_pt_job import FlowerPyTorchJob
from nvflare.client.api import ClientAPIType
from nvflare.client.api_spec import CLIENT_API_TYPE_KEY


def main():
    parser = ArgumentParser()
    parser.add_argument("--job_name", type=str, required=True)
    parser.add_argument("--content_dir", type=str, required=True)
    parser.add_argument("--stream_metrics", action="store_true")
    parser.add_argument("--use_client_api", action="store_true")
    parser.add_argument("--export_dir", type=str, default="jobs")
    parser.add_argument("--workdir", type=str, default="/tmp/nvflare/hello-flower")
    args = parser.parse_args()

    env = {}
    if args.stream_metrics or args.use_client_api:
        # needs to init client api to stream metrics
        # only external client api works with the current flower integration
        env = {CLIENT_API_TYPE_KEY: ClientAPIType.EX_PROCESS_API.value}

    job = FlowerPyTorchJob(
        name=args.job_name,
        flower_content=args.content_dir,
        stream_metrics=args.stream_metrics,
        extra_env=env,
    )

    job.export_job(args.export_dir)
    job.simulator_run(args.workdir, gpu="0", n_clients=2)


if __name__ == "__main__":
    main()
