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

from src.net import Net

from nvflare import FedJob
from nvflare.apis.analytix import ANALYTIC_EVENT_TYPE
from nvflare.app_common.widgets.convert_to_fed_event import ConvertToFedEvent
from nvflare.app_common.workflows.cross_site_eval import CrossSiteEval
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.job_config.model import PTModel
from nvflare.app_opt.tracking.wandb.wandb_receiver import WandBReceiver
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--script", type=str, default="src/train_script.py")
    parser.add_argument("--launch_external_process", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--streamed_to_clients",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="whether to stream to NVFlare client or not",
    )
    parser.add_argument(
        "--streamed_to_server",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to stream to NVFlare server or not",
    )
    parser.add_argument("--export_config", action=argparse.BooleanOptionalAction, default=False)

    return parser.parse_args()


def main():
    # define local parameters
    args = define_parser()

    n_clients = args.n_clients
    num_rounds = args.num_rounds
    script = args.script
    launch_external_process = args.launch_external_process
    streamed_to_clients = args.streamed_to_clients
    streamed_to_server = args.streamed_to_server
    export_config = args.export_config

    job = FedJob(
        name="wandb",
    )
    comp_ids = job.to(PTModel(Net()), "server")
    controller = FedAvg(
        num_clients=n_clients,
        num_rounds=num_rounds,
        persistor_id=comp_ids["persistor_id"],
    )
    job.to(controller, "server")
    controller = CrossSiteEval(
        persistor_id=comp_ids["persistor_id"],
    )
    job.to(controller, "server")
    wandb_args = {
        "project": "wandb-experiment",
        "name": "wandb",
        "notes": "Federated Experiment tracking with W&B \n Example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) to train an image classifier using federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629)) and [PyTorch](https://pytorch.org/) as the deep learning training framework. This example also highlights the Flare streaming capability from the clients to the server and deliver to WandB.\\n\\n> **_NOTE:_** \\n This example uses the *[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset and will load its data within the trainer code.\n",
        "tags": ["baseline"],
        "job_type": "train-validate",
        "config": {"architecture": "CNN", "dataset_id": "CIFAR10", "optimizer": "SGD"},
    }
    if streamed_to_server:
        job.to(
            WandBReceiver(
                mode="online",
                wandb_args=wandb_args,
                # events coming from clients are fed events
                events=[f"fed.{ANALYTIC_EVENT_TYPE}"],
            ),
            "server",
        )

    for i in range(n_clients):
        site_name = f"site-{i + 1}"
        executor = ScriptRunner(
            script=script,
            launch_external_process=launch_external_process,
            framework=FrameworkType.PYTORCH,
        )
        job.to(executor, site_name)
        job.to(ConvertToFedEvent(events_to_convert=[ANALYTIC_EVENT_TYPE]), site_name)

        if streamed_to_clients:
            job.to(
                WandBReceiver(
                    mode="online",
                    wandb_args=wandb_args,
                    # events directly fire and handled in client
                    events=[ANALYTIC_EVENT_TYPE],
                ),
                site_name,
            )

    if export_config:
        export_dir = "/tmp/nvflare/jobs/job_config"
        print(f"job exported to {export_dir}")
        job.export_job(export_dir)
    else:
        job.simulator_run("/tmp/nvflare/jobs/workdir", gpu="0")


if __name__ == "__main__":
    main()
