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
from pt.learner_with_mlflow import PTLearner
from pt.simple_network import SimpleNetwork

from nvflare.app_common.executors.learner_executor import LearnerExecutor
from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval
from nvflare.app_opt.pt.job_config.fed_sag_mlflow import SAGMLFlowJob

if __name__ == "__main__":
    n_clients = 2
    num_rounds = 1

    job = SAGMLFlowJob(
        name="hello-pt-mlflow",
        n_clients=n_clients,
        num_rounds=num_rounds,
        initial_model=SimpleNetwork(),
        tracking_uri="file:///{WORKSPACE}/{JOB_ID}/mlruns",
        kwargs={
            "experiment_name": "hello-pt-experiment",
            "run_name": "hello-pt-with-mlflow",
            "experiment_tags": {"mlflow.note.content": "## **Hello PyTorch experiment with MLflow**"},
            "run_tags": {
                "mlflow.note.content": "## Federated Experiment tracking with MLflow \n### Example of using **[NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html)** to train an image classifier using federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629)) and [PyTorch](https://pytorch.org/) as the deep learning training framework. This example also highlights the Flare streaming capability from the clients to the server for server delivery to MLflow.\n\n> **_NOTE:_** \n This example uses the *[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)* dataset and will load its data within the trainer code.\n"
            },
        },
        artifact_location="artifacts",
    )

    ctrl = CrossSiteModelEval()

    job.to(ctrl, "server")

    for i in range(n_clients):
        site_name = f"site-{i + 1}"
        learner_id = job.to(
            PTLearner(epochs=5, lr=0.01, analytic_sender_id="log_writer"),
            site_name,
            id="pt_learner",
        )
        executor = LearnerExecutor(learner_id=learner_id)
        job.to(executor, site_name, tasks=["train", "submit_model", "validate"])

    job.export_job("./jobs/job_config")
    job.simulator_run("./jobs/workdir")
