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
from add_shareable_parameter import AddShareable
from cifar10trainer import Cifar10Trainer
from cifar10validator import Cifar10Validator
from print_shareable_parameter import PrintShareable
from pt_model_locator import PTModelLocator

from nvflare.apis.dxo import DataKind
from nvflare.app_common.aggregators import InTimeAccumulateWeightedAggregator
from nvflare.app_common.shareablegenerators import FullModelShareableGenerator
from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator
from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval
from nvflare.app_common.workflows.initialize_global_weights import InitializeGlobalWeights
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.app_opt.pt import PTFileModelPersistor
from nvflare.job_config.fed_app_config import ClientAppConfig, FedAppConfig, ServerAppConfig
from nvflare.job_config.fed_job_config import FedJobConfig


class HelloPTJob:
    def __init__(self) -> None:
        super().__init__()
        self.job = self.define_job()

    def define_job(self) -> FedJobConfig:
        # job = FedJobConfig(job_name="hello-pt", min_clients=2, mandatory_clients="site-1")
        job: FedJobConfig = FedJobConfig(job_name="hello-pt", min_clients=2)

        server_app = self._create_server_app()
        client_app = self._create_client_app()

        app = FedAppConfig(server_app=server_app, client_app=client_app)
        job.add_fed_app("app", app)

        # app = FedAppConfig(client_app=client_app)
        # job.add_fed_app("client_app", app)
        # job.set_site_app("server", "app")
        # job.set_site_app("site-1", "app")
        # job.set_site_app("site-2", "client_app")
        # job.add_resource_spec("site-1", {"memory": "8GB"})

        job.set_site_app("@ALL", "app")

        return job

    def _create_client_app(self):
        client_app = ClientAppConfig()
        executor = Cifar10Trainer(lr=0.01, epochs=1)
        client_app.add_executor(["train", "submit_model", "get_weights"], executor)
        validator = Cifar10Validator()
        client_app.add_executor(["validate"], validator)

        task_filter = AddShareable()
        client_app.add_task_result_filter(["train"], task_filter)
        task_filter = PrintShareable()
        client_app.add_task_data_filter(["validate", "train"], task_filter)
        return client_app

    def _create_server_app(self):
        server_app = ServerAppConfig()
        controller = InitializeGlobalWeights(task_name="get_weights")
        server_app.add_workflow("pre_train", controller)
        controller = ScatterAndGather(
            min_clients=2,
            num_rounds=2,
            start_round=0,
            wait_time_after_min_received=10,
            aggregator_id="aggregator",
            persistor_id="persistor",
            shareable_generator_id="shareable_generator",
            train_task_name="train",
            train_timeout=0,
        )
        server_app.add_workflow("scatter_and_gather", controller)
        controller = CrossSiteModelEval(model_locator_id="model_locator")
        server_app.add_workflow("cross_site_validate", controller)

        component = PTFileModelPersistor()
        server_app.add_component("persistor", component)
        component = FullModelShareableGenerator()
        server_app.add_component("shareable_generator", component)
        component = InTimeAccumulateWeightedAggregator(
            expected_data_kind=DataKind.WEIGHTS, aggregation_weights={"site-1": 1.0, "site-2": 1.0}
        )
        server_app.add_component("aggregator", component)
        component = PTModelLocator()
        server_app.add_component("model_locator", component)
        component = ValidationJsonGenerator()
        server_app.add_component("json_generator", component)

        task_filter = AddShareable()
        server_app.add_task_data_filter(["train"], task_filter)
        task_filter = PrintShareable()
        server_app.add_task_result_filter(["validate", "train"], task_filter)
        return server_app

    def export_job(self, job_root):
        self.job.generate_job_config(job_root)

    def simulator_run(self, workspace):
        self.job.simulator_run(workspace, threads=2)


if __name__ == "__main__":
    job = HelloPTJob()

    # job.export_job("/tmp/nvflare/jobs")
    job.simulator_run("/tmp/nvflare/simulator_workspace")
