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
import json
import os.path
from typing import Optional

from nvflare.edge.assessor import Assessor
from nvflare.edge.controllers.sage import ScatterAndGatherForEdge
from nvflare.edge.executors.edge_model_executor import EdgeModelExecutor
from nvflare.edge.simulation.device_task_processor import DeviceTaskProcessor
from nvflare.edge.updaters.emd import AggregatorFactory
from nvflare.edge.widgets.etr import EdgeTaskReceiver
from nvflare.edge.widgets.tp_runner import TPRunner
from nvflare.edge.widgets.tpo_runner import TPORunner
from nvflare.fuel.utils.validation_utils import check_object_type, check_positive_int, check_positive_number, check_str
from nvflare.job_config.api import FedJob
from nvflare.job_config.file_source import FileSource


class EdgeJob(FedJob):

    def __init__(
        self,
        name: str,
        edge_method: str,
        min_clients: int = 1,
    ):
        """Constructor of EdgeJob

        Args:
            name: name of the job.
            edge_method: method for matching job request. Goes to the job's meta.
            min_clients: min number of clients required for the job.
        """
        check_str("edge_method", edge_method)

        FedJob.__init__(self, name=name, min_clients=min_clients, meta_props={"edge_method": edge_method})

        self.server_config_added = False
        self.client_config_added = False
        self.simulation_set = False

    def configure_server(
        self,
        assessor: Assessor,
        num_rounds: int = 1,
        task_name: str = "train",
        assess_interval: float = 0.5,
        update_interval: float = 1.0,
    ):
        """Set up server config.

        Args:
            assessor: The Assessor object for assessing workflow progress.
            num_rounds: number of rounds.
            task_name: name of the task.
            assess_interval: how often to perform assessment.
            update_interval: how often the clients should send updates.

        Returns: None

        """
        if self.server_config_added:
            raise RuntimeError("server config is already added")

        check_object_type("assessor", assessor, Assessor)
        check_positive_int("num_rounds", num_rounds)
        check_str("task_name", task_name)

        assessor_id = self.to_server(assessor, id="wf_assessor")

        controller = ScatterAndGatherForEdge(
            assessor_id=assessor_id,
            num_rounds=num_rounds,
            task_name=task_name,
            task_check_period=0.5,
            assess_interval=assess_interval,
            update_interval=update_interval,
        )

        self.to_server(controller, id="sage")
        self.server_config_added = True

    def configure_client(
        self,
        aggregator_factory: AggregatorFactory,
        max_model_versions: Optional[int] = None,
        update_timeout=5.0,
        executor_task_name="train",
        simulation_config_file: str = None,
    ):
        """Set up client config.

        Args:
            aggregator_factory: an AggregatorFactory object to create aggregators when needed.
            max_model_versions: max number of model versions to keep.
            update_timeout: timeout for status update messages.
            executor_task_name: task name for executor.
            simulation_config_file: config file for local simulation (optional).

        Returns: None

        """
        if self.client_config_added:
            raise RuntimeError("client config is already added")

        # check the validity of max_model_versions if not None
        if max_model_versions:
            check_positive_int("max_model_versions", max_model_versions)

        check_object_type("aggregator_factory", aggregator_factory, AggregatorFactory)
        check_positive_number("update_timeout", update_timeout)
        check_str("executor_task_name", executor_task_name)

        if simulation_config_file:
            check_str("simulation_config_file", simulation_config_file)

        self.to_clients(EdgeTaskReceiver(), id="edge_task_receiver")

        aggr_factory_id = self.to_clients(aggregator_factory, id="aggr_factory")
        executor = self._configure_executor(
            aggr_factory_id=aggr_factory_id, max_model_versions=max_model_versions, update_timeout=update_timeout
        )
        self.to_clients(executor, id="executor", tasks=[executor_task_name])

        if simulation_config_file:
            self.configure_simulation_with_file(simulation_config_file)

        self.client_config_added = True

    def _configure_executor(self, aggr_factory_id, max_model_versions, update_timeout):
        return EdgeModelExecutor(
            aggr_factory_id=aggr_factory_id, max_model_versions=max_model_versions, update_timeout=update_timeout
        )

    def configure_simulation_with_file(self, simulation_config_file: str):
        """Configure simulation with a config file.

        Args:
            simulation_config_file: the simulation config file.

        Returns:

        """
        if self.simulation_set:
            raise RuntimeError("simulation is already configured")

        if not os.path.isfile(simulation_config_file):
            raise ValueError(f"file {simulation_config_file} does not exist or is not a valid file")

        try:
            with open(simulation_config_file, "r") as f:
                json.load(f)
        except Exception as ex:
            raise ValueError(f"file {simulation_config_file} is not a valid JSON file: {ex}")

        self.to_clients(FileSource(simulation_config_file, app_folder_type="config"))

        base_name = os.path.basename(simulation_config_file)
        conf_file = "{JOB_CONFIG_DIR}/" + f"{base_name}"
        self.to_clients(TPRunner(conf_file), id="tp_runner")
        self.simulation_set = True

    def configure_simulation(
        self,
        task_processor: DeviceTaskProcessor,
        job_timeout: float = 60.0,
        num_devices: int = 1000,
        num_workers: int = 10,
    ):
        """Configure simulation with a DeviceTaskProcessor.

        Args:
            task_processor: the DeviceTaskProcessor object to be used for processing tasks.
            job_timeout: timeout for trying to get job.
            num_devices: number of devices to simulate.
            num_workers: number of workers for executing tasks.

        Returns: None

        """
        if self.simulation_set:
            raise RuntimeError("simulation is already configured")

        tp_id = self.to_clients(task_processor, "task_processor")
        runner = TPORunner(tp_id, job_timeout, num_devices, num_workers)
        self.to_clients(runner, "tpo_runner")
        self.simulation_set = True
