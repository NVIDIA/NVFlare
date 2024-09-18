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
from typing import Any, List, Optional

from nvflare.apis.executor import Executor
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.abstract.metric_comparator import MetricComparator
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.abstract.shareable_generator import ShareableGenerator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.ccwf.common import Constant, CyclicOrder
from nvflare.fuel.utils.validation_utils import check_object_type
from nvflare.job_config.api import FedJob, has_add_to_job_method
from nvflare.widgets.widget import Widget

from .cse_client_ctl import CrossSiteEvalClientController
from .cse_server_ctl import CrossSiteEvalServerController
from .cyclic_client_ctl import CyclicClientController
from .cyclic_server_ctl import CyclicServerController
from .swarm_client_ctl import SwarmClientController
from .swarm_server_ctl import SwarmServerController


class SwarmServerConfig:
    def __init__(
        self,
        num_rounds: int,
        start_round: int = 0,
        start_task_timeout=Constant.START_TASK_TIMEOUT,
        configure_task_timeout=Constant.CONFIG_TASK_TIMEOUT,
        participating_clients=None,
        result_clients=None,
        starting_client: str = "",
        max_status_report_interval: float = Constant.PER_CLIENT_STATUS_REPORT_TIMEOUT,
        progress_timeout: float = Constant.WORKFLOW_PROGRESS_TIMEOUT,
        private_p2p: bool = True,
        aggr_clients=None,
        train_clients=None,
    ):
        self.num_rounds = num_rounds
        self.start_round = start_round
        self.start_task_timeout = start_task_timeout
        self.configure_task_timeout = configure_task_timeout
        self.participating_clients = participating_clients
        self.result_clients = result_clients
        self.starting_client = starting_client
        self.max_status_report_interval = max_status_report_interval
        self.progress_timeout = progress_timeout
        self.private_p2p = private_p2p
        self.aggr_clients = aggr_clients
        self.train_clients = train_clients


class SwarmClientConfig:
    def __init__(
        self,
        executor: Any,
        persistor: Any,
        shareable_generator: Any,
        aggregator: Any,
        metric_comparator: Any = None,
        model_selector: Any = None,
        learn_task_check_interval=Constant.LEARN_TASK_CHECK_INTERVAL,
        learn_task_abort_timeout=Constant.LEARN_TASK_ABORT_TIMEOUT,
        learn_task_ack_timeout=Constant.LEARN_TASK_ACK_TIMEOUT,
        learn_task_timeout=None,
        final_result_ack_timeout=Constant.FINAL_RESULT_ACK_TIMEOUT,
        min_responses_required: int = 1,
        wait_time_after_min_resps_received: float = 10.0,
    ):
        # the executor could be a wrapper object that adds real Executor when added to job!
        validate_object_for_job("executor", executor, Executor)
        validate_object_for_job("persistor", persistor, ModelPersistor)
        validate_object_for_job("shareable_generator", shareable_generator, ShareableGenerator)
        validate_object_for_job("aggregator", aggregator, Aggregator)

        if model_selector:
            validate_object_for_job("model_selector", model_selector, Widget)

        if metric_comparator:
            validate_object_for_job("metric_comparator", metric_comparator, MetricComparator)

        self.executor = executor
        self.persistor = persistor
        self.shareable_generator = shareable_generator
        self.aggregator = aggregator
        self.metric_comparator = metric_comparator
        self.model_selector = model_selector
        self.learn_task_check_interval = learn_task_check_interval
        self.learn_task_abort_timeout = learn_task_abort_timeout
        self.learn_task_ack_timeout = learn_task_ack_timeout
        self.learn_task_timeout = learn_task_timeout
        self.final_result_ack_timeout = final_result_ack_timeout
        self.min_responses_required = min_responses_required
        self.wait_time_after_min_resps_received = wait_time_after_min_resps_received


class CyclicServerConfig:
    def __init__(
        self,
        num_rounds: int,
        start_task_timeout=Constant.START_TASK_TIMEOUT,
        configure_task_timeout=Constant.CONFIG_TASK_TIMEOUT,
        participating_clients=None,
        result_clients=None,
        starting_client: str = "",
        max_status_report_interval: float = Constant.PER_CLIENT_STATUS_REPORT_TIMEOUT,
        progress_timeout: float = Constant.WORKFLOW_PROGRESS_TIMEOUT,
        private_p2p: bool = True,
        cyclic_order: str = CyclicOrder.FIXED,
    ):
        self.num_rounds = num_rounds
        self.start_task_timeout = start_task_timeout
        self.configure_task_timeout = configure_task_timeout
        self.participating_clients = participating_clients
        self.result_clients = result_clients
        self.starting_client = starting_client
        self.max_status_report_interval = max_status_report_interval
        self.progress_timeout = progress_timeout
        self.private_p2p = private_p2p
        self.cyclic_order = cyclic_order


class CyclicClientConfig:
    def __init__(
        self,
        executor: Any,
        persistor: Any,
        shareable_generator: Any,
        learn_task_abort_timeout=Constant.LEARN_TASK_ABORT_TIMEOUT,
        learn_task_ack_timeout=Constant.LEARN_TASK_ACK_TIMEOUT,
        final_result_ack_timeout=Constant.FINAL_RESULT_ACK_TIMEOUT,
    ):
        validate_object_for_job("executor", executor, Executor)
        validate_object_for_job("persistor", persistor, ModelPersistor)
        validate_object_for_job("shareable_generator", shareable_generator, ShareableGenerator)

        self.executor = executor
        self.persistor = persistor
        self.shareable_generator = shareable_generator
        self.learn_task_abort_timeout = learn_task_abort_timeout
        self.learn_task_ack_timeout = learn_task_ack_timeout
        self.final_result_ack_timeout = final_result_ack_timeout


class CrossSiteEvalConfig:
    def __init__(
        self,
        start_task_timeout=Constant.START_TASK_TIMEOUT,
        configure_task_timeout=Constant.CONFIG_TASK_TIMEOUT,
        eval_task_timeout=30,
        progress_timeout: float = Constant.WORKFLOW_PROGRESS_TIMEOUT,
        private_p2p: bool = True,
        participating_clients=None,
        evaluators=None,
        evaluatees=None,
        global_model_client=None,
        max_status_report_interval: float = Constant.PER_CLIENT_STATUS_REPORT_TIMEOUT,
        eval_result_dir=AppConstants.CROSS_VAL_DIR,
        get_model_timeout=Constant.GET_MODEL_TIMEOUT,
    ):
        self.start_task_timeout = start_task_timeout
        self.configure_task_timeout = configure_task_timeout
        self.eval_task_timeout = eval_task_timeout
        self.progress_timeout = progress_timeout
        self.private_p2p = private_p2p
        self.participating_clients = participating_clients
        self.evaluators = evaluators
        self.evaluatees = evaluatees
        self.global_model_client = global_model_client
        self.max_status_report_interval = max_status_report_interval
        self.eval_result_dir = eval_result_dir
        self.get_model_timeout = get_model_timeout


class CCWFJob(FedJob):
    def __init__(
        self,
        name: str = "fed_job",
        min_clients: int = 1,
        mandatory_clients: Optional[List[str]] = None,
        external_resources: Optional[str] = None,
    ):
        """Client-Controlled Workflow Job.

        Provides methods for adding client-controlled swarm learning, cyclic, and cross-site evaluation workflows.

        Args:
            name (name, optional): name of the job. Defaults to "fed_job"
            min_clients (int, optional): the minimum number of clients for the job. Defaults to 1.
            mandatory_clients (List[str], optional): mandatory clients to run the job. Default None.
            external_resources (str, optional): External resources directory or filename. Defaults to None.
        """
        super().__init__(name, min_clients, mandatory_clients)
        if external_resources:
            self.to_server(external_resources)
            self.to_clients(external_resources)

    def add_swarm(
        self,
        server_config: SwarmServerConfig,
        client_config: SwarmClientConfig,
        cse_config: CrossSiteEvalConfig = None,
    ):
        controller = SwarmServerController(
            num_rounds=server_config.num_rounds,
            start_round=server_config.start_round,
            start_task_timeout=server_config.start_task_timeout,
            configure_task_timeout=server_config.configure_task_timeout,
            participating_clients=server_config.participating_clients,
            result_clients=server_config.result_clients,
            starting_client=server_config.starting_client,
            max_status_report_interval=server_config.max_status_report_interval,
            progress_timeout=server_config.progress_timeout,
            private_p2p=server_config.private_p2p,
            aggr_clients=server_config.aggr_clients,
            train_clients=server_config.train_clients,
        )
        self.to_server(controller)

        metric_comparator_id = None
        if client_config.metric_comparator:
            metric_comparator_id = self.to_clients(client_config.metric_comparator, id="metric_comparator")

        persistor_id = self.to_clients(client_config.persistor, id="persistor")
        shareable_generator_id = self.to_clients(client_config.shareable_generator, id="shareable_generator")
        aggregator_id = self.to_clients(client_config.aggregator, id="aggregator")

        client_controller = SwarmClientController(
            aggregator_id=aggregator_id,
            persistor_id=persistor_id,
            shareable_generator_id=shareable_generator_id,
            metric_comparator_id=metric_comparator_id,
            learn_task_abort_timeout=client_config.learn_task_abort_timeout,
            learn_task_ack_timeout=client_config.learn_task_ack_timeout,
            learn_task_timeout=client_config.learn_task_timeout,
            final_result_ack_timeout=client_config.final_result_ack_timeout,
            min_responses_required=client_config.min_responses_required,
            wait_time_after_min_resps_received=client_config.wait_time_after_min_resps_received,
        )
        self.to_clients(client_controller, tasks=["swarm_*"])
        self.to_clients(client_config.executor, tasks=["train", "validate", "submit_model"])

        if client_config.model_selector:
            self.to_clients(client_config.model_selector, id="model_selector")

        if cse_config:
            self.add_cross_site_eval(cse_config, persistor_id)

    def add_cyclic(
        self,
        server_config: CyclicServerConfig,
        client_config: CyclicClientConfig,
        cse_config: CrossSiteEvalConfig = None,
    ):
        controller = CyclicServerController(
            num_rounds=server_config.num_rounds,
            start_task_timeout=server_config.start_task_timeout,
            configure_task_timeout=server_config.configure_task_timeout,
            participating_clients=server_config.participating_clients,
            result_clients=server_config.result_clients,
            starting_client=server_config.starting_client,
            max_status_report_interval=server_config.max_status_report_interval,
            progress_timeout=server_config.progress_timeout,
            private_p2p=server_config.private_p2p,
            cyclic_order=server_config.cyclic_order,
        )
        self.to_server(controller)

        persistor_id = self.to_clients(client_config.persistor, id="persistor")
        shareable_generator_id = self.to_clients(client_config.shareable_generator, id="shareable_generator")
        client_controller = CyclicClientController(
            persistor_id=persistor_id,
            shareable_generator_id=shareable_generator_id,
            learn_task_abort_timeout=client_config.learn_task_abort_timeout,
            learn_task_ack_timeout=client_config.learn_task_ack_timeout,
            final_result_ack_timeout=client_config.final_result_ack_timeout,
        )
        self.to_clients(client_controller, tasks=["cyclic_*"])
        self.to_clients(client_config.executor, tasks=["train", "validate", "submit_model"])

        if cse_config:
            self.add_cross_site_eval(cse_config, persistor_id)

    def add_cross_site_eval(
        self,
        cse_config: CrossSiteEvalConfig,
        persistor_id: str,
    ):
        controller = CrossSiteEvalServerController(
            start_task_timeout=cse_config.start_task_timeout,
            configure_task_timeout=cse_config.configure_task_timeout,
            eval_task_timeout=cse_config.eval_task_timeout,
            progress_timeout=cse_config.progress_timeout,
            private_p2p=cse_config.private_p2p,
            participating_clients=cse_config.participating_clients,
            evaluators=cse_config.evaluators,
            evaluatees=cse_config.evaluatees,
            global_model_client=cse_config.global_model_client,
            max_status_report_interval=cse_config.max_status_report_interval,
            eval_result_dir=cse_config.eval_result_dir,
        )
        self.to_server(controller)

        client_controller = CrossSiteEvalClientController(
            persistor_id=persistor_id,
            get_model_timeout=cse_config.get_model_timeout,
        )
        self.to_clients(client_controller, tasks=["cse_*"])


def validate_object_for_job(name, obj, obj_type):
    """Check whether the specified object is valid for job.
    The object must either have the add_to_fed_job method or is valid object type.

    Args:
        name: name of the object
        obj: the object to be checked
        obj_type: the object type that the object should be, if it doesn't have the add_to_fed_job method.

    Returns: None

    """
    if has_add_to_job_method(obj):
        return

    check_object_type(name, obj, obj_type)
